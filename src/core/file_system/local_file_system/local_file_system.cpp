/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-10-19
 */

#include "local_file_system.h"

#include <acl/acl_rt.h>
#include <iostream>
#include <dirent.h>
#include <sys/mman.h>
#include <thread>
#include <fcntl.h>

#include "utils/common.h"

using namespace std;
using namespace MxRec;

void LocalFileSystem::CreateDir(const string& dirName)
{
    if (access(dirName.c_str(), F_OK) == -1) {
        if (mkdir(dirName.c_str(), dirMode) == -1) {
            LOG_DEBUG("Unable to create directory: {}", dirName);
        }
    }
}

vector<string> LocalFileSystem::ListDir(const string& dirName)
{
    vector<string> dirs;
    DIR *dir = opendir(dirName.c_str());
    struct dirent* en;
    if (dir == nullptr) {
        LOG_WARN("Open directory {} failed while trying to traverse the directory.", dirName);
        closedir(dir);
        return dirs;
    }

    for (en = readdir(dir); en != nullptr ; en = readdir(dir)) {
        if (strncmp(en->d_name, currDir.c_str(), strlen(currDir.c_str())) != 0 &&
            strncmp(en->d_name, prevDir.c_str(), strlen(prevDir.c_str())) != 0) {
            dirs.emplace_back(en->d_name);
        }
    }
    closedir(dir);
    return dirs;
}

size_t LocalFileSystem::GetFileSize(const string& filePath)
{
    std::ifstream readFile;
    readFile.open(filePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!readFile.is_open()) {
        throw runtime_error(StringFormat("open file %s to get file size failed.", filePath.c_str()));
    }
    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    readFile.close();
    return datasetSize;
}

ssize_t LocalFileSystem::Write(const string& filePath, const char* fileContent, size_t dataSize)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, fileMode);
    if (fd == -1) {
        throw runtime_error(StringFormat("open file %s to write failed.", filePath.c_str()));
    }

    size_t dataCol = dataSize;
    size_t writeSize = 0;
    size_t idx = 0;
    ssize_t writeBytesNum = 0;

    while (dataCol != 0) {
        if (dataCol > oneTimeReadWriteLen) {
            writeSize = oneTimeReadWriteLen;
        } else {
            writeSize = dataCol;
        }
        ssize_t res = write(fd, fileContent + idx, writeSize);
        if (res == -1) {
            close(fd);
            return res;
        }
        dataCol -= writeSize;
        idx += writeSize;
        writeBytesNum += res;
    }
    close(fd);
    return writeBytesNum;
}

ssize_t LocalFileSystem::Write(const string& filePath, vector<float*> fileContent, size_t dataSize)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, fileMode);
    if (fd == -1) {
        throw runtime_error(StringFormat("open file %s to write failed.", filePath.c_str()));
    }

    buffer.reserve(BUFFER_SIZE);
    BufferQueue queue;
    ssize_t writeBytesNum = 0;
    std::thread writer(&LocalFileSystem::WriterFn, this, std::ref(queue), fd, std::ref(writeBytesNum));

    size_t loops = fileContent.size();
    for (size_t i = 0; i < loops; i++) {
        size_t idx = 0;
        size_t writeSize = 0;
        size_t dataCol = dataSize;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                writeSize = oneTimeReadWriteLen;
            } else {
                writeSize = dataCol;
            }
            FillToBuffer(queue, reinterpret_cast<const char *>(fileContent[i]) + idx, writeSize);
            dataCol -= writeSize;
            idx += writeSize;
        }
    }

    // After all data has been processed, check if there is any data left in the buffer
    if (!buffer.empty()) {
        queue.Push(std::move(buffer));
        buffer.clear();
    }

    queue.Push(std::vector<char>());
    writer.join();
    close(fd);
    return writeBytesNum;
}

/// 用于动态扩容模式下，往本地文件中写embedding
/// \param filePath 文件路径
/// \param embeddingSize embedding的长度
/// \param addressArr 存放embedding的地址vector
/// \param deviceId 运行的卡的id
/// \return
void LocalFileSystem::WriteEmbedding(const string& filePath, const int& embeddingSize,
                                     const vector<int64_t>& addressArr, int deviceId)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, fileMode);
    if (fd == -1) {
        throw runtime_error(StringFormat("open file %s to write failed.", filePath.c_str()));
    }

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        close(fd);
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    for (size_t i = 0; i < addressArr.size(); i += keyAddrElem) {
        vector<float> row(embeddingSize);
        int64_t address = addressArr.at(i);
        float *floatPtr = reinterpret_cast<float *>(address);

        aclError ret;
        try {
            ret = aclrtMemcpy(row.data(), embeddingSize * sizeof(float),
                              floatPtr, embeddingSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        } catch (std::exception& e) {
            close(fd);
            throw runtime_error(StringFormat("error happen when acl memory copy from device to host: %s", e.what()));
        }

        if (ret != ACL_SUCCESS) {
            close(fd);
            throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
        }

        ssize_t result = write(fd, row.data(), embeddingSize * sizeof(float));
        if (result != embeddingSize * sizeof(float)) {
            close(fd);
            throw runtime_error("Error writing to local file, "
                                "please check the disk buffer or temporary folder space or file permissions!");
        }
    }
#endif
    close(fd);
}

ssize_t LocalFileSystem::Read(const string& filePath, char* fileContent, size_t datasetSize)
{
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw runtime_error(StringFormat("Failed to open read file: %s", filePath.c_str()));
    }

    try {
        ValidateReadFile(filePath, datasetSize);
    } catch (const std::invalid_argument& e) {
        close(fd);
        throw runtime_error(StringFormat("Invalid read file path: %s", e.what()));
    }

    size_t idx = 0;
    size_t readSize = 0;
    ssize_t readBytesNum = 0;
    while (datasetSize != 0) {
        if (datasetSize > oneTimeReadWriteLen) {
            readSize = oneTimeReadWriteLen;
        } else {
            readSize = datasetSize;
        }
        ssize_t res = read(fd, fileContent + idx, readSize);
        if (res == -1) {
            close(fd);
            return res;
        }
        datasetSize -= readSize;
        idx += readSize;
        readBytesNum += readSize;
    }
    close(fd);
    return readBytesNum;
}

ssize_t LocalFileSystem::Read(const string& filePath, vector<vector<float>>& fileContent, size_t datasetSize)
{
    size_t embDataOuterSize = fileContent.capacity();
    auto onceReadByteSize { datasetSize / embDataOuterSize };

    size_t mapByteSize;
    size_t mapRowNum;
    CalculateMapSize(datasetSize, mapByteSize, mapRowNum, onceReadByteSize);

    off_t offset = 0;
    size_t remainBytes = datasetSize;
    ssize_t readBytesNum = 0;

    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw runtime_error(StringFormat("Failed to open read file: %s", filePath.c_str()));
    }

    for (size_t i = 0; i < embDataOuterSize; i += mapRowNum) {
        // 如果剩余字节数小于每次映射的字节数，则更新每次映射的字节数和行数
        if (remainBytes < mapByteSize) {
            mapByteSize = remainBytes;
            mapRowNum = mapByteSize / onceReadByteSize;
        }

        void* tempMappedData = mmap(nullptr, mapByteSize, PROT_READ, MAP_PRIVATE, fd, offset);
        if (tempMappedData == MAP_FAILED) {
            close(fd);
            return -1;
        }
        readBytesNum += mapByteSize;

        char* mappedData = static_cast<char*>(tempMappedData);

        // 处理映射的数据
        try {
            HandleMappedData(mappedData, mapRowNum, onceReadByteSize, fileContent, i);
        } catch (const std::runtime_error& e) {
            close(fd);
            munmap(mappedData, mapByteSize);
            throw runtime_error(StringFormat("handle mapped data error: %s", e.what()));
        }
        munmap(mappedData, mapByteSize);

        offset += mapByteSize;
        remainBytes -= mapByteSize;
    }
    close(fd);
    return readBytesNum;
}

/// 用于动态扩容模式下，从本地文件中读取embedding
/// \param filePath 文件路径
/// \param embeddingSize embedding的长度
/// \param addressArr 存放embedding的地址vector
/// \param deviceId 运行的卡的id
/// \return
void LocalFileSystem::ReadEmbedding(const string& filePath, const int& embeddingSize,
                                    vector<int64_t>& addressArr, int deviceId)
{
    std::ifstream readFile;
    readFile.open(filePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!readFile.is_open()) {
        throw runtime_error(StringFormat("open file %s to read failed.", filePath.c_str()));
    }

    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    auto embHashMapSize = static_cast<int>(datasetSize / sizeof(float) / embeddingSize);
    readFile.seekg(0, std::ios::beg);

    try {
        ValidateReadFile(filePath, datasetSize);
    } catch (const std::invalid_argument& e) {
        readFile.close();
        throw runtime_error(StringFormat("Invalid read file path: %s", e.what()));
    }

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        readFile.close();
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    void *newBlock = nullptr;
    aclError ret = aclrtMalloc(&newBlock, static_cast<int>(datasetSize), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        readFile.close();
        throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
    }

    float *floatPtr = static_cast<float *>(newBlock);
    addressArr.reserve(embHashMapSize);

    for (size_t i = 0, j = 0; i < embHashMapSize; i += keyAddrElem, ++j) {
        vector<float> row(embeddingSize);
        readFile.read(reinterpret_cast<char *>(row.data()), embeddingSize * sizeof(float));
        aclError ec;
        try {
            ec = aclrtMemcpy(floatPtr + j * embeddingSize, embeddingSize * sizeof(float),
                             row.data(), embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        } catch (std::exception& e) {
            readFile.close();
            throw runtime_error(StringFormat("error happen when acl memory copy from host to device: %s", e.what()));
        }

        if (ec != ACL_SUCCESS) {
            readFile.close();
            throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ec).c_str());
        }
        int64_t address = reinterpret_cast<int64_t>(floatPtr + j * embeddingSize);
        addressArr.push_back(address);
    }
#endif
    readFile.close();
}

void LocalFileSystem::WriterFn(BufferQueue& queue, int fd, ssize_t& writerBytesNum)
{
    while (true) {
        queue.Pop(writeBuffer);
        if (writeBuffer.size() == 0) {
            break;
        }
        ssize_t res = write(fd, writeBuffer.data(), writeBuffer.size());
        if (res == -1) {
            close(fd);
            writerBytesNum = -1;
            break;
        }
        writerBytesNum += res;
        writeBuffer.clear();
    }
}

void LocalFileSystem::FillToBuffer(BufferQueue& queue, const char* data, size_t dataSize)
{
    size_t dataIdx = 0;
    while (dataIdx < dataSize) {
        size_t remainingSpace = BUFFER_SIZE - buffer.size();
        if (dataSize - dataIdx <= remainingSpace) {
            buffer.insert(buffer.cend(), data + dataIdx, data + dataSize);
            return;
        } else {
            buffer.insert(buffer.cend(), data + dataIdx, data + dataIdx + remainingSpace);
            queue.Push(std::move(buffer));
            if (buffer.capacity() < BUFFER_SIZE) {
                buffer.reserve(BUFFER_SIZE);
            }
            dataIdx += remainingSpace;
        }
    }
}

void LocalFileSystem::CalculateMapSize(off_t fileSize, size_t& mapByteSize,
                                       size_t& mapRowNum, size_t onceReadByteSize) const
{
    // 每次映射的字节数
    mapByteSize = MAP_BYTE_SIZE;
    // 确保mapByteSize是onceReadByteSize和pageSize的整数倍，确保每次映射的offset是页大小的整数倍
    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize == -1) {
        throw std::runtime_error("Failed to get page size: " + std::string(strerror(errno)));
    }
    size_t lcmVal = std::lcm(onceReadByteSize, pageSize);
    mapByteSize = (mapByteSize / lcmVal) * lcmVal;

    // 如果文件大小小于每次映射的字节数，则一次性映射，映射大小不是页大小整数倍的时候，mmap会自动向上取整，额外的字节会初始化成零
    if (fileSize <= mapByteSize) {
        mapByteSize = fileSize;
    }

    mapRowNum = mapByteSize / onceReadByteSize;
}


void LocalFileSystem::HandleMappedData(char* mappedData, size_t mapRowNum, size_t onceReadByteSize,
                                       vector<vector<float>>& dst, size_t cnt) const
{
#pragma omp parallel for
    for (size_t j = 0; j < mapRowNum; ++j) {
        size_t idx = 0;
        size_t readSize = 0;
        size_t dataCol = onceReadByteSize;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                readSize = oneTimeReadWriteLen;
            } else {
                readSize = dataCol;
            }

            errno_t err = memcpy_s(dst[cnt + j].data() + idx, readSize,
                                   mappedData + j * onceReadByteSize + idx, readSize);
            if (err != 0) {
                throw std::runtime_error("Error execution memcpy_s: " + std::to_string(err));
            }
            dataCol -= readSize;
            idx += readSize;
        }
    }
}