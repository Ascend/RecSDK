/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

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
    constexpr int maxDepth = 100;
    int guard = 0;
    stringstream input(dirName);   // 读取str到字符串流中
    stringstream ss;
    string tmp;
    // 按'/'分割，自动创建多级目录
    while (getline(input, tmp, '/')) {
        guard++;
        if (guard > maxDepth) {
            throw runtime_error(StringFormat("create directory %s exceed max depth", dirName.c_str()));
        }
        ss << tmp << '/';
        int ret = mkdir(ss.str().c_str(), dirMode);
        if (ret != 0 && errno != EEXIST) {
            LOG_ERROR("Unable to create directory: {} ret:{} error info: {}", dirName, ret, strerror(errno));
            throw runtime_error(StringFormat("create directory %s failed: %s", dirName.c_str(), strerror(errno)));
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
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_APPEND, fileMode);
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

ssize_t LocalFileSystem::Write(const string& filePath, vector<vector<float>>& fileContent, size_t dataSize)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, fileMode);
    if (fd == -1) {
        throw runtime_error(StringFormat("open file %s to write failed.", filePath.c_str()));
    }

    vector<float> flattenContent;
    for (auto& vec : fileContent) {
        flattenContent.insert(flattenContent.cend(), vec.cbegin(), vec.cend());
    }

    size_t writeBytesRemain = flattenContent.size() * sizeof(float);
    size_t writeSize = 0;
    size_t idx = 0;
    ssize_t writeBytesNum = 0;
    auto dumpPtr = reinterpret_cast<const char*>(flattenContent.data());

    while (writeBytesRemain != 0) {
        if (writeBytesRemain > oneTimeReadWriteLen) {
            writeSize = oneTimeReadWriteLen;
        } else {
            writeSize = writeBytesRemain;
        }
        ssize_t res = write(fd, dumpPtr + idx, writeSize);
        if (res == -1) {
            close(fd);
            return res;
        }
        writeBytesRemain -= res;
        idx += res;
        writeBytesNum += res;
    }

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

ssize_t LocalFileSystem::Read(const string& filePath, vector<vector<float>>& fileContent, int64_t contentOffset,
                              vector<int64_t> offsetArr, const size_t& embeddingSize)
{
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (fp == nullptr) {
        throw runtime_error(StringFormat("Failed to open read file: %s", filePath.c_str()));
    }

    ssize_t readBytesNum = 0;
    size_t embeddingCount = 0;
    for (const auto& offset: offsetArr) {
        fseek(fp, offset * embeddingSize * sizeof(float), SEEK_SET);
        auto res = fread(fileContent[embeddingCount].data()+ contentOffset *embeddingSize, sizeof(float), embeddingSize,
                         fp);
        embeddingCount++;
        readBytesNum += embeddingSize * sizeof(float);
    }

    fclose(fp);
    return readBytesNum;
}


/// 用于动态扩容模式下，从本地文件中读取embedding
/// \param filePath 文件路径
/// \param embedSizeInfo embedding的长度
/// \param addressArr 存放embedding的地址vector
/// \param deviceId 运行的卡的id
/// \return
void LocalFileSystem::ReadEmbedding(const string& filePath, EmbeddingSizeInfo& embedSizeInfo, int64_t firstAddress,
                                    int deviceId, vector<int64_t> offsetArr)
{
#ifndef GTEST
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (fp == nullptr) {
        throw runtime_error(StringFormat("Failed to open read file: %s", filePath.c_str()));
    }
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    float* floatPtr = reinterpret_cast<float*>(firstAddress);
    auto i = 0;
    for (const auto& offset: offsetArr) {
        vector<float> row(embedSizeInfo.embeddingSize);
        fseek(fp, offset * embedSizeInfo.embeddingSize * sizeof(float), SEEK_SET);
        auto res = fread(row.data(), sizeof(float), embedSizeInfo.embeddingSize, fp);
        try {
            auto ec = aclrtMemcpy(floatPtr + i * embedSizeInfo.extendEmbSize,
                                  embedSizeInfo.embeddingSize * sizeof(float),
                                  row.data(), embedSizeInfo.embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        } catch (std::exception& e) {
            fclose(fp);
            throw runtime_error(StringFormat("error happen when acl memory copy from host to device: %s", e.what()));
        }
        i++;
    }
    fclose(fp);
#endif
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