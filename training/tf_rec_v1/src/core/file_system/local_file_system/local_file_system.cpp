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
#include <fcntl.h>

#include "utils/common.h"
#include "error/error.h"

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
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                               StringFormat("Create directory %s exceed max depth.", dirName.c_str()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
        ss << tmp << '/';
        int ret = mkdir(ss.str().c_str(), dirMode);
        if (ret != 0 && errno != EEXIST) {
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                               Logger::Format("Unable to create directory: {} ret:{} error info: {}.",
                                              dirName, ret, strerror(errno)));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
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
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                           StringFormat("Open file %s to get file size failed.", filePath.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    readFile.close();
    return datasetSize;
}

ssize_t LocalFileSystem::Write(const string& filePath, const char* fileContent, size_t dataSize)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_APPEND, fileMode);
    CheckOpenFile4Write(filePath, fd);

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
    CheckOpenFile4Write(filePath, fd);

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
    CheckOpenFile4Write(filePath, fd);

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        close(fd);
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::ACL_ERROR,
                           StringFormat("Set device failed, device_id:%d.", deviceId).c_str());
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
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
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::ACL_ERROR,
                               StringFormat("Error happened when acl memory copy from device to host: %s.", e.what()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }

        if (ret != ACL_SUCCESS) {
            close(fd);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::ACL_ERROR,
                               StringFormat("Invoke aclrtMemcpy failed, ret=%d.", ret).c_str());
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }

        ssize_t result = write(fd, row.data(), embeddingSize * sizeof(float));
        if (result != embeddingSize * sizeof(float)) {
            close(fd);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR, "Error writing to local file, "
                               "please check the disk buffer or temporary folder space or file permissions!");
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
    }
#endif
    close(fd);
}

ssize_t LocalFileSystem::Read(const string& filePath, char* fileContent, size_t datasetSize)
{
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                           StringFormat("Failed to open read file: %s.", filePath.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    try {
        ValidateReadFile(filePath, datasetSize);
    } catch (const std::invalid_argument& e) {
        close(fd);
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                           StringFormat("Invalid read file path: %s.", e.what()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
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
    CheckOpenFileRet(fp, filePath);

    try {
        ValidateReadFile(filePath, GetFileSize(filePath));
    } catch (const std::invalid_argument& e) {
        fclose(fp);
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::INVALID_ARGUMENT,
                           StringFormat("Invalid read file path: %s.", e.what()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    ssize_t readBytesNum = 0;
    size_t embeddingCount = 0;
    for (const auto& offset: offsetArr) {
        int res = fseek(fp, offset * embeddingSize * sizeof(float), SEEK_SET);
        if (res != 0) {
            fclose(fp);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                               StringFormat("Failed to seek file path: %s.", filePath.c_str()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
        size_t elementsRead =
            fread(fileContent[embeddingCount].data() + contentOffset * embeddingSize, sizeof(float), embeddingSize, fp);
        if (elementsRead < embeddingSize && ferror(fp)) {
            fclose(fp);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                               StringFormat("Failed to read file path: %s.", filePath.c_str()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
        embeddingCount++;
        readBytesNum += embeddingSize * sizeof(float);
    }

    if (fclose(fp) != 0) {
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                           StringFormat("Failed to close file path: %s.", filePath.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
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
    if (aclrtSetDevice(static_cast<int32_t>(deviceId)) != ACL_ERROR_NONE) {
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::ACL_ERROR,
                           StringFormat("Set device failed, device_id:%d.", deviceId).c_str());
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    FILE *fp = fopen(filePath.c_str(), "rb");
    CheckOpenFileRet(fp, filePath);

    try {
        ValidateReadFile(filePath, GetFileSize(filePath));
    } catch (const std::invalid_argument& e) {
        fclose(fp);
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::INVALID_ARGUMENT,
                           StringFormat("Invalid read file path: %s.", e.what()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }

    float* floatPtr = reinterpret_cast<float*>(firstAddress);
    auto i = 0;
    for (const auto& offset: offsetArr) {
        vector<float> row(embedSizeInfo.embeddingSize);
        if (fseek(fp, offset * embedSizeInfo.embeddingSize * sizeof(float), SEEK_SET) != 0) {
            fclose(fp);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                               StringFormat("Failed to seek file path: %s.", filePath.c_str()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
        size_t elementsRead = fread(row.data(), sizeof(float), embedSizeInfo.embeddingSize, fp);
        if (elementsRead < embedSizeInfo.embeddingSize && ferror(fp)) {
            fclose(fp);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                               StringFormat("Failed to read file path: %s.", filePath.c_str()));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }

        auto ret = aclrtMemcpy(floatPtr + i * embedSizeInfo.extendEmbSize, embedSizeInfo.embeddingSize * sizeof(float),
                               row.data(), embedSizeInfo.embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            fclose(fp);
            auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::ACL_ERROR,
                               StringFormat("Error happened when acl memory copy from host to device, ret=%d.", ret));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
        i++;
    }
    if (fclose(fp) != 0) {
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                           StringFormat("Failed to close file path: %s.", filePath.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
#endif
}

void LocalFileSystem::CheckOpenFile4Write(const string& filePath, int openRetCode)
{
    if (openRetCode != -1) {
        return;
    }
    auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                       StringFormat("Open file %s to write failed.", filePath.c_str()));
    LOG_ERROR(error.ToString());
    throw std::runtime_error(error.ToString());
}

void LocalFileSystem::CheckOpenFileRet(FILE* fp, const string& filePath)
{
    if (fp == nullptr) {
        auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::IO_ERROR,
                           StringFormat("Failed to open read file: %s.", filePath.c_str()));
        LOG_ERROR(error.ToString());
        throw std::runtime_error(error.ToString());
    }
}
