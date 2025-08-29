/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <string>
#include <sstream>

using namespace std;
using namespace MxRec;

void LocalFileSystem::CreateDir(const string& dirName)
{
    constexpr int maxDepth = 100;
    int guard = 0;
    stringstream input(dirName);  // 读取str到字符串流中
    stringstream ss;
    string tmp;
    // 按'/'分割，自动创建多级目录
    while (getline(input, tmp, '/')) {
        guard++;
        if (guard > maxDepth) {
            auto errMsg = Logger::Format("Create directory:{} exceed max depth.", dirName);
            LOG_ERROR(errMsg);
            throw std::runtime_error(errMsg);
        }
        ss << tmp << '/';
        int ret = mkdir(ss.str().c_str(), dirMode);
        if (ret != 0 && errno != EEXIST) {
            auto errMsg = Logger::Format("Unable to create directory: {} ret:{} error info: {}.",
                                         dirName, ret, strerror(errno));
            LOG_ERROR(errMsg);
            throw std::runtime_error(errMsg);
        }
    }
}

void LocalFileSystem::CreateFileDir(const string& filePath)
{
    std::filesystem::path filePathObj(filePath);
    std::filesystem::path dir = filePathObj.parent_path();
    if (std::filesystem::exists(dir)) {
        return;
    }
    CreateDir(dir.generic_string());
}

vector<string> LocalFileSystem::ListDir(const string& dirName)
{
    vector<string> dirs;
    DIR* dir = opendir(dirName.c_str());
    struct dirent* en;
    if (dir == nullptr) {
        LOG_WARN("Open directory {} failed while trying to traverse the directory.", dirName);
        return dirs;
    }

    for (en = readdir(dir); en != nullptr; en = readdir(dir)) {
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
    std::ifstream thisReadFile;
    thisReadFile.open(filePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!thisReadFile.is_open()) {
        auto errMsg = Logger::Format("Open file:{} to get file size failed.", filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
    size_t datasetSize = static_cast<size_t>(thisReadFile.tellg());
    thisReadFile.close();
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

ssize_t LocalFileSystem::Read(const string& filePath, char* fileContent, size_t datasetSize)
{
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        auto errMsg = Logger::Format("Failed to open read file: {}.", filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }

    try {
        ValidateReadFile(filePath, datasetSize);
    } catch (const std::invalid_argument& e) {
        close(fd);
        auto errMsg = Logger::Format("Invalid read file path: {}.", filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
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
    FILE* fp = fopen(filePath.c_str(), "rb");
    CheckOpenFileRet(fp, filePath);

    try {
        ValidateReadFile(filePath, GetFileSize(filePath));
    } catch (const std::invalid_argument& e) {
        auto ret = fclose(fp);
        if (ret != 0) {
            LOG_ERROR("Close file failed, file:{}", filePath);
        }
        auto errMsg = Logger::Format("Invalid read file path: {}.", filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }

    ssize_t readBytesNum = 0;
    size_t embeddingCount = 0;
    for (const auto& offset : offsetArr) {
        int res = fseek(fp, offset * embeddingSize * sizeof(float), SEEK_SET);
        if (res != 0) {
            fclose(fp);
            auto errMsg = Logger::Format("Failed to seek file path: {}.", filePath);
            LOG_ERROR(errMsg);
            throw std::runtime_error(errMsg);
        }
        size_t elementsRead =
            fread(fileContent[embeddingCount].data() + contentOffset * embeddingSize, sizeof(float), embeddingSize, fp);
        if (elementsRead < embeddingSize && ferror(fp)) {
            fclose(fp);
            auto errMsg = Logger::Format("Failed to read file path: {}.", filePath);
            LOG_ERROR(errMsg);
            throw std::runtime_error(errMsg);
        }
        embeddingCount++;
        readBytesNum += embeddingSize * sizeof(float);
    }

    auto ret = fclose(fp);
    if (ret != 0) {
        auto errMsg = Logger::Format("Close file failed, file:{}", filePath);
        LOG_ERROR(errMsg);
        throw std::runtime_error(errMsg);
    }
    return readBytesNum;
}

void LocalFileSystem::CheckOpenFile4Write(const string& filePath, int openRetCode)
{
    if (openRetCode != -1) {
        return;
    }
    auto errMsg = Logger::Format("Open file: {} to write failed.", filePath);
    LOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
}

void LocalFileSystem::CheckOpenFileRet(FILE* fp, const string& filePath)
{
    if (fp != nullptr) {
        return;
    }
    auto errMsg = Logger::Format("Failed to open read file: {}.", filePath);
    LOG_ERROR(errMsg);
    throw std::runtime_error(errMsg);
}

void LocalFileSystem::Valid4WriteDir(const string& fileDirPath)
{
    if (fileDirPath.size() > FILE_PATH_LEN_MAX) {
        auto errMsg = Logger::Format("File dir path length exceed limit, file dir path:{}, len limit:{}.", fileDirPath,
            FILE_PATH_LEN_MAX);
        throw std::runtime_error(errMsg);
    }
    std::filesystem::path filePathObj(fileDirPath);
    if (!std::filesystem::exists(filePathObj)) {
        auto errMsg = Logger::Format("Directory path is not exist when write, file dir path:{}.", fileDirPath);
        throw std::runtime_error(errMsg);
    }
    if (!std::filesystem::is_directory(filePathObj)) {
        auto errMsg = Logger::Format("Param path is not a directory path, file dir path:{}.", fileDirPath);
        throw std::runtime_error(errMsg);
    }
    if (std::filesystem::is_symlink(filePathObj)) {
        auto errMsg = Logger::Format("Directory path is symbol link and is invalid, file dir path:{}.", fileDirPath);
        throw std::runtime_error(errMsg);
    }
}

bool MxRec::CheckFilePermission(const string& filePath)
{
    struct stat fileInfo;
    int ret = stat(filePath.c_str(), &fileInfo);
    if (ret != 0) {
        LOG_ERROR("Get stat info failed, file:{}, error:{}.", filePath, ret);
        return false;
    }

    mode_t mask = 0700;
    mode_t maxMode = 0640;
    const int perPermWidth = 3;
    vector<string> permMsg = { "Other group permission", "Owner group permission", "Owner permission" };
    for (int i = perPermWidth; i > 0; i--) {
        int curPerm = (fileInfo.st_mode & mask) >> ((i - 1) * perPermWidth);
        int maxPerm = (maxMode & mask) >> ((i - 1) * perPermWidth);
        mask = mask >> perPermWidth;
        if (curPerm > maxPerm) {
            LOG_ERROR("File permission wrong, file:%s, type:{}, current permission:{}, required no greater than:{}.",
                      filePath, permMsg[i - 1], curPerm, maxPerm);
            return false;
        }
    }
    return true;
}

void MxRec::ValidateReadFile(const string& dataDir, size_t datasetSize)
{
    // validate soft link
    struct stat fileInfo;
    if (lstat(dataDir.c_str(), &fileInfo) != -1) {
        if (S_ISLNK(fileInfo.st_mode)) {
            auto errMsg = Logger::Format("Found soft link in path:{}.", dataDir);
            LOG_ERROR(errMsg);
            throw std::invalid_argument(errMsg);
        }
    }
    // validate file size
    if (datasetSize > FILE_MAX_SIZE) {
        auto errMsg = Logger::Format("The reading file size is invalid, not in range [{}, {}], path:{}.",
                                     FILE_MIN_SIZE, FILE_MAX_SIZE, dataDir);
        LOG_ERROR(errMsg);
        throw std::invalid_argument(errMsg);
    }
    // validate file privilege
    std::filesystem::perms permissions = std::filesystem::status(dataDir).permissions();
    if ((permissions & std::filesystem::perms::owner_read) == std::filesystem::perms::none) {
        throw invalid_argument(Logger::Format("no read permission for file:{}", dataDir));
    }
}
