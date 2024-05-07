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

#include "hdfs_file_system.h"

#include <acl/acl_rt.h>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>

#include "hdfs_wrapper.h"
#include "utils/logger.h"

using namespace std;
using namespace MxRec;

void HdfsFileSystem::CreateDir(const string& dirName)
{
    hdfsFS fs = ConnectHdfs();
    int ret = hdfs->CreateDirectory(fs, dirName.c_str());
    if (ret == -1) {
        LOG_DEBUG("Unable to create hdfs directory: {}", dirName);
    }
    hdfs->Disconnect(fs);
}

vector<string> HdfsFileSystem::ListDir(const string& dirName)
{
    vector<string> dirs;
    hdfsFS fs = ConnectHdfs();

    int numEntries = 0;
    hdfsFileInfo* subDirs = hdfs->ListDirectory(fs, dirName.c_str(), &numEntries);
    for (int i = 0; i < numEntries; ++i) {
        if (subDirs[i].mKind == kObjectKindDirectory) {
            dirs.emplace_back(subDirs[i].mName);
        }
    }

    hdfs->FreeFileInfo(subDirs, numEntries);
    hdfs->Disconnect(fs);
    return dirs;
}

size_t HdfsFileSystem::GetFileSize(const string& filePath)
{
    hdfsFS fs = ConnectHdfs();
    hdfsFileInfo* fileInfo = hdfs->GetPathInfo(fs, filePath.c_str());
    hdfs->Disconnect(fs);
    if (fileInfo == nullptr) {
        throw runtime_error(StringFormat("Error: Unable to get hdfs file info : {}.", filePath.c_str()));
    }
    auto fileSize = static_cast<size_t>(fileInfo->mSize);
    return fileSize;
}

ssize_t HdfsFileSystem::Write(const string& filePath, const char* fileContent, size_t dataSize)
{
    hdfsFS fs = ConnectHdfs();
    int flag = O_WRONLY | O_CREAT;
    hdfsFileInfo* fileInfo = hdfs->GetPathInfo(fs, filePath.c_str());
    if (fileInfo) {
        flag = O_WRONLY | O_APPEND;
    }
    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), flag, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Error: Unable to open hdfs file : {}.", filePath.c_str()));
    }

    tSize writeBytesNum = 0;
    tSize res = hdfs->Write(fs, file, fileContent, dataSize);
    if (res == -1) {
        hdfs->CloseFile(fs, file);
        hdfs->Disconnect(fs);
        return static_cast<ssize_t>(res);
    }
    writeBytesNum += res;

    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
    return static_cast<ssize_t>(writeBytesNum);
}

ssize_t HdfsFileSystem::Write(const string& filePath, vector<float*> fileContent, size_t dataSize)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_WRONLY | O_CREAT, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Error: Unable to open hdfs file : {}.", filePath.c_str()));
    }

    tSize writeBytesNum = 0;
    size_t loops = fileContent.size();
    for (size_t i = 0; i < loops; i++) {
        tSize res = hdfs->Write(fs, file, fileContent[i], dataSize);
        if (res == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            return static_cast<ssize_t>(res);
        }
        writeBytesNum += res;
    }
    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
    return static_cast<ssize_t>(writeBytesNum);
}

/// 用于动态扩容模式下，往hdfs文件中写embedding
/// \param filePath 文件路径
/// \param embeddingSize embedding的长度
/// \param addressArr 存放embedding的地址vector
/// \param deviceId 运行的卡的id
/// \return
void HdfsFileSystem::WriteEmbedding(const string& filePath, const int& embeddingSize,
                                    const vector<int64_t>& addressArr, int deviceId)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_WRONLY | O_CREAT, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Error: Unable to open hdfs file : {}.", filePath.c_str()));
    }

#ifndef GTEST
    for (size_t i = 0; i < addressArr.size(); i += embHashNum) {
        vector<float> row(embeddingSize);
        int64_t address = addressArr.at(i);
        float *floatPtr = reinterpret_cast<float *>(address);

        aclError ret = aclrtMemcpy(row.data(), embeddingSize * sizeof(float),
                                   floatPtr, embeddingSize * sizeof(float),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error("Error: Execute aclrtmemcpy from device to host failed.");
        }

        tSize res = hdfs->Write(fs, file, row.data(), embeddingSize * sizeof(float));
        if (res == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(StringFormat("Error: An error occurred while writing file: {}.", filePath.c_str()));
        }

        if (res != embeddingSize * sizeof(float)) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(
                    StringFormat("Error: Expected to write {} bytes, but actually write {} bytes to file {}.",
                                 embeddingSize * sizeof(float), res, filePath.c_str()));
        }
    }
#endif
    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
}

ssize_t HdfsFileSystem::Read(const string& filePath, char* fileContent, size_t datasetSize)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_RDONLY, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Error: Unable to open hdfs file : {}.", filePath.c_str()));
    }

    tSize readBytesNum = 0;
    tSize res = hdfs->Read(fs, file, fileContent, datasetSize);
    if (res == -1) {
        hdfs->CloseFile(fs, file);
        hdfs->Disconnect(fs);
        return static_cast<ssize_t>(res);
    }
    readBytesNum += res;

    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
    return static_cast<ssize_t>(readBytesNum);
}

ssize_t HdfsFileSystem::Read(const string& filePath, vector<vector<float>>& fileContent, int64_t contentOffset,
                             vector<int64_t> offsetArr, const size_t& embeddingSize)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_RDONLY, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Error: Unable to open hdfs file : {}.", filePath.c_str()));
    }

    ssize_t readBytesNum = 0;
    size_t embeddingCount = 0;
    for (const auto& offset: offsetArr) {
        hdfs->Seek(fs, file, offset * embeddingSize * sizeof(float));

        tSize res = hdfs->Read(fs, file, fileContent[embeddingCount].data() + contentOffset * embeddingSize,
                               embeddingSize * sizeof(float));
        if (res == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            return static_cast<ssize_t>(res);
        }
        embeddingCount++;
        readBytesNum += res;
    }

    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
    return static_cast<ssize_t>(readBytesNum);
}

/// 用于动态扩容模式下，从hdfs文件中读embedding
/// \param filePath 文件路径
/// \param embedSizeInfo embedding的长度
/// \param addressArr 存放embedding的地址vector
/// \param deviceId 运行的卡的id
/// \return
void HdfsFileSystem::ReadEmbedding(const string& filePath, EmbeddingSizeInfo& embedSizeInfo, int64_t firstAddress,
                                   int deviceId, vector<int64_t> offsetArr)
{
#ifndef GTEST
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_RDONLY, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Error: Unable to open hdfs file : {}.", filePath.c_str()));
    }

    float* floatPtr = reinterpret_cast<float*>(firstAddress);
    auto i = 0;
    for (const auto& offset: offsetArr) {
        vector<float> row(embedSizeInfo.embeddingSize);
        int seekRes = hdfs->Seek(fs, file, offset * embedSizeInfo.embeddingSize * sizeof(float));
        if (seekRes == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(StringFormat("Error: hdfsSeek failed with error. file offset: {}",
                                             offset * embedSizeInfo.embeddingSize * sizeof(float)));
        }

        tSize res = hdfs->Read(fs, file, row.data(), embedSizeInfo.embeddingSize * sizeof(float));
        if (res == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(StringFormat("Error: An error occurred while reading file: {}.", filePath.c_str()));
        }
        if (res != embedSizeInfo.embeddingSize * sizeof(float)) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(StringFormat(
                    StringFormat("Error: Expected to read {} bytes, but actually read {} bytes from file {}.",
                                 embedSizeInfo.embeddingSize * sizeof(float), res, filePath.c_str()));
        }

        aclError ret = aclrtMemcpy(floatPtr + i * embedSizeInfo.extendEmbSize,
                                   embedSizeInfo.embeddingSize * sizeof(float),
                                   row.data(), embedSizeInfo.embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error("Error: Execute aclrtmemcpy from host to device failed.");
        }
        i++;
    }
    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
#endif
}

hdfsFS HdfsFileSystem::ConnectHdfs()
{
    hdfsFS fs = hdfs->Connect("default", 0);
    if (!fs) {
        throw runtime_error("Connect hdfs file system failed.");
    }
    return fs;
}