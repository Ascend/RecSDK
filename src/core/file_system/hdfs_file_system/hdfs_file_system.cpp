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
        return 0;
    }
    auto fileSize = static_cast<size_t>(fileInfo->mSize);
    return fileSize;
}

ssize_t HdfsFileSystem::Write(const string& filePath, const char* fileContent, size_t dataSize)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_WRONLY | O_CREAT, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error("Error writing to hdfs file.");
    }

    size_t dataCol = dataSize;
    size_t writeSize = 0;
    size_t idx = 0;
    tSize writeBytesNum = 0;

    while (dataCol != 0) {
        if (dataCol > oneTimeReadWriteLen) {
            writeSize = oneTimeReadWriteLen;
        } else {
            writeSize = dataCol;
        }

        tSize res = hdfs->Write(fs, file, fileContent + idx, writeSize);
        if (res == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            return static_cast<ssize_t>(res);
        }
        dataCol -= writeSize;
        idx += writeSize;
        writeBytesNum += res;
    }

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
        throw runtime_error("Error writing to hdfs file.");
    }

    tSize writeBytesNum = 0;
    size_t loops = fileContent.size();
    for (size_t i = 0; i < loops; i++) {
        size_t dataCol = dataSize;
        size_t writeSize = 0;
        size_t idx = 0;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                writeSize = oneTimeReadWriteLen;
            } else {
                writeSize = dataCol;
            }
            tSize res = hdfs->Write(fs, file, fileContent[i] + idx, writeSize);
            if (res == -1) {
                hdfs->CloseFile(fs, file);
                hdfs->Disconnect(fs);
                return static_cast<ssize_t>(res);
            }
            dataCol -= writeSize;
            idx += writeSize;
            writeBytesNum += res;
        }
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
        throw runtime_error("Error writing to hdfs file.");
    }

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        hdfs->CloseFile(fs, file);
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

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
            throw runtime_error("aclrtMemcpy failed");
        }

        auto numBytesWritten = hdfs->Write(fs, file, row.data(), embeddingSize * sizeof(float));
        if (numBytesWritten != embeddingSize * sizeof(float)) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error("Error writing to hdfs file.");
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
        throw runtime_error("open hdfs file failed.");
    }

    size_t dataCol = datasetSize;
    size_t idx = 0;
    size_t readSize = 0;
    tSize readBytesNum = 0;
    while (dataCol != 0) {
        if (dataCol > oneTimeReadWriteLen) {
            readSize = oneTimeReadWriteLen;
        } else {
            readSize = dataCol;
        }
        tSize res = hdfs->Read(fs, file, fileContent + idx, readSize);
        if (res == -1) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            return static_cast<ssize_t>(res);
        }
        dataCol -= readSize;
        idx += readSize;
        readBytesNum += res;
    }

    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
    return static_cast<ssize_t>(readBytesNum);
}

ssize_t HdfsFileSystem::Read(const string& filePath, vector<vector<float>>& fileContent, size_t datasetSize)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_RDONLY, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error("open hdfs file failed.");
    }

    size_t embDataOuterSize = fileContent.capacity();
    auto onceReadByteSize { datasetSize / embDataOuterSize };
    tSize readBytesNum = 0;

    for (size_t i = 0; i < embDataOuterSize; ++i) {
        size_t idx = 0;
        size_t readSize = 0;
        size_t dataCol = onceReadByteSize;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                readSize = oneTimeReadWriteLen;
            } else {
                readSize = dataCol;
            }
            tSize res = hdfs->Read(fs, file, reinterpret_cast<char *>(fileContent[i].data()) + idx, readSize);
            if (res == -1) {
                hdfs->CloseFile(fs, file);
                hdfs->Disconnect(fs);
                return static_cast<ssize_t>(res);
            }
            dataCol -= readSize;
            idx += readSize;
            readBytesNum += res;
        }
    }
    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
    return static_cast<ssize_t>(readBytesNum);
}

/// 用于动态扩容模式下，从hdfs文件中读embedding
/// \param filePath 文件路径
/// \param embeddingSize embedding的长度
/// \param addressArr 存放embedding的地址vector
/// \param deviceId 运行的卡的id
/// \return
void HdfsFileSystem::ReadEmbedding(const string& filePath, const int& embeddingSize,
                                   vector<int64_t>& addressArr, int deviceId)
{
    size_t datasetSize = GetFileSize(filePath);
    auto embHashMapSize = static_cast<int>(datasetSize / sizeof(float) / embeddingSize);
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_RDONLY, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error("open hdfs file failed.");
    }

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        hdfs->CloseFile(fs, file);
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    aclError ret;
    void *newBlock = nullptr;
    ret = aclrtMalloc(&newBlock, static_cast<int>(datasetSize), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        hdfs->CloseFile(fs, file);
        hdfs->Disconnect(fs);
        throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
    }

    float *floatPtr = static_cast<float *>(newBlock);
    for (size_t i = 0, j = 0; i < embHashMapSize; i += keyAddrElem, ++j) {
        vector<float> row(embeddingSize);
        hdfs->Read(fs, file, row.data(), embeddingSize * sizeof(float));

        aclError ec = aclrtMemcpy(floatPtr + j * embeddingSize, embeddingSize * sizeof(float),
                                  row.data(), embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ec != ACL_SUCCESS) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ec).c_str());
        }
        int64_t address = reinterpret_cast<int64_t>(floatPtr + j * embeddingSize);
        addressArr.push_back(address);
    }
#endif
    hdfs->CloseFile(fs, file);
    hdfs->Disconnect(fs);
}

hdfsFS HdfsFileSystem::ConnectHdfs()
{
    hdfsFS fs = hdfs->Connect("default", 0);
    if (!fs) {
        throw runtime_error("Connect hdfs file system failed.");
    }
    return fs;
}