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

ssize_t HdfsFileSystem::Read(const string& filePath, vector<vector<float>>& fileContent, int64_t contentOffset,
                             vector<int64_t> offsetArr, const size_t& embeddingSize)
{
    hdfsFS fs = ConnectHdfs();

    hdfsFile file = hdfs->OpenFile(fs, filePath.c_str(), O_RDONLY, 0, 0, 0);
    if (!file) {
        hdfs->Disconnect(fs);
        throw runtime_error("open hdfs file failed.");
    }

    ssize_t readBytesNum = 0;
    size_t embeddingCount = 0;
    for (const auto& offset: offsetArr) {
        hdfs->Seek(fs, file, offset * embeddingSize * sizeof(float));

        tSize res = hdfs->Read(fs, file, fileContent[embeddingCount].data() + contentOffset * embeddingSize,
                               embeddingSize * sizeof(float));

        embeddingCount++;
        readBytesNum += embeddingSize * sizeof(float);
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
        throw runtime_error("open hdfs file failed.");
    }

    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    float* floatPtr = reinterpret_cast<float*>(firstAddress);
    auto i = 0;
    for (const auto& offset: offsetArr) {
        vector<float> row(embedSizeInfo.embeddingSize);
        hdfs->Seek(fs, file, offset * embedSizeInfo.embeddingSize * sizeof(float));
        tSize res = hdfs->Read(fs, file, row.data(), embedSizeInfo.embeddingSize * sizeof(float));
        try {
            aclrtMemcpy(floatPtr + i * embedSizeInfo.extendEmbSize, embedSizeInfo.embeddingSize * sizeof(float),
                        row.data(), embedSizeInfo.embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        } catch (std::exception& e) {
            hdfs->CloseFile(fs, file);
            hdfs->Disconnect(fs);
            throw runtime_error(StringFormat("error happen when acl memory copy from host to device: %s", e.what()));
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