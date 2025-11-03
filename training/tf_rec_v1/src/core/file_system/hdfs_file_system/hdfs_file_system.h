/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef MX_REC_HDFS_FILE_SYSTEM_H
#define MX_REC_HDFS_FILE_SYSTEM_H

#include "file_system/file_system.h"
#include "hdfs_wrapper.h"

namespace MxRec {
    using namespace std;

    class HdfsFileSystem : public FileSystem {
    public:
        HdfsFileSystem() {};

        HdfsFileSystem(const HdfsFileSystem&) = delete;
        HdfsFileSystem& operator=(const HdfsFileSystem&) = delete;

        ~HdfsFileSystem() override
        {
            hdfs->Disconnect(fs);
        }

        void CreateDir(const string& dirName) override;
        vector<string> ListDir(const string& dirName) override;
        size_t GetFileSize(const string& filePath) override;

        ssize_t Write(const string& filePath, const char* fileContent, size_t dataSize) override;
        ssize_t Write(const string& filePath, vector<vector<float>>& fileContent, size_t dataSize) override;
        void WriteEmbedding(const string& filePath, const int& embeddingSize,
                            const vector<int64_t>& addressArr, int deviceId) override;

        ssize_t Read(const string& filePath, char* fileContent, size_t datasetSize) override;
        ssize_t Read(const string& filePath, vector<vector<float>>& fileContent, int64_t contentOffset,
                     vector<int64_t> offsetArr, const size_t& embeddingSize) override;
        void ReadEmbedding(const string& filePath, EmbeddingSizeInfo& embedSizeInfo, int64_t firstAddress, int deviceId,
                           vector <int64_t> offsetArr) override;

        hdfsFS ConnectHdfs();

        void CheckHdfsReadRet(hdfsFile file, tSize res, size_t expectReadBytes, const string& filePath);
        static void CheckOpenHdfsFileRet(hdfsFile file, const string& filePath);

        unique_ptr<HdfsWrapper> hdfs = make_unique<HdfsWrapper>();
        hdfsFS fs = ConnectHdfs();
    };
}

#endif // MX_REC_HDFS_FILE_SYSTEM_H