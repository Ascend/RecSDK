/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-10-19
 */

#ifndef MX_REC_HDFS_FILE_SYSTEM_H
#define MX_REC_HDFS_FILE_SYSTEM_H

#include "file_system/file_system.h"
#include "hdfs_wrapper.h"

namespace MxRec {
    using namespace std;

    class HdfsFileSystem : public FileSystem {
    public:
        HdfsFileSystem()
        {
            hdfs = make_unique<HdfsWrapper>();
        };
        ~HdfsFileSystem() override {}

        void CreateDir(const string& dirName) override;
        vector<string> ListDir(const string& dirName) override;
        size_t GetFileSize(const string& filePath) override;

        ssize_t Write(const string& filePath, const char* fileContent, size_t dataSize) override;
        ssize_t Write(const string& filePath, vector<float*> fileVector, size_t dataSize) override;
        void WriteEmbedding(const string& filePath, const int& embeddingSize,
                            const vector<int64_t>& addressArr, int deviceId) override;

        ssize_t Read(const string& filePath, char* fileContent, size_t datasetSize) override;
        ssize_t Read(const string& filePath, vector<vector<float>>& fileContent, size_t datasetSize) override;
        void ReadEmbedding(const string &filePath, const int& embeddingSize,
                           vector<int64_t>& addressArr, int deviceId) override;

        hdfsFS ConnectHdfs();

        unique_ptr<HdfsWrapper> hdfs;
    };
}

#endif // MX_REC_HDFS_FILE_SYSTEM_H