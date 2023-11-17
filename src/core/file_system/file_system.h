/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-10-19
 */

#ifndef MX_REC_FILE_SYSTEM_H
#define MX_REC_FILE_SYSTEM_H

#include "checkpoint/buffer_queue.h"

namespace MxRec {
    using namespace std;

    class FileSystem {
    public:
        FileSystem() = default;
        virtual ~FileSystem() = default;

        virtual void CreateDir(const string& dirName) = 0;
        virtual vector<string> ListDir(const string& dirName) = 0;
        virtual size_t GetFileSize(const string& filePath) = 0;

        virtual ssize_t Write(const string& filePath, const char* fileContent, size_t dataSize) = 0;
        virtual ssize_t Write(const string& filePath, vector<float*> fileContent, size_t dataSize) = 0;
        virtual void WriteEmbedding(const string& filePath, const int& embeddingSize,
                                    const vector<int64_t>& addressArr, int deviceId) = 0;

        virtual ssize_t Read(const string& filePath, char* fileContent, size_t datasetSize) = 0;
        virtual ssize_t Read(const string& filePath, vector<vector<float>>& fileContent, size_t datasetSize) = 0;
        virtual void ReadEmbedding(const string& filePath, const int& embeddingSize,
                                   vector<int64_t>& addressArr, int deviceId) = 0;

        // The parameter oneTimeReadWriteLen specifies the maximum length of a file read or write at a time.
        // The parameter can be adjusted based on the service requirements.
        const size_t oneTimeReadWriteLen = 32768;
        const int embHashNum = 1;
        const int keyAddrElem = 1;
        std::vector<char> buffer;
        std::vector<char> writeBuffer;
    };
}

#endif //MX_REC_FILE_SYSTEM_H
