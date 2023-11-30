/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-10-19
 */

#ifndef MX_REC_LOCAL_FILE_SYSTEM_H
#define MX_REC_LOCAL_FILE_SYSTEM_H

#include "file_system/file_system.h"
#include "file_system/buffer_queue.h"

namespace MxRec {
    using namespace std;
    const int DIR_RIGHT_MODE = 0750;
    const int FILE_RIGHT_MODE = 0640;
    class LocalFileSystem : public FileSystem {
    public:
        LocalFileSystem() : dirMode(DIR_RIGHT_MODE), fileMode(FILE_RIGHT_MODE), currDir("."), prevDir("..") {}
        ~LocalFileSystem() override {}

        void CreateDir(const string& dirName) override;
        vector<string> ListDir(const string& dirName) override;
        size_t GetFileSize(const string& filePath) override;

        ssize_t Write(const string& filePath, const char* fileContent, size_t dataSize) override;
        ssize_t Write(const string& filePath, vector<float*> fileVector, size_t dataSize) override;
        void WriteEmbedding(const string& filePath, const int& embeddingSize,
                            const vector<int64_t>& addressArr, int deviceId) override;

        ssize_t Read(const string& filePath, char* fileContent, size_t datasetSize) override;
        ssize_t Read(const string& filePath, vector<vector<float>>& fileContent, size_t datasetSize) override;
        void ReadEmbedding(const string& filePath, const int& embeddingSize,
                           vector<int64_t>& addressArr, int deviceId) override;

        void WriterFn(BufferQueue& queue, int fd, ssize_t& writerBytesNum);
        void FillToBuffer(BufferQueue& queue, const char* data, size_t dataSize);
        void CalculateMapSize(off_t fileSize, size_t& mapByteSize, size_t& mapRowNum, size_t onceReadByteSize) const;
        void HandleMappedData(char* mappedData, size_t mapRowNum, size_t onceReadByteSize,
                                               vector<vector<float>>& dst, size_t cnt) const;

    private:
        const mode_t dirMode;
        const mode_t fileMode;
        const string currDir;
        const string prevDir;
    };
}

#endif // MX_REC_LOCAL_FILE_SYSTEM_H