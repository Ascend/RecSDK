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
        ssize_t Write(const string& filePath, vector<float*> fileContent, size_t dataSize) override;
        void WriteEmbedding(const string& filePath, const int& embeddingSize,
                            const vector<int64_t>& addressArr, int deviceId) override;

        ssize_t Read(const string& filePath, char* fileContent, size_t datasetSize) override;
        ssize_t Read(const string& filePath, vector<vector<float>>& fileContent, int64_t contentOffset,
                     vector<int64_t> offsetArr, const size_t& embeddingSize) override;
        void ReadEmbedding(const string& filePath, EmbeddingSizeInfo& embedSizeInfo, int64_t addressArr, int deviceId,
                           vector <int64_t> offsetArr) override;

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
