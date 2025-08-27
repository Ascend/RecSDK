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

#ifndef MXREC_FILE_H
#define MXREC_FILE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>

#include "utils/common.h"
#include "error/error.h"

namespace MxRec {
    using namespace std;
    namespace fs = std::experimental::filesystem;

    using offset_t = uint32_t;

    class File {
        static constexpr uint64_t KEY_DATA_LEN = sizeof(emb_cache_key_t);
        static constexpr uint64_t OFFSET_DATA_LEN = sizeof(offset_t);

    public:
        File(uint64_t fileID, string& fileDir);

        File(uint64_t fileID, string& fileDir, string& loadDir,
             int step);  // initialize with loading specific step data

        File(const File&) = delete;
        File& operator=(const File&) = delete;

        ~File();

        bool IsKeyExist(emb_cache_key_t key) const;

        void InsertEmbeddings(vector<emb_cache_key_t>& keys, vector<vector<float>>& embeddings);

        vector<vector<float>> FetchEmbeddings(vector<emb_cache_key_t>& keys);

        void DeleteEmbedding(emb_cache_key_t key);

        void Save(const string& saveDir, int step, const map<emb_key_t, KeyInfo>& keyInfo);

        void Save(const string& saveDir, int step);

        vector<emb_cache_key_t> GetKeys();

        uint64_t GetDataCnt() const;

        uint64_t GetFileID() const;

        uint64_t GetStaleDataCnt() const;

        void InsertEmbeddingsByAddr(vector<emb_cache_key_t>& keys, vector<float*>& embeddingsAddr,
                                    uint64_t extEmbeddingSize);

        static void ThrowRuntimeError(ErrorType errorType, const string& errMsg);

        static void ThrowInvalidArgError(ErrorType errorType, const string& errMsg);
    private:
        uint64_t fileID;  // init by constructor
        string fileDir;  // init by constructor
        fs::path dataFilePath = "";
        fs::path metaFilePath = "";
        fstream localFileData{};
        fstream localFileMeta{};

        // for safety validation
        const uint64_t maxEmbSize = 8192 * 10;  // x10 for optimizer state data

        uint64_t dataCnt = 0;
        uint64_t staleDataCnt = 0;
        unordered_map<emb_cache_key_t, offset_t> keyToOffset{}; // offset_t >> maxDataNumInFile * embDataSize
        offset_t lastWriteOffset = 0;

        void Load();
    };
}

#endif // MXREC_FILE_H
