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

#ifndef MXREC_TABLE_H
#define MXREC_TABLE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <memory>
#include <set>

#include "file.h"
#include "utils/common.h"
#include "error/error.h"

namespace MxRec {
    using namespace std;

    class Table {
    public:
        Table(const string& name, vector<string>& savePaths, uint64_t maxTableSize, double compactThreshold);

        // initialize with loading specific step data
        Table(const string& name, vector<string>& saveDirs, uint64_t maxTableSize, double compactThreshold, int step);

        bool IsKeyExist(emb_cache_key_t key);

        void InsertEmbeddings(vector<emb_cache_key_t>& keys, vector<vector<float>>& embeddings);

        vector<vector<float>> FetchEmbeddings(vector<emb_cache_key_t>& keys);

        void DeleteEmbeddings(vector<emb_cache_key_t>& keys);

        void Save(int step, const map<emb_key_t, KeyInfo>& keyInfo);

        void Save(int step);

        uint64_t GetTableAvailableSpace();

        virtual void Compact(bool fullCompact, const map<emb_key_t, KeyInfo>& keyInfo);

        void Compact(bool fullCompact);

        uint64_t GetTableUsage();

        void InsertEmbeddingsByAddr(vector<emb_cache_key_t>& keys, vector<float*>& embeddingsAddr,
                                    uint32_t extEmbeddingSize);

        static void ThrowRuntimeError(const string& errMsg, ErrorType errorType = ErrorType::IO_ERROR);

        static void ThrowInvalidArgError(const string& errMsg, ErrorType errorType = ErrorType::INVALID_ARGUMENT);

        vector<emb_cache_key_t> ExportKeys();

    private:
        static void CreateTableDir(const string& path);

        void Load(const string& metaFilePath, int step);

        void InsertEmbeddingsInner(vector<emb_cache_key_t>& keys, vector<vector<float>>& embeddings);

        void DeleteEmbeddingsInner(vector<emb_cache_key_t>& keys);

        vector<vector<float>> FetchEmbeddingsInner(vector<emb_cache_key_t>& keys);

        void LoadDataFileSet(const shared_ptr<fstream>& metaFile, int step);

        void SetTablePathToDiskWithSpace();

        void InsertEmbeddingsByAddrInner(vector<emb_cache_key_t>& keys, vector<float*>& embeddingsAddr,
                                         uint64_t extEmbeddingSize);

        void CheckIsGraterThanMaxSize() const;

        string name;  // init by constructor
        vector<string> savePaths;  // init by constructor, support Save and Load from multiple path
        uint64_t maxTableSize;    // init by constructor, maximum key-value volume
        uint64_t totalKeyCnt = 0;
        unordered_map<emb_cache_key_t, shared_ptr<File>> keyToFile{}; // max mem cost 1.5G*2 for 100m keys
        set<shared_ptr<File>> staleDataFileSet{};
        string curTablePath = "";
        uint32_t curSavePathIdx = 0;
        set<shared_ptr<File>> fileSet{};
        mutex rwLock{};
        shared_ptr<File> curFile = nullptr;
        uint64_t curMaxFileID = 0; // no concurrent writing, always atomic increase
        const string saveDirPrefix = "ssd_sparse_model_rank_";
        const int convertToPercentage = 100;

        // for safety validation
        const uint32_t maxNameSize = 1024;

        /* args for performance(not expose to user yet)
         * 2 read thread is optimal when:
         *   embedding's dimension=240, maxDataNumInFile=10000
         *   fetch 1000000 keys at a time
         *   QPS(get n embedding per second) reach 109685
         * when maxDataNumInFile=10000:
         *   QPS(write n embedding per second) reach 194212
        */
        int readThreadNum = 2;
        uint32_t maxDataNumInFile = 10000;  // relax constrain for performance, need tuning
        double compactThreshold = 0.5;
        double diskAvailSpaceThreshold = 0.05;  // in range [0, 1), leave diskAvailSpaceThreshold*100 % for disk space
    };

    enum class SsdCompactLevel {
        NO_COMPACT = 0,
        PARTIAL_COMPACT = 1,
        FULL_COMPACT = 2
    };
}

#endif // MXREC_TABLE_H
