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

#ifndef MXREC_ENGINE_H
#define MXREC_ENGINE_H

#include "table.h"

#include <string>
#include <map>
#include <vector>

#include "l3_storage/l3_storage.h"
#include "error/error.h"


namespace MxRec {

    class SSDEngine : public L3Storage {
    public:
        bool IsTableExist(const string &tableName) override;

        bool IsKeyExist(const string &tableName, emb_cache_key_t key) override;

        void CreateTable(const string &tableName, vector<string> savePaths, uint64_t maxTableSize) override;

        int64_t GetTableAvailableSpace(const string &tableName) override;

        void InsertEmbeddings(const string &tableName, vector<emb_cache_key_t> &keys,
                              vector<vector<float>> &embeddings);

        void DeleteEmbeddings(const string &tableName, vector<emb_cache_key_t> &keys) override;

        vector<vector<float>> FetchEmbeddings(const string &tableName, vector<emb_cache_key_t> &keys) override;

        void Save(int step, const map<string, map<emb_key_t, KeyInfo>>& keyInfoMap) override;

        void Save(int step) override;

        void Load(const string &tableName, vector<string> savePaths, uint64_t maxTableSize, int step) override;

        void Start() override;

        void Stop() override;

        void SetCompactPeriod(chrono::seconds seconds);

        void SetCompactThreshold(double threshold);

        int64_t GetTableUsage(const string& tableName) override;

        void InsertEmbeddingsByAddr(const string &tableName, vector<emb_cache_key_t> &keys,
                                    vector<float*> &embeddingsAddr, uint64_t extEmbeddingSize) override;

        static void CheckTableExist(bool isThrowError, const string& tableName);

        vector<std::pair<string, vector<emb_cache_key_t>>> ExportTableKey() override;

    private:
        void CheckSSDEngineIsRunning() const;

        bool isRunning = false;

        // leave 50% space for stale data to avoid modification in file
        double compactThreshold = 0.5;
        chrono::seconds compactPeriod = chrono::seconds(60);

        map<string, shared_ptr<Table>> tableMap{};
        shared_ptr<thread> compactThread = nullptr;

        void CompactMonitor();

        int loadStep = -1;
        int saveStep = -1;
    };
}

#endif // MXREC_ENGINE_H
