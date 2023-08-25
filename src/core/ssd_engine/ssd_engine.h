/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */
#ifndef MXREC_ENGINE_H
#define MXREC_ENGINE_H

#include "table.h"

#include <string>
#include <map>
#include <vector>

#include "utils/common.h"


namespace MxRec {

    class SSDEngine {
    public:
        bool IsTableExist(const string &tableName);

        bool IsKeyExist(const string &tableName, emb_key_t key);

        void CreateTable(const string &tableName, vector<string> savePaths, uint64_t maxTableSize);

        int64_t GetTableAvailableSpace(const string &tableName);

        void InsertEmbeddings(const string &tableName, vector<emb_key_t> &keys, vector<vector<float>> &embeddings);

        void DeleteEmbeddings(const string &tableName, vector<emb_key_t> &keys);

        vector<vector<float>> FetchEmbeddings(const string &tableName, vector<emb_key_t> &keys);

        void Save(int step);

        void Load(const string &tableName, vector<string> savePaths, uint64_t maxTableSize, int step);

        void Start();

        void Stop();

        void SetCompactPeriod(chrono::seconds seconds);

        void SetCompactThreshold(double threshold);

    private:
        bool isRunning = false;

        // leave 50% space for stale data to avoid modification in file
        double compactThreshold = 0.5;
        chrono::seconds compactPeriod = chrono::seconds(60);

        map<string, shared_ptr<Table>> tableMap{};
        shared_ptr<thread> compactThread = nullptr;

        void CompactMonitor();
    };
}

#endif // MXREC_ENGINE_H
