/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */
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

namespace MxRec {
    using namespace std;

    class Table {
    public:
        Table(const string &name, vector<string> &savePaths, uint64_t maxTableSize, double compactThreshold);

        // initialize with loading specific step data
        Table(const string &name, vector<string> &saveDirs, uint64_t maxTableSize, double compactThreshold, int step);

        bool IsKeyExist(emb_key_t key);

        void InsertEmbeddings(vector<emb_key_t> &keys, vector<vector<float>> &embeddings);

        vector<vector<float>> FetchEmbeddings(vector<emb_key_t> &keys);

        void DeleteEmbeddings(vector<emb_key_t> &keys);

        void Save(int step);

        uint64_t GetTableAvailableSpace();

        void Compact(bool fullCompact);

    private:
        void Load(const string& metaFilePath, int step);

        void InsertEmbeddingsInner(vector<emb_key_t> &keys, vector<vector<float>> &embeddings);

        void DeleteEmbeddingsInner(vector<emb_key_t> &keys);

        vector<vector<float>> FetchEmbeddingsInner(vector<emb_key_t> &keys);

        void LoadDataFileSet(const shared_ptr<fstream>& metaFile, int step);

        void SetTablePathToDiskWithSpace();

        string name;  // init by constructor
        vector<string> savePaths;  // init by constructor, support Save and Load from multiple path
        uint64_t maxTableSize;    // init by constructor, maximum key-value volume
        uint64_t totalKeyCnt = 0;
        unordered_map<emb_key_t, shared_ptr<File>> keyToFile{}; // max mem cost 1.5G*2 for 100m keys
        set<shared_ptr<File>> staleDataFileSet{};
        string curTablePath = "";
        uint32_t curSavePathIdx = 0;
        set<shared_ptr<File>> fileSet{};
        mutex rwLock{};
        shared_ptr<File> curFile = nullptr;
        uint64_t curMaxFileID = 0; // no concurrent writing, always atomic increase
        const uint32_t maxNameSize = 1024;
        const string saveDirPrefix = "ssd_sparse_model_rank_";
        const int convertToPercentage = 100;

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
}

#endif // MXREC_TABLE_H
