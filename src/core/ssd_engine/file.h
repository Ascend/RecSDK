/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */
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

namespace MxRec {
    using namespace std;
    namespace fs = std::experimental::filesystem;

    using offset_t = uint32_t;

    class File {
        static const uint64_t keyDataLen = sizeof(emb_key_t);
        static const uint64_t offsetDataLen = sizeof(offset_t);

    public:
        File(uint64_t fileID, string &fileDir);

        File(uint64_t fileID, string &fileDir, string &loadDir, int step); // initialize with loading specific step data

        ~File();

        bool IsKeyExist(emb_key_t key);

        void InsertEmbeddings(vector<emb_key_t> &keys, vector<vector<float>> &embeddings);

        vector<vector<float>> FetchEmbeddings(vector<emb_key_t> &keys);

        void DeleteEmbedding(emb_key_t key);

        void Save(const string &saveDir, int step);

        vector<emb_key_t> GetKeys();

        uint64_t GetDataCnt();

        uint64_t GetFileID();

        uint64_t GetStaleDataCnt();

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
        unordered_map<emb_key_t, offset_t> keyToOffset{}; // offset_t >> maxDataNumInFile * embDataSize
        offset_t lastWriteOffset = 0;

        void Load();
    };
}

#endif // MXREC_FILE_H
