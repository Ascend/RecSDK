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

#include <gtest/gtest.h>
#include <mpi.h>

#include "utils/common.h"
#include "ssd_engine/table.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(Table, WriteAndReadAndDeleteAndCompact)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string tbName = "test";
    vector<string> savePath = {GlogConfig::gRankId};
    uint64_t maxTableSize = 1000000;
    uint64_t embDim = 240;
    double compactThreshold = 0.5;

    // create
    auto tb = make_shared<Table>(tbName, savePath, maxTableSize, compactThreshold);

    // write
    emb_key_t nData = 1000000;
    emb_key_t batchSize = 10000;
    vector<emb_key_t> allKeys;
    vector<vector<float>> allEmbs;
    vector<emb_key_t> batchKeys;
    vector<vector<float>> batchEmbs;

    chrono::milliseconds writeCost = 0ms;
    for (emb_key_t k = 0; k < nData; k++) {
        vector<float> emb;
        emb.resize(embDim);
        for (uint64_t i = 0; i < embDim; ++i) {
            emb[i] = static_cast<float>(k + float(i) / float(10));
        }
        allKeys.emplace_back(k);
        allEmbs.emplace_back(emb);
        batchKeys.emplace_back(k);
        batchEmbs.emplace_back(emb);
        if ((k + 1) % batchSize == 0) {
            auto start = chrono::high_resolution_clock::now();
            tb->InsertEmbeddings(batchKeys, batchEmbs);
            auto end = chrono::high_resolution_clock::now();
            writeCost += chrono::duration_cast<std::chrono::milliseconds>(end - start);
            batchKeys.clear();
            batchEmbs.clear();
        }
    }

    LOG_INFO("n data:{} ,batch size:{} ,write cost(ms): {} ,QPS:{}",
        nData, batchSize, writeCost.count(), float(nData) * 1000 / writeCost.count());

    // read
    auto start = chrono::high_resolution_clock::now();
    auto ret = tb->FetchEmbeddings(allKeys);
    auto end = chrono::high_resolution_clock::now();
    auto readCost = chrono::duration_cast<std::chrono::milliseconds>(end - start);

    LOG_INFO("n data:{} ,batch size:{} ,read cost(ms):{} ,QPS:{}",
        nData, batchSize, readCost.count(), float(nData) * 1000 / readCost.count());

    ASSERT_EQ(allEmbs, ret);

    // check space
    auto availSpace = tb->GetTableAvailableSpace();
    ASSERT_EQ(availSpace, maxTableSize - allKeys.size());

    // delete
    tb->DeleteEmbeddings(allKeys);
    for (emb_key_t k: allKeys) {
        ASSERT_EQ(tb->IsKeyExist(k), false);
    }

    // full compact, old file will delete, valid data will move to new file
    tb->Compact(true);
    string oldDataFilePath =
        savePath.front() + "/ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" + tbName + "/" + "0.data.latest";
    string oldMetaFilePath =
        savePath.front() + "/ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" + tbName + "/" + "0.meta.latest";
    ASSERT_EQ(fs::exists(oldDataFilePath), false);
    ASSERT_EQ(fs::exists(oldMetaFilePath), false);

    for (const string& p: savePath) {
        fs::remove_all(p);
    }
}

TEST(Table, SaveAndLoad)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string tbName = "test";
    vector<string> savePath = {GlogConfig::gRankId};
    uint64_t maxTableSize = 100;
    double compactThreshold = 0.5;
    int saveStep = 0;

    // create
    auto tbSave = make_shared<Table>(tbName, savePath, maxTableSize, compactThreshold);

    // write and save
    emb_key_t nData = 10;
    vector<emb_key_t> keys;
    vector<vector<float>> embs;
    for (emb_key_t k = 0; k < nData; k++) {
        vector<float> emb = {static_cast<float>(k + 0.1), static_cast<float>(k + 0.2)};
        keys.emplace_back(k);
        embs.emplace_back(emb);
    }
    tbSave->InsertEmbeddings(keys, embs);
    tbSave->Save(saveStep);

    // load
    auto tbLoad = make_shared<Table>(tbName, savePath, maxTableSize, compactThreshold, saveStep);
    auto ret = tbLoad->FetchEmbeddings(keys);

    ASSERT_EQ(embs, ret);

    for (const string &p: savePath) {
        fs::remove_all(p);
    }
}

TEST(Table, GetTableUsage)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string tbName = "test";
    vector<string> savePath = {GlogConfig::gRankId};
    uint64_t maxTableSize = 100;
    double compactThreshold = 0.5;
    int saveStep = 0;

    // create
    auto tbSave = make_shared<Table>(tbName, savePath, maxTableSize, compactThreshold);

    // write
    uint64_t expectKeyCnt = 2;
    vector<emb_key_t> keys = {1, 2};
    vector<vector<float>> embs = {{0.1}, {0.2}};
    tbSave->InsertEmbeddings(keys, embs);

    // check before saving
    uint64_t keyCntSave = tbSave->GetTableUsage();
    ASSERT_EQ(keyCntSave, expectKeyCnt);

    // check after saving
    tbSave->Save(saveStep);
    uint64_t keyCntSave2 = tbSave->GetTableUsage();
    ASSERT_EQ(keyCntSave2, expectKeyCnt);

    // check after load
    auto tbLoad = make_shared<Table>(tbName, savePath, maxTableSize, compactThreshold, saveStep);
    uint64_t keyCntLoad = tbLoad->GetTableUsage();
    ASSERT_EQ(keyCntLoad, expectKeyCnt);
}