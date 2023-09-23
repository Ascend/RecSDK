/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "utils/common.h"
#include "ssd_engine/ssd_engine.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(SSDEngine, CreateAndWriteAndReadAndAutoCompactAndSave)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string tbName = "test";
    vector<string> savePath = {"."};
    uint64_t maxTableSize = 100;
    double compactThreshold = 0.5;
    chrono::seconds compactPeriod = chrono::seconds(5);
    int saveStep = 0;

    // create and start
    SSDEngine *eng = new SSDEngine();
    eng->SetCompactThreshold(compactThreshold);
    eng->SetCompactPeriod(compactPeriod);
    eng->Start();
    eng->CreateTable(tbName, savePath, maxTableSize);

    // check table
    ASSERT_EQ(eng->IsTableExist(tbName), true);

    // write
    vector<emb_key_t> keys;
    vector<vector<float>> embeddings;
    for (emb_key_t k = 0; k < 10; k++) {
        keys.emplace_back(k);
        vector<float> emb = {static_cast<float>(k + 0.1), static_cast<float>(k + 0.2)};
        embeddings.emplace_back(emb);
    }
    eng->InsertEmbeddings(tbName, keys, embeddings);

    // read data
    auto ret = eng->FetchEmbeddings(tbName, keys);
    ASSERT_EQ(embeddings, ret);

    // check space
    ASSERT_EQ(eng->GetTableAvailableSpace(tbName), maxTableSize - keys.size());

    // delete and wait auto compact
    vector<emb_key_t> deleteKeys = {0};
    eng->DeleteEmbeddings(tbName, deleteKeys);
    this_thread::sleep_for(compactPeriod);

    // check space to see if stale data space released
    ASSERT_EQ(eng->GetTableAvailableSpace(tbName), maxTableSize - keys.size() + deleteKeys.size());

    // save
    eng->Save(saveStep);

    eng->Stop();
    delete eng;

    // after saving, full compact will perform, old file will be deleted
    string oldDataFilePath =
        savePath.front() + "ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" + tbName + "/" + "0.data.latest";
    string oldMetaFilePath =
        savePath.front() + "ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" + tbName + "/" + "0.meta.latest";
    ASSERT_EQ(fs::exists(oldDataFilePath), false);
    ASSERT_EQ(fs::exists(oldMetaFilePath), false);

    // check saved data existence
    string newDataFilePath =
        savePath.front() + "/ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" +
        tbName + "/" + "1.data." + to_string(saveStep);
    string newMetaFilePath =
        savePath.front() + "/ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" +
        tbName + "/" + "1.meta." + to_string(saveStep);
    string newTableMetaFilePath =
        savePath.front() + "/ssd_sparse_model_rank_" + GlogConfig::gRankId + "/" +
        tbName + "/" + tbName + ".meta." +
        to_string(saveStep);
    ASSERT_EQ(fs::exists(newDataFilePath), true);
    ASSERT_EQ(fs::exists(newMetaFilePath), true);
    ASSERT_EQ(fs::exists(newTableMetaFilePath), true);

    for (const string& p: savePath) {
        fs::remove_all(p + "/ssd_sparse_model_rank_" + GlogConfig::gRankId);
    }
}

TEST(SSDEngine, LoadAndRead)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string tbName = "test";
    vector<string> savePath = {GlogConfig::gRankId};
    uint64_t maxTableSize = 100;
    int saveStep = 0;

    // create and start
    shared_ptr<SSDEngine> engSave = make_shared<SSDEngine>();
    chrono::seconds compactPeriod = chrono::seconds(5);
    engSave->SetCompactPeriod(compactPeriod);
    engSave->Start();
    engSave->CreateTable(tbName, savePath, maxTableSize);

    // write
    vector<emb_key_t> keys;
    vector<vector<float>> embeddings;
    for (emb_key_t k = 0; k < 10; k++) {
        keys.emplace_back(k);
        vector<float> emb = {static_cast<float>(k + 0.1), static_cast<float>(k + 0.2)};
        embeddings.emplace_back(emb);
    }
    engSave->InsertEmbeddings(tbName, keys, embeddings);

    // save
    engSave->Save(saveStep);
    engSave->Stop();

    // load
    shared_ptr<SSDEngine> engLoad = make_shared<SSDEngine>();
    engLoad->Start();
    engLoad->Load(tbName, savePath, maxTableSize, saveStep);
    for (emb_key_t k: keys) {
        ASSERT_EQ(engLoad->IsKeyExist(tbName, k), true);
    }
    auto ret = engLoad->FetchEmbeddings(tbName, keys);
    ASSERT_EQ(embeddings, ret);
    engLoad->Stop();

    for (string p: savePath) {
        fs::remove_all(p);
    }
}