/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */


#include <gtest/gtest.h>
#include <mpi.h>

#include "utils/common.h"
#include "ssd_engine/file.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(File, CreateEmptyFile)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    g_rankId = to_string(rankId);

    string savePath = g_rankId;
    bool isExceptionThrown = false;
    try {
        auto f = make_shared<File>(0, savePath);
    } catch (runtime_error &e) {
        isExceptionThrown = true;
        LOG(ERROR) << e.what();
    }
    ASSERT_EQ(isExceptionThrown, false);
    fs::remove_all(savePath);
}

TEST(File, LoadFromFile)
{
    // prepare
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    g_rankId = to_string(rankId);

    string savePath = g_rankId;
    if (!fs::exists(fs::absolute(savePath))) {
        if (!fs::create_directories(fs::absolute(savePath))) {
            throw runtime_error("fail to create Save directory");
        }
    }

    emb_key_t key = 0;
    offset_t offset = 0;
    vector<float> val = {1.0};

    fstream localFileMeta;
    localFileMeta.open(savePath + "/0.meta.0", ios::out | ios::trunc | ios::binary);
    localFileMeta.write(reinterpret_cast<char const *>(&key), sizeof(key));
    localFileMeta.write(reinterpret_cast<char const *>(&offset), sizeof(offset));
    localFileMeta.flush();
    if (localFileMeta.fail()) {
        throw runtime_error("fail to prepare meta file");
    }
    localFileMeta.close();

    fstream localFileData;
    localFileData.open(savePath + "/0.data.0", ios::out | ios::trunc | ios::binary);
    uint64_t embSize = val.size();
    localFileData.write(reinterpret_cast<char const *>(&embSize), sizeof(embSize));
    localFileData.write(reinterpret_cast<char const *>(val.data()), val.size() * sizeof(float));
    localFileData.flush();
    if (localFileData.fail()) {
        throw runtime_error("fail to prepare data file");
    }
    localFileData.close();

    // start test
    bool isExceptionThrown = false;
    try {
        auto f = make_shared<File>(0, savePath, 0);
    } catch (runtime_error &e) {
        LOG(ERROR) << e.what();
        isExceptionThrown = true;
    }
    ASSERT_EQ(isExceptionThrown, false);
    fs::remove_all(savePath);
}

TEST(File, WriteAndRead)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    g_rankId = to_string(rankId);

    string savePath = g_rankId;
    auto f = make_shared<File>(0, savePath);

    vector<emb_key_t> keys;
    vector<vector<float>> embeddings;
    for (emb_key_t k = 0; k < 10; k++) {
        keys.emplace_back(k);
        vector<float> emb = {static_cast<float>(k + 0.1), static_cast<float>(k + 0.2)};
        embeddings.emplace_back(emb);
    }

    f->InsertEmbeddings(keys, embeddings);
    auto ret = f->FetchEmbeddings(keys);
    ASSERT_EQ(embeddings, ret);


    f->DeleteEmbedding(0);
    ASSERT_EQ(f->IsKeyExist(0), false);

    fs::remove_all(savePath);
}

TEST(File, SaveAndLoad)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    g_rankId = to_string(rankId);

    int saveStep = 0;
    string savePath = g_rankId;
    auto fTmp = make_shared<File>(0, savePath);

    vector<emb_key_t> key = {0};
    vector<vector<float>> expect = {{1.0, 1.1}};
    fTmp->InsertEmbeddings(key, expect);
    fTmp->Save(saveStep);

    auto fLoad = make_shared<File>(0, savePath, saveStep);
    auto actual = fLoad->FetchEmbeddings(key);
    ASSERT_EQ(expect, actual);

    fs::remove_all(savePath);
}
