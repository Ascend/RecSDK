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
#include "ssd_engine/file.h"

using namespace std;
using namespace MxRec;
using namespace testing;

TEST(File, CreateEmptyFile)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string fileDir = GlogConfig::gRankId;
    bool isExceptionThrown = false;
    try {
        auto f = make_shared<File>(0, fileDir);
    } catch (runtime_error &e) {
        isExceptionThrown = true;
        LOG_ERROR(e.what());
    }
    ASSERT_EQ(isExceptionThrown, false);
    fs::remove_all(fileDir);
}

TEST(File, LoadFromFile)
{
    // prepare
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string fileDir = GlogConfig::gRankId;
    if (!fs::exists(fs::absolute(fileDir)) && !fs::create_directories(fs::absolute(fileDir))) {
        throw runtime_error("fail to create Save directory");
    }

    emb_key_t key = 0;
    offset_t offset = 0;
    vector<float> val = {1.0};

    fstream localFileMeta;
    localFileMeta.open(fileDir + "/0.meta.0", ios::out | ios::trunc | ios::binary);
    localFileMeta.write(reinterpret_cast<char const *>(&key), sizeof(key));
    localFileMeta.write(reinterpret_cast<char const *>(&offset), sizeof(offset));
    localFileMeta.flush();
    if (localFileMeta.fail()) {
        throw runtime_error("fail to prepare meta file");
    }
    localFileMeta.close();

    fstream localFileData;
    localFileData.open(fileDir + "/0.data.0", ios::out | ios::trunc | ios::binary);
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
    string loadDir = fileDir;  // for test convenience
    try {
        auto f = make_shared<File>(0, fileDir, loadDir, 0);
    } catch (runtime_error &e) {
        LOG_ERROR(e.what());
        isExceptionThrown = true;
    }
    ASSERT_EQ(isExceptionThrown, false);
    fs::remove_all(fileDir);
}

TEST(File, WriteAndRead)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string savePath = GlogConfig::gRankId;
    auto f = make_shared<File>(0, savePath);

    vector<emb_cache_key_t> keys;
    vector<vector<float>> embeddings;
    for (emb_cache_key_t k = 0; k < 10; k++) {
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
    GlogConfig::gRankId = to_string(rankId);

    int saveStep = 0;
    string fileDir = GlogConfig::gRankId;
    auto fTmp = make_shared<File>(0, fileDir);

    vector<emb_cache_key_t> key = {0};
    vector<vector<float>> expect = {{1.0, 1.1}};
    fTmp->InsertEmbeddings(key, expect);
    string saveDir = fileDir;  // for test convenience
    fTmp->Save(saveDir, saveStep);

    string loadDir = fileDir;  // for test convenience
    auto fLoad = make_shared<File>(0, fileDir, loadDir, saveStep);
    auto actual = fLoad->FetchEmbeddings(key);
    ASSERT_EQ(expect, actual);

    fs::remove_all(fileDir);
}

TEST(File, WriteByAddrAndRead)
{
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    GlogConfig::gRankId = to_string(rankId);

    string savePath = GlogConfig::gRankId;
    auto f = make_shared<File>(0, savePath);

    vector<emb_cache_key_t> keys;
    vector<float*> embeddings;
    uint64_t extEmbeddingSize = 1;
    for (emb_cache_key_t k = 0; k < 10; k++) {
        keys.emplace_back(k);
        float* emb = new float;
        *emb = static_cast<float>(k + 0.1);
        embeddings.emplace_back(emb);
    }

    f->InsertEmbeddingsByAddr(keys, embeddings, extEmbeddingSize);
    auto ret = f->FetchEmbeddings(keys);
    for (int i = 0; i < 10; i++) {
        if (std::abs(ret[i][0] - *embeddings[i]) > std::numeric_limits<float>::epsilon()) {
            FAIL() << "embedding result not equal to input";
        }
    }

    for (auto emb : embeddings)
    {
        delete emb;
        emb = nullptr;
    }
    

    fs::remove_all(savePath);
}