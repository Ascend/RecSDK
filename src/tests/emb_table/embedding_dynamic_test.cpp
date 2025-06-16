/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <random>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <emock/emock.hpp>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <limits>
#include <mpi.h>

#include "utils/common.h"
#include "utils/logger.h"
#include "utils/error.h"
#include "emb_table/embedding_dynamic.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingDynamicTest : public testing::Test {
protected:
    EmbeddingDynamicTest()
    {
        int embSize = 1000;
        int extEmbSize = 2000;
        struct EmbInfoParams embParam(string("test1"), 0, embSize, extEmbSize, true, true, false, false);
        std::vector<size_t> vocabsize = {10000, 0, 0};
        vector<EmbCache::InitializerInfo> initializeInfos = {};
        std::vector<std::string> ssdDataPath = {""};
        std::vector<int64_t> paddingKeys = {1};
        vector<int> maxStep = {1000};
        embInfo_ = EmbInfo(embParam, vocabsize, initializeInfos, ssdDataPath, paddingKeys);
        int rankId;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        rankInfo_ = RankInfo(rankId, 0, 0, 1, maxStep);
        rankInfo_.useDynamicExpansion = true;
        savePath << "test_dynamic/device" << rankInfo_.rankId;
    }

    void SetUp() {
    }
    void TearDown()
    {
        GlobalMockObject::reset();
    }

    static void SetupTestCase()
    {
        if (access("test_dynamic", F_OK) == 0) {
            system("rm -rf test_dynamic");
        }
    }

    static void TearDownTestCase()
    {
        if (access("test_dynamic", F_OK) == 0) {
            system("rm -rf test_dynamic");
        }
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
    stringstream savePath;
};

TEST_F(EmbeddingDynamicTest, TestMallocEmbeddingBlockShouldThrowErrorWhenNoUseDynamicExpansion)
{
    rankInfo_.useDynamicExpansion = false;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testKeys = {0, 1, 2, 3};
    EXPECT_THROW(table->Key2Offset(testKeys, TRAIN_CHANNEL_ID), std::bad_alloc);
}

TEST_F(EmbeddingDynamicTest, Key2Offset)
{
    rankInfo_.useDynamicExpansion = false;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testKeys = {-1, 1, 2, 3};
    EMOCK(&EmbeddingDynamic::GetEmptyEmbeddingAddress).stubs().will(returnValue(1));
    EMOCK(&EmbeddingTable::RecordPaddingKeysOffset).stubs();
    EXPECT_NO_THROW(table->Key2Offset(testKeys, TRAIN_CHANNEL_ID));
}

TEST_F(EmbeddingDynamicTest, Key2Offset_EmptyKeys)
{
    rankInfo_.useDynamicExpansion = false;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testKeys = {};
    EXPECT_NO_THROW(table->Key2Offset(testKeys, EVAL_CHANNEL_ID));
}

TEST_F(EmbeddingDynamicTest, Key2OffsetForDp)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testKeys = {-1, 1, 2, 3};
    EXPECT_NO_THROW(table->Key2OffsetForDp(testKeys, EVAL_CHANNEL_ID));
}

TEST_F(EmbeddingDynamicTest, Key2OffsetForDp_Error)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testKeys = {-1, 1, 2, 3};
    EXPECT_THROW(table->Key2OffsetForDp(testKeys, TRAIN_CHANNEL_ID), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, SaveKey)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = true;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[1] = KeyInfo();
    absl::flat_hash_map<emb_key_t, int64_t> keyMap;
    keyMap[1] = 0;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    table->keyOffsetMap = keyMap;
    auto localFileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(localFileSys);

    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    using WriteCharFunc = ssize_t (LocalFileSystem::*)(const string&, const char*, size_t);
    EMOCK(static_cast<WriteCharFunc>(&LocalFileSystem::Write)).stubs().will(returnValue(1));
    EXPECT_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, SaveKey_FailedError)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = true;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[1] = KeyInfo();
    absl::flat_hash_map<emb_key_t, int64_t> keyMap;
    keyMap[1] = 0;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    table->keyOffsetMap = keyMap;
    auto localFileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(localFileSys);

    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    using WriteCharFunc = ssize_t (LocalFileSystem::*)(const string&, const char*, size_t);
    EMOCK(static_cast<WriteCharFunc>(&LocalFileSystem::Write)).stubs().will(returnValue(-1));
    EXPECT_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, SaveKey_Error)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = true;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[0] = KeyInfo();
    absl::flat_hash_map<emb_key_t, int64_t> keyMap;
    keyMap[1] = 0;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    table->keyOffsetMap = keyMap;

    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EXPECT_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, SaveKey_NotSaveDelta)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = false;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[0] = KeyInfo();
    absl::flat_hash_map<emb_key_t, int64_t> keyMap;
    keyMap[1] = 0;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    table->keyOffsetMap = keyMap;

    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EXPECT_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, SaveEmbAndOptim_SaveEmbData)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = false;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[1] = KeyInfo();
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    auto fileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(fileSys);

    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    EMOCK(&EmbeddingDynamic::SaveKey).stubs();
    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EMOCK(&LocalFileSystem::WriteEmbedding).stubs();
    EXPECT_NO_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap));
}

TEST_F(EmbeddingDynamicTest, SaveEmbAndOptim_SaveOptimData)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = false;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[1] = KeyInfo();
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);

    EMOCK(&EmbeddingDynamic::SaveKey).stubs();
    EMOCK(&EmbeddingDynamic::SaveEmbData).stubs();
    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EXPECT_NO_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap));
}

TEST_F(EmbeddingDynamicTest, LoadKey)
{
    const string savePath = "./test/path";
    map<string, unordered_set<emb_cache_key_t>> trainKeySetActual;
    auto& trainKeySet = trainKeySetActual;
    vector<string> warmStartTablesActual;
    const auto& warmStartTables = warmStartTablesActual;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    auto fileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(fileSys);

    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    EMOCK(&LocalFileSystem::GetFileSize).stubs().will(returnValue(1));
    using ReadCharFunc = ssize_t (LocalFileSystem::*)(const string&, char*, size_t);
    EMOCK(static_cast<ReadCharFunc>(&LocalFileSystem::Read)).stubs().will(returnValue(1));
    EXPECT_THROW(table->Load(savePath, trainKeySet, warmStartTables), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, LoadKey_ReadError)
{
    const string savePath = "./test/path";
    map<string, unordered_set<emb_cache_key_t>> trainKeySetActual;
    auto& trainKeySet = trainKeySetActual;
    vector<string> warmStartTablesActual;
    const auto& warmStartTables = warmStartTablesActual;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    auto fileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(fileSys);

    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    EMOCK(&LocalFileSystem::GetFileSize).stubs().will(returnValue(1));
    EXPECT_THROW(table->Load(savePath, trainKeySet, warmStartTables), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, LoadEmbAndOptim)
{
    const string savePath = "./test/path";
    map<string, unordered_set<emb_cache_key_t>> trainKeySetActual;
    auto& trainKeySet = trainKeySetActual;
    vector<string> warmStartTablesActual;
    const auto& warmStartTables = warmStartTablesActual;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    auto fileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(fileSys);

    EMOCK(&EmbeddingDynamic::LoadKey).stubs();
    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    EXPECT_NO_THROW(table->Load(savePath, trainKeySet, warmStartTables));
}