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
#include "log/logger.h"
#include "error/error.h"
#include "emb_table/embedding_static.h"
#include "emb_table/embedding_mgmt.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingStaticTest : public testing::Test {
protected:
    EmbeddingStaticTest()
    {
        int embSize = 1000;
        int extEmbSize = 2000;
        struct EmbInfoParams embParam(string("test1"), 0, embSize, extEmbSize, true, true, false, false);
        std::vector<size_t> vocabsize = {100, 100, 100};
        vector<EmbCache::InitializerInfo> initializeInfos = {};
        std::vector<std::string> ssdDataPath = {""};
        std::vector<int64_t> paddingKeys = {1};
        vector<int> maxStep = {1000};
        embInfo_ = EmbInfo(embParam, vocabsize, initializeInfos, ssdDataPath, paddingKeys);
        int rankId;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        rankInfo_ = RankInfo(rankId, 0, 0, 1, maxStep);
    }

    void SetUp() {
    }
    void TearDown()
    {
        GlobalMockObject::reset();
    }

    static void SetupTestCase()
    {
        if (access("test_static", F_OK) == 0) {
            system("rm -rf test_static");
        }
    }
    static void TearDownTestCase()
    {
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
};

/**
 * 正常情况，将表装满
 */
TEST_F(EmbeddingStaticTest, Key2OffsetBasic)
{
    vector<EmbInfo> embInfos = {embInfo_};

    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> tmp1;
    for (size_t i = 0; i < 100; ++i) {
        tmp1.push_back(i);
    }

    table->Key2Offset(tmp1, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < tmp1.size(); ++i) {
        EXPECT_EQ(tmp1[i], i);
    }
    EXPECT_EQ(table->size(), 100);
    EXPECT_EQ(table->size(), table->GetMaxOffset());
    EXPECT_EQ(table->capacity(), 100);
}

/**
 * 边界条件:101超过100容量
 */
TEST_F(EmbeddingStaticTest, TestKey2OffsetShouldThrowErrorWhenOverflow)
{
    vector<EmbInfo> embInfos = {embInfo_};

    shared_ptr<EmbeddingStatic> table= std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> tmp1;
    for (size_t i = 0; i < 101; ++i) {
        tmp1.push_back(i);
    }

    EXPECT_EQ(table->capacity(), 100);
    EXPECT_THROW(table->Key2Offset(tmp1, TRAIN_CHANNEL_ID), std::runtime_error);
}

TEST_F(EmbeddingStaticTest, TestKey2OffsetWhenUseEvalChannel)
{
    vector<EmbInfo> embInfos = {embInfo_};

    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> testData;
    for (size_t i = 0; i < 100; ++i) {
        testData.push_back(i);
    }
    table->Key2Offset(testData, EVAL_CHANNEL_ID);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(testData[i], INVALID_KEY_VALUE);
    }
}

/**
 * 正常: 使用淘汰的位置
 */
TEST_F(EmbeddingStaticTest, TestEvictKeys)
{
    vector<EmbInfo> embInfos = {embInfo_};
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    constexpr size_t tableNum = 100;
    constexpr size_t testNum = 10;

    vector<emb_key_t> testData;
    for (size_t i = 0; i < tableNum; ++i) {
        testData.push_back(i);
    }
    table->Key2Offset(testData, TRAIN_CHANNEL_ID);
    // 全部淘汰
    vector<emb_cache_key_t> testDataAdapt(testData.cbegin(), testData.cend());
    table->EvictKeys(testDataAdapt);

    vector<emb_key_t> new_data;
    for (size_t i = 0; i < testNum; ++i) {
        new_data.push_back(i);
    }
    table->Key2Offset(new_data, TRAIN_CHANNEL_ID);
    // 查看是否淘汰
    std::vector<int64_t> evicted_keys = table->GetEvictedKeys();
    EXPECT_EQ(evicted_keys.size(), tableNum - testNum);
}

TEST_F(EmbeddingStaticTest, TestKey2OffsetForDpWhenUseEvalChannel)
{
    vector<EmbInfo> embInfos = {embInfo_};
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    int invalidCount = 20;
    vector<emb_key_t> tmp1;
    for (size_t i = 0; i < 100; ++i) {
        if (i < invalidCount) {
            tmp1.push_back(INVALID_KEY_VALUE);
        } else {
            tmp1.push_back(i);
        }
    }

    table->Key2OffsetForDp(tmp1, EVAL_CHANNEL_ID);
    for (size_t i = 0; i < tmp1.size(); ++i) {
        EXPECT_EQ(tmp1[i], INVALID_KEY_VALUE);
    }
    EXPECT_EQ(table->capacity(), 100);
}

TEST_F(EmbeddingStaticTest, TestKey2OffsetForDpShouldThrowErrorWhenUseTrainChannel)
{
    vector<EmbInfo> embInfos = {embInfo_};
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    vector<emb_key_t> tmp1;
    for (size_t i = 0; i < 100; ++i) {
        tmp1.push_back(i);
    }

    EXPECT_EQ(table->capacity(), 100);
    EXPECT_THROW(table->Key2OffsetForDp(tmp1, TRAIN_CHANNEL_ID), std::runtime_error);
}

TEST_F(EmbeddingStaticTest, TestSaveAndLoadKeyDataWhenSaveDeltaIsFalse)
{
    const string tableName = "test1";
    vector<EmbInfo> embInfos = {embInfo_};
    map<emb_key_t, KeyInfo> keyInfo;
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> keyData;
    for (size_t i = 0; i < testNum; ++i) {
        keyData.push_back(i);
    }

    table->Key2Offset(keyData, TRAIN_CHANNEL_ID);
    MxRec::KeyOffsetMemT kom = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
    map<EmbNameT, string> tmp;
    for (auto it = kom.begin(); it != kom.end(); ++it) {
        tmp.insert(pair<EmbNameT, string>(it->first, MapToString(it->second).c_str()));
    }
    LOG_INFO("test Key2Offset: lookupKeys: {}, keyOffsetMap: {}",
             VectorToString(keyData), MapToString(tmp));

    stringstream savePath;
    savePath << "test_static/device" << rankInfo_.rankId;
    table->SetFileSystemPtr(savePath.str());
    table->Save(savePath.str(), 1, false, keyInfo);
    stringstream saveKeyPath;
    saveKeyPath << savePath.str() << "/" << tableName << "/key";
    EXPECT_EQ(access(saveKeyPath.str().c_str(), F_OK), 0);

    stringstream fileKeyPath;
    fileKeyPath << saveKeyPath.str() << "/slice_" << rankInfo_.rankId << ".data";
    stringstream newFileKeyPath;
    newFileKeyPath << saveKeyPath.str() << "/slice.data";
    if (rankInfo_.rankId == 0) {
        RenameFilePath(fileKeyPath.str(), newFileKeyPath.str());

        map<string, unordered_set<emb_cache_key_t>> trainKeySetOne;
        vector<string> warmStartTablesOne;
        table->Load(savePath.str(), trainKeySetOne, warmStartTablesOne);
        vector<int64_t> deviceOffsetOne = table->GetDeviceOffset();
        EXPECT_EQ(table->GetMaxOffset(), testNum);
        EXPECT_NE(deviceOffsetOne.size(), 0);
    }
}

TEST_F(EmbeddingStaticTest, TestSaveKeyDataShouldThrowErrorWhenSaveDeltaIsTrue)
{
    vector<EmbInfo> embInfos = {embInfo_};
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    constexpr int testNum = 100;
    map<emb_key_t, KeyInfo> keyInfo;
    std::random_device rd;
    std::mt19937 gen(rd());
    int leftBound = 0;
    int rightBound = 100;
    std::uniform_int_distribution<> dis(leftBound, rightBound);

    for (size_t i = 0; i < testNum; ++i) {
        KeyInfo info;
        info.lastUseTime = std::time(nullptr);
        info.recentCount = dis(gen);
        info.isChanged = false;
        info.batchID = i;
        info.totalCount = dis(gen);

        keyInfo[i] = info;
    }

    stringstream savePathOne;
    savePathOne << "test_static/SaveDeltaTrue" << rankInfo_.rankId;
    table->SetFileSystemPtr(savePathOne.str());
    EXPECT_THROW(table->Save(savePathOne.str(), 1, true, keyInfo), std::runtime_error);
    EXPECT_EQ(table->capacity(), testNum);
}

TEST_F(EmbeddingStaticTest, ShouldReturnTargetOffsetWhenFindExistKey)
{
    vector<EmbInfo> embInfos = {embInfo_};
    auto table = std::make_unique<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> targetKey = {3};
    table->Key2Offset(targetKey, TRAIN_CHANNEL_ID);
    EXPECT_EQ(*targetKey.begin(), 0);

    vector<emb_key_t> observeKey = {3};
    table->Key2Offset(observeKey, TRAIN_CHANNEL_ID);
    EXPECT_EQ(*observeKey.begin(), 0);
}

TEST_F(EmbeddingStaticTest, ShouldReturnTargetMaxOffsetWhenFindKeyIn5Threads)
{
    std::vector<std::unique_ptr<std::thread>> threads;
    vector<EmbInfo> embInfos = {embInfo_};
    auto table = std::make_unique<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testData;
    constexpr size_t testNum = 50;
    for (size_t i = 0; i < testNum; ++i) {
        testData.push_back(i);
    }
    table->Key2Offset(testData, TRAIN_CHANNEL_ID);
    auto proc = [&testData, &table]() {
        auto tempData = testData;
        table->Key2Offset(tempData, TRAIN_CHANNEL_ID);
        for (size_t i = 0; i < tempData.size(); ++i) {
            EXPECT_EQ(tempData[i], i);
        }
    };
    constexpr int threadNum = 5;
    for (int i = 0; i < threadNum; ++i) {
        threads.push_back(std::make_unique<std::thread>(proc));
    }
    for (auto& it : threads) {
        it->join();
    }
    EXPECT_EQ(table->size(), testNum);
    EXPECT_EQ(table->size(), table->GetMaxOffset());
}

TEST_F(EmbeddingStaticTest, ShouldReturnTargetMaxOffsetWhenEmplaceKeyIn5Threads)
{
    std::vector<std::unique_ptr<std::thread>> threads;
    vector<EmbInfo> embInfos = {embInfo_};
    auto table = std::make_unique<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    
    auto proc = [&table](int index) {
        vector<emb_key_t> testData;
        for (int i = index * 20; i < (index + 1) * 20; ++i) {
            testData.push_back(i);
        }
        table->Key2Offset(testData, TRAIN_CHANNEL_ID);
    };
    constexpr int threadNum = 5;
    for (int i = 0; i < threadNum; ++i) {
        threads.push_back(std::make_unique<std::thread>(proc, i));
    }
    for (auto& it : threads) {
        it->join();
    }
    EXPECT_EQ(table->size(), 100);
    EXPECT_EQ(table->size(), table->GetMaxOffset());
}

TEST_F(EmbeddingStaticTest, ShouldReturnTargetMaxOffsetWhenEmplaceAndFindKeyIn5Threads)
{
    std::vector<std::unique_ptr<std::thread>> threads;
    vector<EmbInfo> embInfos = {embInfo_};
    auto table = std::make_unique<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testData;
    for (size_t i = 0; i < 100; ++i) {
        testData.push_back(i);
    }

    auto proc = [&testData, &table]() {
        auto tempData = testData;
        table->Key2Offset(tempData, TRAIN_CHANNEL_ID);
        for (size_t i = 0; i < tempData.size(); ++i) {
            EXPECT_EQ(tempData[i], i);
        }
    };
    constexpr int threadNum = 5;
    for (int i = 0; i < threadNum; ++i) {
        threads.push_back(std::make_unique<std::thread>(proc));
    }
    for (auto& it : threads) {
        it->join();
    }
    EXPECT_EQ(table->size(), 100);
    EXPECT_EQ(table->size(), table->GetMaxOffset());
}

TEST_F(EmbeddingStaticTest, TestBackUpAndRecoverTrainStatus)
{
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);

    const size_t testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    table->Key2Offset(testKeys, TRAIN_CHANNEL_ID);
    table->BackUpTrainStatus();
    EXPECT_EQ(table->keyOffsetMapBackUp.size(), testNum);
    table->RecoverTrainStatus();
    EXPECT_EQ(table->keyOffsetMapBackUp.size(), 0);
}

TEST_F(EmbeddingStaticTest, SaveKeyWritingFaild_Error)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = false;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[0] = KeyInfo();
    absl::flat_hash_map<emb_key_t, int64_t> keyMap;
    keyMap[1] = 0;
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    table->keyOffsetMap = keyMap;
    auto localFileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(localFileSys);

    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    using WriteCharFunc = ssize_t (LocalFileSystem::*)(const string&, const char*, size_t);
    EMOCK(static_cast<WriteCharFunc>(&LocalFileSystem::Write)).stubs().will(returnValue(-1));
    EXPECT_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap), std::runtime_error);
}

TEST_F(EmbeddingStaticTest, SaveKeyBytesFaild_Error)
{
    const string savePath = "./test/path";
    const int pythonBatchId = 1;
    bool saveDelta = false;
    map<emb_key_t, KeyInfo> keyInfoMap;
    keyInfoMap[0] = KeyInfo();
    absl::flat_hash_map<emb_key_t, int64_t> keyMap;
    keyMap[1] = 0;
    shared_ptr<EmbeddingStatic> table = std::make_shared<EmbeddingStatic>(embInfo_, rankInfo_, 0);
    table->keyOffsetMap = keyMap;
    auto localFileSys = make_unique<LocalFileSystem>();
    table->fileSystemPtr_ = move(localFileSys);

    EMOCK(&EmbeddingTable::MakeDir).stubs();
    EMOCK(&EmbeddingTable::CheckFileSystemPtr).stubs();
    using WriteCharFunc = ssize_t (LocalFileSystem::*)(const string&, const char*, size_t);
    EMOCK(static_cast<WriteCharFunc>(&LocalFileSystem::Write)).stubs().will(returnValue(0));
    EXPECT_THROW(table->Save(savePath, pythonBatchId, saveDelta, keyInfoMap), std::runtime_error);
}