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
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <limits>
#include <mpi.h>

#include "utils/common.h"
#include "utils/logger.h"
#include "utils/error.h"
#include "emb_table/embedding_mgmt.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingMgmtTest : public testing::Test {
protected:
    EmbeddingMgmtTest()
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
        rankInfo_.isDDR = false;
    }

    void SetUp() {
    }
    void TearDown() {
    }

    static void SetupTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    static void TearDownTestCase()
    {
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
};

TEST_F(EmbeddingMgmtTest, Init)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2Offset(tableName, testKeys, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], i);
    }
    EXPECT_EQ(EmbeddingMgmt::Instance()->GetMaxOffset(tableName), testNum);
}

TEST_F(EmbeddingMgmtTest, GetAttributes)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2Offset(tableName, testKeys, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], i);
    }
    unordered_set<int64_t> paddingKeysOffset = EmbeddingMgmt::Instance()->GetPaddingKeysOffset(tableName);

    EXPECT_EQ(EmbeddingMgmt::Instance()->GetMaxOffset(tableName), testNum);
    EXPECT_EQ(EmbeddingMgmt::Instance()->GetSize(tableName), testNum);
    EXPECT_EQ(EmbeddingMgmt::Instance()->GetCapacity(tableName), testNum);
    EXPECT_EQ(paddingKeysOffset.size(), 0);
}

TEST_F(EmbeddingMgmtTest, TestGetKeyOffsetMapAndNoError)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2Offset(tableName, testKeys, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], i);
    }

    MxRec::KeyOffsetMemT kom = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
    map<EmbNameT, size_t> maxOffsetMap = EmbeddingMgmt::Instance()->GetMaxOffset();
    EXPECT_EQ(kom.size(), 1);
    EXPECT_EQ(maxOffsetMap[tableName], testNum);
}

TEST_F(EmbeddingMgmtTest, TestEvictKeysAndEvictKeysCombine)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2Offset(tableName, testKeys, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], i);
    }

    size_t partialNum = testNum / 4;
    size_t startIdx = testNum - partialNum;
    vector<emb_cache_key_t> testKeysBakOne(testKeys.begin(), testKeys.begin() + partialNum);
    vector<emb_cache_key_t> testKeysBakTwo(testKeys.begin() + startIdx, testKeys.end());

    EmbeddingMgmt::Instance()->EvictKeys(tableName, testKeysBakOne);
    EmbeddingMgmt::Instance()->EvictKeysCombine(testKeysBakTwo);

    MxRec::KeyOffsetMemT kom = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
    LOG_INFO("test Key2Offset: lookupKeys: {}, kom[tableName] size: {}",
             VectorToString(testKeys), kom[tableName].size());

    EXPECT_EQ(kom[tableName].size(), testNum - partialNum - partialNum);
}

TEST_F(EmbeddingMgmtTest, TestSaveAndLoadWhenSingleTable)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2Offset(tableName, testKeys, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], i);
    }

    stringstream savePathOne;
    savePathOne << "test_dir/SingleTable" << rankInfo_.rankId;
    EmbeddingMgmt::Instance()->Save(tableName, savePathOne.str(), 1);
    stringstream saveKeyPathOne;
    saveKeyPathOne << savePathOne.str() << "/" << tableName << "/key";
    EXPECT_EQ(access(saveKeyPathOne.str().c_str(), F_OK), 0);

    stringstream fileKeyPathOne;
    fileKeyPathOne << saveKeyPathOne.str() << "/slice_" << rankInfo_.rankId << ".data";
    stringstream newfileKeyPathOne;
    newfileKeyPathOne << saveKeyPathOne.str() << "/slice.data";
    RenameFilePath(fileKeyPathOne.str(), newfileKeyPathOne.str());

    if (rankInfo_.rankId == 0) {
        map<string, unordered_set<emb_cache_key_t>> trainKeySetOne;
        vector<string> warmStartTablesOne;
        EmbeddingMgmt::Instance()->Load(tableName, savePathOne.str(), trainKeySetOne, warmStartTablesOne);
        EXPECT_EQ(EmbeddingMgmt::Instance()->GetMaxOffset(tableName), testNum);
    }
}

TEST_F(EmbeddingMgmtTest, TestSaveWhenMultiTable)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2Offset(tableName, testKeys, TRAIN_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], i);
    }

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

    map<string, map<emb_key_t, KeyInfo>> keyInfoMap;
    keyInfoMap[tableName] = keyInfo;
    stringstream savePathTwo;
    savePathTwo << "test_dir/MultiTable" << rankInfo_.rankId;
    EmbeddingMgmt::Instance()->Save(savePathTwo.str(), 1, true, keyInfoMap);
    stringstream saveKeyPathTwo;
    saveKeyPathTwo << savePathTwo.str() << "/" << tableName << "/key";
    OffsetMapT allDeviceOffsets = EmbeddingMgmt::Instance()->GetDeviceOffsets();

    EXPECT_EQ(allDeviceOffsets[tableName].size(), testNum);
    EXPECT_EQ(access(saveKeyPathTwo.str().c_str(), F_OK), 0);
}

TEST_F(EmbeddingMgmtTest, TestLoadWhenMultiTable)
{
    const string tableName = "test1";
    constexpr int testNum = 100;

    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    stringstream savePathTwo;
    savePathTwo << "test_dir/MultiTable" << rankInfo_.rankId;
    stringstream saveKeyPathTwo;
    saveKeyPathTwo << savePathTwo.str() << "/" << tableName << "/key";
    stringstream fileKeyPathTwo;
    fileKeyPathTwo << saveKeyPathTwo.str() << "/slice_" << rankInfo_.rankId << ".data";
    stringstream newfileKeyPathTwo;
    newfileKeyPathTwo << saveKeyPathTwo.str() << "/slice.data";

    if (rankInfo_.rankId==0) {
        RenameFilePath(fileKeyPathTwo.str(), newfileKeyPathTwo.str());

        map<string, unordered_set<emb_cache_key_t>> trainKeySetOne;
        vector<string> warmStartTablesOne;
        EmbeddingMgmt::Instance()->Load(savePathTwo.str(), trainKeySetOne, warmStartTablesOne);
        map<EmbNameT, size_t> maxOffsetMap = EmbeddingMgmt::Instance()->GetMaxOffset();
        EXPECT_EQ(maxOffsetMap[tableName], testNum / rankInfo_.rankSize);
    }
}

TEST_F(EmbeddingMgmtTest, TestKey2OffsetForDpWhenUseEvalChannel)
{
    const string tableName = "test1";
    ThresholdValue thvalue(tableName, 0, 0, 0, false);
    vector<EmbInfo> embInfos = {embInfo_};
    vector<ThresholdValue> thresholds = {thvalue};
    EmbeddingMgmt::Instance()->Init(rankInfo_, embInfos, 0);

    constexpr int testNum = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < testNum; ++i) {
        testKeys.push_back(i);
    }
    EmbeddingMgmt::Instance()->Key2OffsetForDp(tableName, testKeys, EVAL_CHANNEL_ID);
    for (size_t i = 0; i < testNum; ++i) {
        EXPECT_EQ(testKeys[i], INVALID_KEY_VALUE);
    }

    EXPECT_EQ(EmbeddingMgmt::Instance()->GetCapacity(tableName), testNum);
}