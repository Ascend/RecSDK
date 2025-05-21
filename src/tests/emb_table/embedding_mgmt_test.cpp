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
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
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

/**
 * 测试GetKeyOffsetMap
 */
TEST_F(EmbeddingMgmtTest, GetKeyOffsetMap)
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

    int exp = 0;
    try {
        MxRec::KeyOffsetMemT kom = EmbeddingMgmt::Instance()->GetKeyOffsetMap();
    } catch (exception& e) {
        exp = 1;
    }
    map<EmbNameT, size_t> maxOffsetMap = EmbeddingMgmt::Instance()->GetMaxOffset();

    EXPECT_EQ(maxOffsetMap[tableName], testNum);
    EXPECT_EQ(exp, 0);
}

/**
 * 测试EvictKeys与EvictKeysCombine
 */
TEST_F(EmbeddingMgmtTest, EvictKeys)
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

/**
 * 测试Save与Load:单表
 */
TEST_F(EmbeddingMgmtTest, SaveAndLoadSingleTable)
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

    EmbeddingMgmt::Instance()->Save(tableName, "test_dir", 1);
    const char* filePath = "./test_dir/test1/key/slice_0.data";
    // 检查文件是否存在
    if (access(filePath, F_OK) != 0) {
        LOG_INFO("Error: File does not exist: {}", filePath);
    } else {
        // 重命名文件
        const char* newfilePath = "./test_dir/test1/key/slice.data";
        if (rename(filePath, newfilePath) == 0) {
            LOG_INFO("File renamed successfully: {}", newfilePath);
        }
    }

    if (rankInfo_.rankId == 0) {
        map<string, unordered_set<emb_cache_key_t>> trainKeySetOne;
        vector<string> warmStartTablesOne;
        EmbeddingMgmt::Instance()->Load(tableName, "./test_dir", trainKeySetOne, warmStartTablesOne);

        bool fileExist = false;
        if (access("./test_dir/test1/key/slice.data", F_OK) == 0) {
            fileExist = true;
        }

        EXPECT_EQ(fileExist, true);
        EXPECT_EQ(EmbeddingMgmt::Instance()->GetMaxOffset(tableName), testNum);
    }
}

/**
 * 测试Save与Load:多表
 */
TEST_F(EmbeddingMgmtTest, SaveAndLoadAllTable)
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
    EmbeddingMgmt::Instance()->Save("test_dir", 1, true, keyInfoMap);

    const char* filePath = "./test_dir/test1/key/slice_0.data";
    // 检查文件是否存在
    if (access(filePath, F_OK) != 0) {
        LOG_INFO("Error: File does not exist: {}", filePath);
    } else {
        // 重命名文件
        const char* newfilePath = "./test_dir/test1/key/slice.data";
        if (rename(filePath, newfilePath) == 0) {
            LOG_INFO("File renamed successfully: {}", newfilePath);
        }
    }

    if (rankInfo_.rankId == 0) {
        map<string, unordered_set<emb_cache_key_t>> trainKeySetOne;
        vector<string> warmStartTablesOne;
        EmbeddingMgmt::Instance()->Load("./test_dir", trainKeySetOne, warmStartTablesOne);

        bool fileExist = false;
        if (access("./test_dir/test1/key/slice.data", F_OK) == 0) {
            fileExist = true;
        }

        map<EmbNameT, size_t> maxOffsetMap = EmbeddingMgmt::Instance()->GetMaxOffset();
        OffsetMapT allDeviceOffsets = EmbeddingMgmt::Instance()->GetDeviceOffsets();

        EXPECT_EQ(allDeviceOffsets[tableName].size(), testNum);
        EXPECT_EQ(maxOffsetMap[tableName], testNum);
        EXPECT_EQ(fileExist, true);
    }
}

/**
 * 测试Key2OffsetForDp:使用 eval channel
 */
TEST_F(EmbeddingMgmtTest, Key2OffsetForDpEval)
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