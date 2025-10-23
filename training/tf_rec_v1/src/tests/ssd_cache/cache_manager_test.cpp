/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>

#include "absl/container/flat_hash_map.h"
#include "l3_storage/lfu_cache.h"
#include "l3_storage/cache_manager.h"
#include "utils/common.h"
#include "emb_cache_manager_mock.h"

using namespace std;
using namespace MxRec;
using namespace testing;

static const string SSD_SAVE_PATH = "savePath1";

static const float EPSILON = 1e-6f;

void InitSSDEngine(CacheManager& manager, string embTableName, uint64_t ssdSize)
{
    // Init ssd engine data
    chrono::seconds period = chrono::seconds(120);
    auto ssdEngine = static_pointer_cast<SSDEngine>(manager.l3Storage);
    ssdEngine->SetCompactPeriod(period);
    ssdEngine->SetCompactThreshold(1);
    ssdEngine->CreateTable(embTableName, {SSD_SAVE_PATH}, ssdSize);
    vector<emb_cache_key_t> ssdKeys = {15, 25}; // 预设15， 25存储在SSD
    auto emb1 = new float(15.0f);
    auto emb2 = new float(25.0f);
    uint64_t extEmbeddingSize = 1;
    std::vector<float*> ssdEmbData = {{emb1}, {emb2}};
    auto& excludeMap = manager.preProcessMapper[embTableName].excludeDDRKeyCountMap;
    excludeMap[15] = 3; // 初始化次数
    excludeMap[25] = 5;
    ssdEngine->InsertEmbeddingsByAddr(embTableName, ssdKeys, ssdEmbData, extEmbeddingSize);
    delete emb1;
    delete emb2;
}

void InitDDREmbData(absl::flat_hash_map<string, HostEmbTable>& loadData, string& embTableName,
                    vector<EmbInfo>& mgmtEmbInfos)
{
    // 构造 HostEmb 对象
    EmbInfo embInfo;
    embInfo.name = embTableName;
    embInfo.hostVocabSize = 20;
    embInfo.devVocabSize = 100;
    embInfo.ssdVocabSize = 100;
    embInfo.ssdDataPath = {SSD_SAVE_PATH};
    mgmtEmbInfos.emplace_back(embInfo);

    std::vector<std::vector<float>> t_embData; // 以DDR vocabSize=100设置
    t_embData.assign(100, {});
    t_embData[0] = {1.0f};
    t_embData[1] = {2.0f};
    t_embData[91] = {3.0f};
    t_embData[92] = {4.0f};
    t_embData[94] = {6.0f};
    t_embData[96] = {8.0f};
    t_embData[97] = {9.0f};
    HostEmbTable hEmbTable = {embInfo, t_embData};
    loadData[embTableName] = hEmbTable;
}

void PutKeyInfo(LFUCache& lfu, vector<emb_key_t>& embKeys)
{
    for (auto& key : embKeys) {
        lfu.Put(key);
    }
}

class CacheManagerTest : public testing::Test {
protected:
    void SetUp()
    {
        // 设置全局rankId，ssdEngine保存时会使用
        int workRankId;
        MPI_Comm_rank(MPI_COMM_WORLD, &workRankId);
        GlogConfig::gRankId = to_string(workRankId);

        cacheManager.ddrKeyFreqMap[embTableName] = cache;
        PutKeyInfo(cacheManager.ddrKeyFreqMap[embTableName], input_keys);
        LFUCache cache2;
        cacheManager.ddrKeyFreqMap[embTableName2] = cache2;
        PutKeyInfo(cacheManager.ddrKeyFreqMap[embTableName2], input_keys);
        unordered_map<emb_cache_key_t, freq_num_t> excludeDDRKeyFreq;
        excludeDDRKeyFreq[27] = 10;
        excludeDDRKeyFreq[30] = 10;
        cacheManager.excludeDDRKeyCountMap[embTableName] = excludeDDRKeyFreq;

        // init cache manager
        vector<EmbInfo> mgmtEmbInfos;
        absl::flat_hash_map<string, HostEmbTable> loadData = {};
        InitDDREmbData(loadData, embTableName, mgmtEmbInfos);
        InitDDREmbData(loadData, embTableName2, mgmtEmbInfos);
        InitDDREmbData(loadData, embTableName3, mgmtEmbInfos);

        m_embCachePtr = std::make_shared<EmbCache::EmbCacheManagerMock>();

        auto ssdEngine = make_shared<SSDEngine>();
        cacheManager.Init(m_embCachePtr, mgmtEmbInfos, ssdEngine);

        InitSSDEngine(cacheManager, embTableName, 5);
        InitSSDEngine(cacheManager, embTableName2, 10);
        // load ddr emb data
    }

    CacheManager cacheManager;
    LFUCache cache;
    std::shared_ptr<EmbCache::EmbCacheManagerMock> m_embCachePtr{nullptr};
    /*
     * 频次-对应key列表
     * 1 - 9,8
     * 2 - 6,4
     * 3 - 3,2,1
     */
    vector<emb_key_t> input_keys = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 6, 6, 8, 9};
    string embTableName = "table1";
    string embTableName2 = "table2";
    string embTableName3 = "table3";

    void TearDown()
    {
    }
};

TEST_F(CacheManagerTest, PutKey)
{
    vector<emb_key_t> putDDRKeys = {1, 9, 8, 15};
    for (auto& key : putDDRKeys) {
        cacheManager.PutKey(embTableName, key, RecordType::DDR);
    }
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 1);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].freqTable[1].size(), 1);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(15), 1);
    LOG_INFO("test PutKey end.");
}

TEST_F(CacheManagerTest, IsKeyInL3Storage)
{
    vector<emb_key_t> checkKeys = {1, 2, 15, 25};
    ASSERT_FALSE(cacheManager.IsKeyInL3Storage(embTableName, checkKeys[0]));
    ASSERT_FALSE(cacheManager.IsKeyInL3Storage(embTableName, checkKeys[1]));
    ASSERT_TRUE(cacheManager.IsKeyInL3Storage(embTableName, checkKeys[2]));
    ASSERT_TRUE(cacheManager.IsKeyInL3Storage(embTableName, checkKeys[3]));
    LOG_INFO("test IsKeyInL3Storage end.");
}

TEST_F(CacheManagerTest, EvictL3StorageEmbedding)
{
    // 构造时ssd中已存在的key: 15 25
    emb_cache_key_t key = 15;
    vector<emb_cache_key_t> ssdKeys = {key};
    cacheManager.EvictL3StorageEmbedding(embTableName, ssdKeys);
    int maxLoop = 1000;
    while (!cacheManager.l3StorageEvictThreads.empty() && maxLoop > 0) {
        this_thread::sleep_for(1ms);
        maxLoop--;
    }
    ASSERT_FALSE(cacheManager.IsKeyInL3Storage(embTableName, key));
    const auto it = cacheManager.excludeDDRKeyCountMap[embTableName].find(key);
    ASSERT_EQ(it, cacheManager.excludeDDRKeyCountMap[embTableName].end());
    LOG_INFO("test EvictL3StorageEmbedding end.");
}

TEST_F(CacheManagerTest, LoadTest)
{
}

TEST_F(CacheManagerTest, TransferDDR2L3Storage)
{
    std::vector<emb_cache_key_t> ssdKeys = {15, 25}; // 预设15， 25存储在SSD
    auto emb1 = (float*)malloc(sizeof(float));
    if (emb1 == nullptr) {
        return;
    }
    auto emb2 = (float*)malloc(sizeof(float));
    if (emb2 == nullptr) {
        free(emb1);
        return;
    }
    *emb1 = 15.0f;
    *emb2 = 25.0f;
    uint64_t extEmbeddingSize = 1;
    std::vector<float*> ssdEmbData = {{emb1}, {emb2}};
    cacheManager.TransferDDR2L3Storage(embTableName3, extEmbeddingSize, ssdKeys, ssdEmbData);

    EXPECT_TRUE(cacheManager.IsKeyInL3Storage(embTableName3, ssdKeys[0]));
    EXPECT_TRUE(cacheManager.IsKeyInL3Storage(embTableName3, ssdKeys[1]));
}

TEST_F(CacheManagerTest, GetTableUsage)
{
    auto usage = cacheManager.GetTableUsage(embTableName);
    EXPECT_EQ(usage, 2); // expect usage is 2
}

TEST_F(CacheManagerTest, ProcessSwapOutKeys)
{
    std::vector<emb_cache_key_t> swapOutKeys = {1, 25, 10};
    HBMSwapOutInfo hbmSwapInfo;
    cacheManager.ProcessSwapOutKeys(embTableName, swapOutKeys, hbmSwapInfo);

    EXPECT_EQ(hbmSwapInfo.swapOutDDRKeys[0], swapOutKeys[0]);
    EXPECT_EQ(hbmSwapInfo.swapOutDDRAddrOffs[0], 0);
    EXPECT_EQ(hbmSwapInfo.swapOutL3StorageKeys[0], swapOutKeys[1]);
    EXPECT_EQ(hbmSwapInfo.swapOutL3StorageAddrOffs[0], 1);
}

TEST_F(CacheManagerTest, ProcessSwapInKeys)
{
    std::vector<emb_cache_key_t> swapInKeys = {1, 25};
    std::vector<emb_cache_key_t> l3StorageToDDRKeys;
    std::vector<emb_cache_key_t> ddrToL3StorageKeys;

    cacheManager.ProcessSwapInKeys(embTableName, swapInKeys, ddrToL3StorageKeys, l3StorageToDDRKeys);
    EXPECT_EQ(l3StorageToDDRKeys.size(), 1);
    EXPECT_EQ(l3StorageToDDRKeys[0], swapInKeys[1]);
    EXPECT_EQ(ddrToL3StorageKeys.size(), 0);
}

TEST_F(CacheManagerTest, UpdateL3StorageEmb)
{
    float emb1 = 30.0f;
    uint64_t extEmbeddingSize = 1;
    std::vector<emb_cache_key_t> ssdKeys = {30};
    std::vector<uint64_t> swapOutL3StorageOffset = {0};

    cacheManager.UpdateL3StorageEmb(embTableName, &emb1, extEmbeddingSize, ssdKeys, swapOutL3StorageOffset);
    EXPECT_TRUE(cacheManager.IsKeyInL3Storage(embTableName, ssdKeys[0]));
}

TEST_F(CacheManagerTest, FetchL3StorageEmb2DDR)
{
    uint64_t extEmbeddingSize = 1;
    float emb{0};
    std::vector<float*> ssdEmbData = {&emb};
    std::vector<emb_cache_key_t> ssdKeys{15};

    cacheManager.FetchL3StorageEmb2DDR(embTableName, extEmbeddingSize, ssdKeys, ssdEmbData);
    EXPECT_EQ(emb, 15.0f);
}

TEST_F(CacheManagerTest, BackUpTrainStatus)
{
    cacheManager.BackUpTrainStatus();

    EXPECT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 1);
    EXPECT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(1), 3); // expect key 1 freq is 3

    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).Times(1).WillRepeatedly(Return(0));
    EXPECT_CALL(*m_embCachePtr, EmbeddingRemove(_, _, _)).Times(1).WillRepeatedly(Return(0));
    EXPECT_CALL(*m_embCachePtr, EmbeddingUpdate(_, _, _, _)).Times(1).WillRepeatedly(Return(0));
    cacheManager.RecoverTrainStatus();
}

TEST_F(CacheManagerTest, InsertL3StorageKey)
{
    uint64_t key = 10;
    cacheManager.preProcessMapper[embTableName].InsertL3StorageKey(key);
    auto space = cacheManager.preProcessMapper[embTableName].L3StorageAvailableSize();

    EXPECT_EQ(space, 100 - 3); // expect space is ssdVocabSize : 100 - usage : 3

    size_t transNum = 1;
    std::vector<uint64_t> swapInKeys = {9};
    std::vector<uint64_t> dDRSwapOutKeys;
    try {
        auto processMapper = cacheManager.preProcessMapper[embTableName];
        processMapper.GetAndDeleteLeastFreqDDRKey2L3Storage(transNum, swapInKeys, dDRSwapOutKeys);
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(dDRSwapOutKeys.size(), 0);
    }
}

TEST_F(CacheManagerTest, InsertL3StorageExistKey)
{
    uint64_t key = 15;
    try {
        cacheManager.preProcessMapper[embTableName].InsertL3StorageKey(key);
    } catch (const std::invalid_argument& e) {
        EXPECT_TRUE(cacheManager.IsKeyInL3Storage(embTableName, key));
    }
}

TEST_F(CacheManagerTest, L3StorageUnavailableSize)
{
    cacheManager.preProcessMapper[embTableName].l3StorageAvailableSize = 1;
    size_t space = 0;
    try {
        space = cacheManager.preProcessMapper[embTableName].L3StorageAvailableSize();
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(space, 0);
    }
}