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

#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>

#include "absl/container/flat_hash_map.h"
#include "ssd_cache/lfu_cache.h"
#include "ssd_cache/cache_manager.h"
#include "utils/common.h"

using namespace std;
using namespace MxRec;
using namespace testing;

static const string SSD_SAVE_PATH = "savePath1";

static const float EPSILON = 1e-6f;

void InitSSDEngine(CacheManager& manager, string embTableName, uint64_t ssdSize)
{
    // Init ssd engine data
    chrono::seconds period = chrono::seconds(120);
    manager.ssdEngine->SetCompactPeriod(period);
    manager.ssdEngine->SetCompactThreshold(1);
    manager.ssdEngine->CreateTable(embTableName, {SSD_SAVE_PATH}, ssdSize);
    vector<emb_cache_key_t> ssdKeys = {15, 25}; // 预设15， 25存储在SSD
    std::vector<std::vector<float>> ssdEmbData = {{15.0f},
                                                  {25.0f}};
    auto& excludeMap = manager.preProcessMapper[embTableName].excludeDDRKeyCountMap;
    excludeMap[15] = 3; // 初始化次数
    excludeMap[25] = 5;
    manager.ssdEngine->InsertEmbeddings(embTableName, ssdKeys, ssdEmbData);
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

        ock::ctr::EmbCacheManagerPtr embCachePtr = nullptr;

        cacheManager.Init(embCachePtr, mgmtEmbInfos);

        InitSSDEngine(cacheManager, embTableName, 5);
        InitSSDEngine(cacheManager, embTableName2, 10);
        // load ddr emb data
    }

    CacheManager cacheManager;
    LFUCache cache;
    /*
     * 频次-对应key列表
     * 1 - 9,8
     * 2 - 6,4
     * 3 - 3,2,1
     */
    vector<emb_key_t> input_keys = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 6, 6, 8, 9};
    string embTableName = "table1";
    string embTableName2 = "table2";

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

TEST_F(CacheManagerTest, IsKeyInSSD)
{
    vector<emb_key_t> checkKeys = {1, 2, 15, 25};
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, checkKeys[0]));
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, checkKeys[1]));
    ASSERT_TRUE(cacheManager.IsKeyInSSD(embTableName, checkKeys[2]));
    ASSERT_TRUE(cacheManager.IsKeyInSSD(embTableName, checkKeys[3]));
    LOG_INFO("test IsKeyInSSD end.");
}

TEST_F(CacheManagerTest, EvictSSDEmbedding)
{
    // 构造时ssd中已存在的key: 15 25
    emb_cache_key_t key = 15;
    vector<emb_cache_key_t> ssdKeys = {key};
    cacheManager.EvictSSDEmbedding(embTableName, ssdKeys);
    int maxLoop = 1000;
    while (!cacheManager.ssdEvictThreads.empty() && maxLoop > 0) {
        this_thread::sleep_for(1ms);
        maxLoop--;
    }
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, key));
    const auto it = cacheManager.excludeDDRKeyCountMap[embTableName].find(key);
    ASSERT_EQ(it, cacheManager.excludeDDRKeyCountMap[embTableName].end());
    LOG_INFO("test EvictSSDEmbedding end.");
}

TEST_F(CacheManagerTest, LoadTest)
{
}