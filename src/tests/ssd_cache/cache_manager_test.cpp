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
#include "host_emb/host_emb.h"
#include "ssd_cache/lfu_cache.h"
#include "ssd_cache/cache_manager.h"
#include "utils/common.h"
#include "emb_table/embedding_ddr.h"

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
    vector<emb_key_t> ssdKeys = {15, 25}; // 预设15， 25存储在SSD
    std::vector<std::vector<float>> ssdEmbData = {{15.0f},
                                                  {25.0f}};
    auto& excludeMap = manager.excludeDDRKeyCountMap[embTableName];
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
        unordered_map<emb_key_t, freq_num_t> excludeDDRKeyFreq;
        excludeDDRKeyFreq[27] = 10;
        excludeDDRKeyFreq[30] = 10;
        cacheManager.excludeDDRKeyCountMap[embTableName] = excludeDDRKeyFreq;

        // init cache manager
        vector<EmbInfo> mgmtEmbInfos;
        absl::flat_hash_map<string, HostEmbTable> loadData = {};
        InitDDREmbData(loadData, embTableName, mgmtEmbInfos);
        InitDDREmbData(loadData, embTableName2, mgmtEmbInfos);

        cacheManager.Init(hEmb, mgmtEmbInfos);

        InitSSDEngine(cacheManager, embTableName, 5);
        InitSSDEngine(cacheManager, embTableName2, 10);
        // load ddr emb data
        cacheManager.hostEmbs->hostEmbs = loadData;

        auto& embMap = cacheManager.hostEmbs->hostEmbs;
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
    HostEmb* hEmb = Singleton<MxRec::HostEmb>::GetInstance();

    void TearDown()
    {
    }
};

TEST_F(CacheManagerTest, RefreshFreqInfo)
{
    vector<emb_key_t> ddr2HbmKeys = {8, 9};
    cacheManager.RefreshFreqInfoCommon(embTableName, ddr2HbmKeys, TransferType::DDR_2_HBM);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 2);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].keyTable.size(), 5);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].freqTable.size(), 2);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(8), -1);
    ASSERT_EQ(cacheManager.excludeDDRKeyCountMap[embTableName].size(), 6);

    // HBM转移到DDR 频次数据设置构造
    cacheManager.excludeDDRKeyCountMap[embTableName][150] = 4;
    cacheManager.excludeDDRKeyCountMap[embTableName][151] = 1;
    vector<emb_key_t> hbm2DdrKeys = {150, 151};
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(151), -1);
    cacheManager.RefreshFreqInfoCommon(embTableName, hbm2DdrKeys, TransferType::HBM_2_DDR);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(150), 4);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(151), 1);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 1);
    ASSERT_EQ(cacheManager.excludeDDRKeyCountMap[embTableName].size(), 6);

    vector<emb_key_t> ddr2EvictKeys = {151};
    cacheManager.RefreshFreqInfoCommon(embTableName, ddr2EvictKeys, TransferType::DDR_2_EVICT);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(151), -1);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].freqTable.size(), 3);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 2);

    // HBM2Evict
    cacheManager.excludeDDRKeyCountMap[embTableName][160] = 1;
    vector<emb_key_t> hbm2EvictKeys = {160};
    cacheManager.RefreshFreqInfoCommon(embTableName, hbm2EvictKeys, TransferType::HBM_2_EVICT);
    const auto it = cacheManager.excludeDDRKeyCountMap[embTableName].find(160);
    ASSERT_EQ(it, cacheManager.excludeDDRKeyCountMap[embTableName].end());
    LOG_INFO("test RefreshFreqInfo end.");
}

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

TEST_F(CacheManagerTest, TransferDDREmbWithSSDByEmptyExternalKey)
{
    EmbeddingDDR table;

    vector<emb_key_t> currentKeys = {55, 65, 75};
    table.keyOffsetMap[55] = 119;
    table.keyOffsetMap[65] = 118;
    table.keyOffsetMap[75] = 116;

    TableInfo ti = table.GetTableInfo();

    auto ret = cacheManager.TransferDDREmbWithSSD(ti, currentKeys, TRAIN_CHANNEL_ID);
    ASSERT_EQ(ret, TransferRet::TRANSFER_OK);
    LOG_INFO("test TransferDDREmbWithSSDByEmptyExternalKey end.");
}

TEST_F(CacheManagerTest, TransferDDREmbWithSSDByAllProcess)
{
    vector<emb_key_t> ssdKeys = {15, 25};
    vector<vector<float>> ssdKeyEmbInfo = {{1.5f}, {2.5f}};

    // init EmbeddingDDR
    EmbeddingDDR table;
    table.name = embTableName;
    table.devVocabSize = 20;
    table.hostVocabSize = 100;
    table.maxOffset = 118;
    table.evictHostPos.emplace_back(110); // 淘汰列表

    TableInfo ti = table.GetTableInfo();

    // 构造已经存储早DDR中key和offset对应关系; DDR的offset在映射表中范围是 20~119
    table.keyOffsetMap[9] = 117; // DDR中相对位置: 97
    table.keyOffsetMap[8] = 116; // DDR中相对位置: 96
    table.keyOffsetMap[6] = 114; // DDR中相对位置: 94
    table.keyOffsetMap[4] = 112; // DDR中相对位置: 92
    table.keyOffsetMap[3] = 111; // DDR中相对位置: 91
    table.keyOffsetMap[2] = 21; // DDR中相对位置: 1
    table.keyOffsetMap[1] = 20; // DDR中相对位置: 0

    // 检查构造数据正确性
    auto& embMap = cacheManager.hostEmbs->hostEmbs;
    const auto& it = embMap.find(embTableName);
    auto& hostData = it->second.embData;
    ASSERT_TRUE(fabs(hostData[0][0] - 1.0f) < EPSILON);
    ASSERT_TRUE(fabs(hostData[1][0] - 2.0f) < EPSILON);
    ASSERT_TRUE(fabs(hostData[94][0] - 6.0f) < EPSILON);
    ASSERT_TRUE(fabs(hostData[97][0] - 9.0f) < EPSILON);
    auto& excludeKeyCountMap = cacheManager.excludeDDRKeyCountMap[embTableName];
    ASSERT_EQ(excludeKeyCountMap[15], 3);
    ASSERT_EQ(excludeKeyCountMap[25], 5);
    ASSERT_FALSE(cacheManager.ssdEngine->IsKeyExist(embTableName, 9));
    ASSERT_FALSE(cacheManager.ssdEngine->IsKeyExist(embTableName, 8));
    ASSERT_TRUE(cacheManager.IsKeyInSSD(embTableName, 15));

    // externalKeys: SSD(15, 25) + newKey(55, 65, 75)
    // 训练场景，构造结果：offsetAvailableSize=20+100-118+evictPos.size()=3
    // cacheManager中的频次数据(低-高): 9 8 6 4 3 2 1
    // 构造空间超出SSD可用上限
    vector<emb_key_t> exceedKeys = {15, 25, 6, 4, 55, 65, 75, 85, 95, 105, 115};
    auto spaceError1 = cacheManager.TransferDDREmbWithSSD(ti, exceedKeys, TRAIN_CHANNEL_ID);
    ASSERT_EQ(spaceError1, TransferRet::SSD_SPACE_NOT_ENOUGH);

    // 构造训练+超SSD可用+当前批次中不包含报错在SSD的key
    vector<emb_key_t> keys2 = {6, 4, 55, 65, 75, 85, 95, 105, 115, 125, 135};
    auto spaceError2 = cacheManager.TransferDDREmbWithSSD(ti, exceedKeys, TRAIN_CHANNEL_ID);
    ASSERT_EQ(spaceError2, TransferRet::SSD_SPACE_NOT_ENOUGH);

    // 构造当前批次key 存储位置: SSD(15, 25) DDR(6, 4) newKey(55, 65, 75)
    vector<emb_key_t> currentKeys = {15, 25, 6, 4, 55, 65, 75};
    // 需要从ddr转移4个key到ssd, 低频数据中6 4在当前批次key中,不会被转移,构造的数据转移key:9, 8, 3, 2
    auto ret = cacheManager.TransferDDREmbWithSSD(ti, currentKeys, TRAIN_CHANNEL_ID);

    // 检查处理后数据正确性
    ASSERT_EQ(ret, TransferRet::TRANSFER_OK);
    ASSERT_TRUE(fabs(hostData[94][0] - 6.0f) < EPSILON); // DDR内未移动的数据
    ASSERT_TRUE(fabs(hostData[96][0] - 25.0f) < EPSILON); // SSD转移到DDR的数据
    ASSERT_TRUE(fabs(hostData[97][0] - 15.0f) < EPSILON); // SSD转移到DDR的数据
    ASSERT_EQ(table.evictHostPos.size(), 1);
    ASSERT_EQ(table.evictHostPos.back(), 110);

    // 原DDR中最小频次key(9,8)次数(1)被转移到SSD,SSD转移到DDR的key(15,25)次数(3,5), DDR内频次索引应变为2
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 2);
    ASSERT_TRUE(cacheManager.IsKeyInSSD(embTableName, 9));
    ASSERT_TRUE(cacheManager.IsKeyInSSD(embTableName, 8));
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, 15));
    LOG_INFO("test TransferDDREmbWithSSDByAllProcess end.");
}

TEST_F(CacheManagerTest, TransferDDREmbWithSSDByEmptyExternalSSDKey)
{
    // 训练+评估：构造DDR剩余空间足够，externalSSDKeys为空
    EmbeddingDDR table;
    table.name = embTableName;
    table.devVocabSize = 20;
    table.hostVocabSize = 100;
    table.keyOffsetMap[6] = 114; // DDR中相对位置: 94
    table.keyOffsetMap[4] = 112; // DDR中相对位置: 92
    // 剩余3个可用空间(DDR剩余2个, 相对位置:98 99； DDR淘汰列表1个)
    table.maxOffset = 118;
    table.evictHostPos.emplace_back(110);

    TableInfo ti = table.GetTableInfo();

    vector<emb_key_t> currentKeys = {6, 4, 55, 65, 75};
    auto ret = cacheManager.TransferDDREmbWithSSD(ti, currentKeys, TRAIN_CHANNEL_ID);
    ASSERT_EQ(ret, TransferRet::TRANSFER_OK);
    auto retByEval = cacheManager.TransferDDREmbWithSSD(ti, currentKeys, EVAL_CHANNEL_ID);
    ASSERT_EQ(retByEval, TransferRet::TRANSFER_OK);

    // 评估场景， DDR剩余空间不足， externalSSDKeys为空
    vector<emb_key_t> currentKeys2 = {6, 4, 55, 65, 75, 85, 95, 105, 115};
    auto ret2 = cacheManager.TransferDDREmbWithSSD(ti, currentKeys2, EVAL_CHANNEL_ID);
    ASSERT_EQ(ret2, TransferRet::TRANSFER_OK);
    // 训练场景，返回ssd空间不足
    auto ret3 = cacheManager.TransferDDREmbWithSSD(ti, currentKeys2, TRAIN_CHANNEL_ID);
    ASSERT_EQ(ret3, TransferRet::SSD_SPACE_NOT_ENOUGH);
    LOG_INFO("test TransferDDREmbWithSSDByEmptyExternalSSDKey end.");
}

TEST_F(CacheManagerTest, TransferDDREmbWithSSDByEval)
{
    // 评估+DDR剩余空间足够+externalSSDKeys为空
    EmbeddingDDR table;
    table.name = embTableName;
    table.devVocabSize = 20;
    table.hostVocabSize = 100;
    table.keyOffsetMap[9] = 117; // DDR中相对位置: 97
    table.keyOffsetMap[8] = 116; // DDR中相对位置: 96
    table.keyOffsetMap[6] = 114; // DDR中相对位置: 94
    table.keyOffsetMap[4] = 112; // DDR中相对位置: 92
    // 剩余3个可用空间(DDR剩余2个, 相对位置:98 99； DDR淘汰列表1个)
    table.maxOffset = 118;
    table.evictHostPos.emplace_back(110); // 淘汰列表

    TableInfo ti = table.GetTableInfo();

    vector<emb_key_t> currentKeys = {6, 4, 55, 65, 75};
    auto ret = cacheManager.TransferDDREmbWithSSD(ti, currentKeys, EVAL_CHANNEL_ID);
    ASSERT_EQ(ret, TransferRet::TRANSFER_OK);
    LOG_INFO("test eval+space enough+externalSSDKeysEmpty ok.");

    // 评估+DDR剩余空间足够+externalSSDKeys非空
    vector<emb_key_t> currentKeys2 = {15, 25, 6, 4, 55, 65, 75, 85, 95, 105, 115};
    auto ret2 = cacheManager.TransferDDREmbWithSSD(ti, currentKeys2, EVAL_CHANNEL_ID);
    ASSERT_EQ(ret2, TransferRet::TRANSFER_OK);
    // 检查处理后数据正确性
    const auto& it = cacheManager.hostEmbs->hostEmbs.find(embTableName);
    auto& hostData = it->second.embData;
    ASSERT_TRUE(fabs(hostData[94][0] - 6.0f) < EPSILON); // DDR内未移动的数据
    ASSERT_TRUE(fabs(hostData[98][0] - 25.0f) < EPSILON); // SSD转移到DDR的数据
    ASSERT_TRUE(fabs(hostData[90][0] - 15.0f) < EPSILON); // SSD转移到DDR的数据
    ASSERT_EQ(table.evictHostPos.size(), 0);
    // 原DDR中最小频次key(9,8)次数(1)被转移到SSD,SSD转移到DDR的key(15,25)次数(3,5), DDR内频次索引应变为2
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].minFreq, 1);
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, 9));
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, 8));
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, 15));
    LOG_INFO("test eval+space enough+externalSSDKeysNotEmpty ok.");
}

TEST_F(CacheManagerTest, TransferDDREmbWithSSDByDDRSpaceNotEnough)
{
    // 构造DDR所有空间不满足存放当前批次数据
    EmbeddingDDR table;
    table.name = embTableName2;
    table.devVocabSize = 20;
    table.hostVocabSize = 10;
    table.maxOffset = 30;
    table.keyOffsetMap[6] = 9;
    table.keyOffsetMap[4] = 8;

    TableInfo ti = table.GetTableInfo();

    // keys size:10, ddr keys:2 externalKeys:8 externalSSDKeys:0
    vector<emb_key_t> currentKeys = {6, 4, 101, 102, 103, 104, 105, 106, 107, 108};
    auto ret = cacheManager.TransferDDREmbWithSSD(ti, currentKeys, TRAIN_CHANNEL_ID);
    ASSERT_EQ(ret, TransferRet::DDR_SPACE_NOT_ENOUGH);
    LOG_INFO("test train+ddr space enough+externalSSDKeysEmpty ok.");
}

TEST_F(CacheManagerTest, EvictSSDEmbedding)
{
    // 构造时ssd中已存在的key: 15 25
    emb_key_t key = 15;
    vector<emb_key_t> ssdKeys = {key};
    cacheManager.EvictSSDEmbedding(embTableName, ssdKeys);
    ASSERT_FALSE(cacheManager.IsKeyInSSD(embTableName, key));
    const auto it = cacheManager.excludeDDRKeyCountMap[embTableName].find(key);
    ASSERT_EQ(it, cacheManager.excludeDDRKeyCountMap[embTableName].end());
    LOG_INFO("test EvictSSDEmbedding end.");
}

TEST_F(CacheManagerTest, LoadTest)
{
    cacheManager.ddrKeyFreqMap.clear();
    cacheManager.excludeDDRKeyCountMap.clear();
    unordered_map<std::string, unordered_map<emb_key_t, freq_num_t>> ddrMap;
    string embTableName = "table1";
    unordered_map<emb_key_t, freq_num_t> ddrTableMap;
    ddrTableMap.emplace(1, 3);
    ddrTableMap.emplace(2, 3);
    ddrTableMap.emplace(3, 3);
    ddrTableMap.emplace(4, 2);
    ddrTableMap.emplace(6, 2);
    ddrTableMap.emplace(8, 1);
    ddrTableMap.emplace(9, 1);
    ddrMap.emplace(embTableName, ddrTableMap);
    unordered_map<std::string, unordered_map<emb_key_t, freq_num_t>> excludeDdrMap;
    unordered_map<emb_key_t, freq_num_t> excludeDdrTableMap;
    excludeDdrTableMap.emplace(15, 1);
    excludeDdrTableMap.emplace(25, 5);
    excludeDdrMap.emplace(embTableName, excludeDdrTableMap);
    cacheManager.Load(ddrMap, excludeDdrMap, 0);
    // 数据检查
    auto& ddrKeyFreqMap = cacheManager.ddrKeyFreqMap;
    auto& excludeDDRKeyCountMap = cacheManager.excludeDDRKeyCountMap;
    ASSERT_EQ(ddrKeyFreqMap[embTableName].minFreq, 1);
    ASSERT_EQ(ddrKeyFreqMap[embTableName].freqTable.size(), 3);
    ASSERT_EQ(ddrKeyFreqMap[embTableName].Get(2), 3);
    ASSERT_EQ(ddrKeyFreqMap[embTableName].Get(12), -1);
    ASSERT_EQ(excludeDDRKeyCountMap[embTableName][25], 5);
}