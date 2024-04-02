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

#include "emb_hashmap/emb_hashmap.h"
#include "hybrid_mgmt/hybrid_mgmt_block.h"
#include "ssd_cache/cache_manager.h"
#include "utils/common.h"

using namespace std;
using namespace MxRec;
using namespace testing;

const int HBM_VOCAB_SIZE = 10;
const int DDR_VOCAB_SIZE = 100;
const int SSD_VOCAB_SIZE = 100;
const int INT_2 = 2;
const int INT_4 = 4;
const int INT_21 = 21;
const int INT_42 = 42;
const int NEGATIVE_INT_1 = -1;

// 刷新换入换出频次和打印信息
void RefreshSwapFreqInfoAndPrint(EmbHashMap& hostHashMaps, string embTableName, int opTimes)
{
    auto& embHashMap = hostHashMaps.embHashMaps[embTableName];
    hostHashMaps.RefreshFreqInfoWithSwap(embTableName, embHashMap);
    vector<emb_key_t> hbm2DdrKeyList;
    vector<emb_key_t> ddr2HbmKeyList;
    for (auto it : embHashMap.oldSwap) {
        hbm2DdrKeyList.emplace_back(it.first);
        ddr2HbmKeyList.emplace_back(it.second);
    }
    LOG_INFO("embHashMap hbm2DdrKeyList: {}", VectorToString(hbm2DdrKeyList));
    LOG_INFO("embHashMap ddr2HbmKeyList: {}", VectorToString(ddr2HbmKeyList));
    embHashMap.oldSwap.clear();
    LOG_INFO("RefreshSwapFreqInfoAndPrint end, opTimes:{}", opTimes);
}

vector<EmbInfo> GetEmbInfoList()
{
    EmbInfo embInfo;
    embInfo.name = "table1";
    embInfo.devVocabSize = HBM_VOCAB_SIZE;
    embInfo.hostVocabSize = DDR_VOCAB_SIZE;
    embInfo.ssdVocabSize = SSD_VOCAB_SIZE;
    embInfo.ssdDataPath = {"ssd_data"};
    vector<EmbInfo> embInfos;
    embInfos.emplace_back(embInfo);
    return embInfos;
}

// 测试HBM与DDR换入换出时CacheManager模块频次刷新
TEST(EmbHashMap, TestFindOffset)
{
    LOG_INFO("start TestFindOffset");
    string embTableName = "table1";
    EmbHashMap hostHashMaps;
    RankInfo rankInfo;
    rankInfo.isDDR = true;
    auto embInfo = GetEmbInfoList();
    hostHashMaps.Init(rankInfo, embInfo, false);
    CacheManager cacheManager;
    cacheManager.Init(nullptr, embInfo);
    bool isSSDEnabled = true;
    hostHashMaps.isSSDEnabled = isSSDEnabled;
    hostHashMaps.cacheManager = &cacheManager;
    int channelId = 0;
    size_t currentBatchId = 0;
    size_t keepBatchId = 0;
    int opTimes = 0;

    vector<emb_key_t> keys = {1, 2, 3, 4, 5};
    hostHashMaps.FindOffset(embTableName, keys, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);

    vector<emb_key_t> keys2 = {6, 7, 8, 9, 10};
    hostHashMaps.FindOffset(embTableName, keys2, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);

    auto& excludeKeyMap = cacheManager.excludeDDRKeyCountMap[embTableName];
    auto& ddrKeyMap = cacheManager.ddrKeyFreqMap[embTableName];

    auto logLevelTemp = Logger::GetLevel();
    Logger::SetLevel(Logger::TRACE);
    vector<emb_key_t> keys4 = {21, 21, 21, 21}; // 新key重复值, 且需要换入换出
    hostHashMaps.FindOffset(embTableName, keys4, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);
    ASSERT_EQ(excludeKeyMap[INT_21], INT_4);
    ASSERT_EQ(ddrKeyMap.Get(1), 1);

    keys4 = {41, 42, 43, 44, 45, 46, 47, 48, 49, 50}; // 整个hbm大小key换入换出
    hostHashMaps.FindOffset(embTableName, keys4, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);
    ASSERT_EQ(ddrKeyMap.Get(INT_21), INT_4);

    keys4 = {51, 52, 53, 1, 2, 21, 41, 42, 43, 44}; // 3个新key， 3个在ddr, 4个在hbm
    hostHashMaps.FindOffset(embTableName, keys4, currentBatchId, keepBatchId, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes);
    ASSERT_EQ(excludeKeyMap[1], INT_2);
    ASSERT_EQ(excludeKeyMap[INT_42], INT_2);
    ASSERT_EQ(ddrKeyMap.Get(INT_21), NEGATIVE_INT_1);
    ASSERT_EQ(ddrKeyMap.Get(1), NEGATIVE_INT_1);
    Logger::SetLevel(logLevelTemp); // 恢复日志级别
    LOG_INFO("test TestFindOffset end.");
}

TEST(EmbHashMap, TESTGetHashMaps)
{
    string embTableName = "table1";
    EmbHashMap hostHashMaps;
    RankInfo rankInfo;
    rankInfo.isDDR = true;
    auto embInfo = GetEmbInfoList();
    hostHashMaps.Init(rankInfo, embInfo, false);
    CacheManager cacheManager;
    cacheManager.Init(nullptr, embInfo);
    hostHashMaps.isSSDEnabled = true;
    hostHashMaps.cacheManager = &cacheManager;
    int channelId = 0;
    size_t currentBatchId = 0;
    size_t keepBatchId = 0;
    int opTimes = 0;

    vector<emb_key_t> keys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    hostHashMaps.FindOffset(embTableName, keys, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);
    auto testEmbHashMap = hostHashMaps.GetHashMaps().at(embTableName);
    hostHashMaps.embHashMaps.at(embTableName).maxOffsetOld = testEmbHashMap.maxOffset;
    // 增加10个key, offset长度变为10
    ASSERT_EQ(testEmbHashMap.maxOffset, 10);

    keys = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    hostHashMaps.FindOffset(embTableName, keys, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);
    testEmbHashMap = hostHashMaps.GetHashMaps().at(embTableName);
    // 再增加10个key，offset变为20
    ASSERT_EQ(testEmbHashMap.maxOffset, 20);

    HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
    hybridMgmtBlock->lastRunChannelId = channelId;
    hybridMgmtBlock->hybridBatchId[0] = 1;
    testEmbHashMap = hostHashMaps.GetHashMaps().at(embTableName);
    // 回退一步，offset变回10
    ASSERT_EQ(testEmbHashMap.maxOffset, 10);

    hybridMgmtBlock->hybridBatchId[0] = 2;
    // 回退2步，抛出异常
    ASSERT_THROW(hostHashMaps.GetHashMaps(), HybridMgmtBlockingException);
    hybridMgmtBlock->hybridBatchId[0] = 0;

    keys = {10, 11};
    hostHashMaps.EvictDeleteEmb(embTableName, keys);
    testEmbHashMap = hostHashMaps.GetHashMaps().at(embTableName);
    // 淘汰1个hbm key和1个ddr key，表中无法查找到该key
    ASSERT_EQ(testEmbHashMap.hostHashMap.find(10), testEmbHashMap.hostHashMap.end());
    ASSERT_EQ(testEmbHashMap.hostHashMap.find(11), testEmbHashMap.hostHashMap.end());
    ASSERT_EQ(cacheManager.excludeDDRKeyCountMap[embTableName][11], 0);
    ASSERT_EQ(cacheManager.ddrKeyFreqMap[embTableName].Get(10), -1);

    keys = {1, 2};
    hostHashMaps.FindOffset(embTableName, keys, currentBatchId++, keepBatchId++, channelId);
    RefreshSwapFreqInfoAndPrint(hostHashMaps, embTableName, opTimes++);
    testEmbHashMap = hostHashMaps.GetHashMaps().at(embTableName);
    // 从ddr中换回2个key到hbm，交换变量长度为2
    ASSERT_EQ(testEmbHashMap.ddr2HbmKeys.size(), 2);
    hostHashMaps.ClearLookupAndSwapOffset(hostHashMaps.embHashMaps.at(embTableName));
    testEmbHashMap = hostHashMaps.GetHashMaps().at(embTableName);
    // 清理后，交换变量长度为0
    ASSERT_EQ(testEmbHashMap.ddr2HbmKeys.size(), 0);
}