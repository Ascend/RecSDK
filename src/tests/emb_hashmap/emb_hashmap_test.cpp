/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: emb_hashmap test
 * Author: MindX SDK
 * Date: 2023/9/18
 */

#include <iostream>
#include <gtest/gtest.h>

#include "emb_hashmap/emb_hashmap.h"

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