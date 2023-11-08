/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: lfu cache module
 * Author: MindX SDK
 * Date: 2023/8/10
 */

#include <iostream>
#include <gtest/gtest.h>

#include "ssd_cache/lfu_cache.h"

using namespace std;
using namespace MxRec;
using namespace testing;

/*
 * 要放入的key, 频次对应key列表中元素顺序和放入顺序相反; 如下列表key逐个放入的结果：
 * 频次-对应key列表
 * 1 - 9,8
 * 2 - 6,4
 * 3 - 3,2,1
 */
vector<emb_key_t> INPUT_KEYS = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 6, 6, 8, 9};

inline void CompareHandleRet(vector<emb_key_t>& leastFreqKeys, vector<freq_num_t>& leastFreq,
                             vector<emb_key_t>& expectKeys,
                             vector<freq_num_t>& expectFreq)
{
    ASSERT_EQ(leastFreqKeys.size(), expectKeys.size());
    ASSERT_EQ(leastFreq.size(), expectFreq.size());
    for (size_t i = 0; i < leastFreqKeys.size(); i++) {
        ASSERT_EQ(leastFreqKeys[i], expectKeys[i]);
        ASSERT_EQ(leastFreq[i], expectFreq[i]);
    }
}

void PutKeys(LFUCache& lfu, vector<emb_key_t>& embKeys)
{
    for (auto& key : embKeys) {
        lfu.Put(key);
    }
}

TEST(LFUCache, TestGetFreqTable)
{
    LFUCache cache;
    PutKeys(cache, INPUT_KEYS);
    auto ret = cache.GetFreqTable();
    ASSERT_EQ(ret[9], 1);
    ASSERT_EQ(ret[6], 2);
    ASSERT_EQ(ret[3], 3);
}

TEST(LFUCache, PopTest)
{
    LFUCache cache;
    PutKeys(cache, INPUT_KEYS);
    cache.Pop(8);
    cache.Pop(9);
    ASSERT_EQ(cache.minFreq, 2);
    ASSERT_EQ(cache.Get(8), -1);
    ASSERT_EQ(cache.Get(9), -1);
}

TEST(LFUCache, PutInitTest)
{
    LFUCache cache;
    cache.PutWithInit(1, 3);
    cache.PutWithInit(2, 3);
    cache.PutWithInit(3, 3);
    cache.PutWithInit(4, 2);
    cache.PutWithInit(6, 2);
    cache.PutWithInit(8, 1);
    cache.PutWithInit(9, 1);
    vector<emb_key_t> retainedKeys = {4, 6};
    vector<emb_key_t> leastFreqKeys;
    vector<freq_num_t> leastFreq;
    cache.GetAndDeleteLeastFreqKeyInfo(2, retainedKeys, leastFreqKeys, leastFreq);
    vector<emb_key_t> expectKeys = {9, 8};
    vector<freq_num_t> expectFreq = {1, 1};
    CompareHandleRet(leastFreqKeys, leastFreq, expectKeys, expectFreq);
    ASSERT_EQ(cache.minFreq, 2);
}

TEST(LFUCache, LFUDeleteTotalFreqListTest)
{
    LFUCache cache;
    PutKeys(cache, INPUT_KEYS);
    vector<emb_key_t> retainedKeys = {4, 6, 8, 9};
    vector<emb_key_t> leastFreqKeys;
    vector<freq_num_t> leastFreq;
    cache.GetAndDeleteLeastFreqKeyInfo(2, retainedKeys, leastFreqKeys, leastFreq);
    vector<emb_key_t> expectKeys = {3, 2};
    vector<freq_num_t> expectFreq = {3, 3};
    CompareHandleRet(leastFreqKeys, leastFreq, expectKeys, expectFreq);
}

TEST(LFUCache, BaseCacheTest)
{
    LFUCache cache;
    PutKeys(cache, INPUT_KEYS);
    vector<emb_key_t> retainedKeys = {8, 4, 6, 2};
    vector<emb_key_t> leastFreqKeys;
    vector<freq_num_t> leastFreq;
    cache.GetAndDeleteLeastFreqKeyInfo(2, retainedKeys, leastFreqKeys, leastFreq);
    vector<emb_key_t> expectKeys = {9, 3};
    vector<freq_num_t> expectFreq = {1, 3};
    CompareHandleRet(leastFreqKeys, leastFreq, expectKeys, expectFreq);
    ASSERT_EQ(cache.minFreq, 1);
    ASSERT_EQ(cache.Get(9), -1);
    cache.Put(9);
    ASSERT_EQ(cache.Get(9), 1);
    cache.Put(9);
    ASSERT_EQ(cache.minFreq, 2);
}
