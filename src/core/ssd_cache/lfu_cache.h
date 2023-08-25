/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: lfu cache module
 * Author: MindX SDK
 * Date: 2023/8/10
 */
#ifndef MXREC_LFU_CACHE_H
#define MXREC_LFU_CACHE_H

#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include "utils/common.h"

namespace MxRec {
    using namespace std;

    using freq_num_t = int_fast32_t;

    // 记录key和次数信息
    struct LFUCacheNode {
        emb_key_t key;
        freq_num_t freq;

        LFUCacheNode(emb_key_t key, freq_num_t freq) : key(key), freq(freq)
        {}
    };

    class LFUCache {
    public:
        LFUCache();

        freq_num_t Get(emb_key_t key);

        void GetAndDeleteLeastFreqKeyInfo(int64_t num, const vector<emb_key_t>& keys,
                                          vector<emb_key_t>& ddrSwapOutKeys,
                                          vector<freq_num_t>& ddrSwapOutCounts);

        void Put(emb_key_t key);

        void PutKeys(vector<emb_key_t>& keys);

        bool Pop(emb_key_t key);

        void PutWithInit(emb_key_t key, freq_num_t freq);

        std::unordered_map<emb_key_t, freq_num_t> GetFreqTable();
        // 最小频次
        freq_num_t minFreq = 0;
        // 次数, 该次数对应的key列表(key, freq)
        std::unordered_map<freq_num_t, std::list<LFUCacheNode>> freqTable;
        // key, key所属node在freqTable的节点列表中的存储位置地址
        std::unordered_map<emb_key_t, std::list<LFUCacheNode>::iterator> keyTable;
    };
}

#endif // MXREC_LFU_CACHE_H
