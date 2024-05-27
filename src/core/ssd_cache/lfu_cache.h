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
        emb_cache_key_t key;
        freq_num_t freq;

        LFUCacheNode(emb_cache_key_t key, freq_num_t freq) : key(key), freq(freq)
        {}
    };

    class LFUCache {
    public:
        LFUCache();

        explicit LFUCache(const string& cacheName);

        freq_num_t Get(emb_cache_key_t key);

        void GetAndDeleteLeastFreqKeyInfo(uint64_t num, const vector<emb_cache_key_t>& keys,
                                          vector<emb_cache_key_t>& ddrSwapOutKeys,
                                          vector<freq_num_t>& ddrSwapOutCounts);

        void Put(emb_cache_key_t key);

        bool Pop(emb_cache_key_t key);

        void PutWithInit(emb_cache_key_t key, freq_num_t freq);

        std::unordered_map<emb_cache_key_t, freq_num_t> GetFreqTable();
        // 最小频次
        freq_num_t minFreq = 0;
        // 次数, 该次数对应的key列表(key, freq)
        std::unordered_map<freq_num_t, std::list<LFUCacheNode>> freqTable;
        // key, key所属node在freqTable的节点列表中的存储位置地址
        std::unordered_map<emb_cache_key_t, std::list<LFUCacheNode>::iterator> keyTable;
    private:
        string name;
    };
}

#endif // MXREC_LFU_CACHE_H
