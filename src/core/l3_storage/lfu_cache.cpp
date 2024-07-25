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

#include "lfu_cache.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace MxRec;

/// 仅获取当前key的频次，不增加频次；key不存在时返回-1
/// \param key key
/// \return key的频次
freq_num_t LFUCache::Get(emb_cache_key_t key)
{
    auto it = keyTable.find(key);
    if (it == keyTable.end()) { return -1; }
    return it->second->freq;
}

/// 返回num个不在keys中的最低频次的key和对应次数；并清理相应的key数据
/// \param num 要返回的最低频次key info的个数
/// \param keys 要返回的最低频次key不能在该列表内
/// \param ddrSwapOutKeys 记录最低频次key
/// \param ddrSwapOutCounts 记录最低频次key对应次数
void LFUCache::GetAndDeleteLeastFreqKeyInfo(uint64_t num, const vector<emb_cache_key_t>& keys,
                                            vector<emb_cache_key_t>& ddrSwapOutKeys,
                                            vector<freq_num_t>& ddrSwapOutCounts)
{
    freq_num_t tempMinFreq = minFreq;
    unordered_set<emb_cache_key_t> retainedKeySet(keys.begin(), keys.end());
    uint64_t counter = 0;
    const size_t freqSize = freqTable.size();
    LOG_DEBUG("table:{}, num:{}, freqTable.size:{}, keys.size:{}, ddrSwapOutKeys.size:{}, ddrSwapOutCounts.size:{}",
              name, num, freqTable.size(), keys.size(), ddrSwapOutKeys.size(), ddrSwapOutCounts.size());
    // 遍历freqTable<次数，keyList>时，次数可能不连续，要实际使用了1个keyList后才自增，手动增加计数器
    for (size_t i = 0; i < freqSize;) {
        auto nodesIter = freqTable.find(tempMinFreq);
        if (nodesIter == freqTable.end()) {
            tempMinFreq++;
            continue;
        }
        auto nodeIt = freqTable[tempMinFreq].begin();
        while (nodeIt != freqTable[tempMinFreq].end() && !freqTable[tempMinFreq].empty() && counter < num) {
            emb_cache_key_t currentKey = nodeIt->key;
            if (retainedKeySet.find(currentKey) != retainedKeySet.end()) {
                // 当前key在指定的集合中，不满足
                nodeIt++;
                continue;
            }
            ddrSwapOutKeys.emplace_back(currentKey);
            ddrSwapOutCounts.emplace_back(nodeIt->freq);
            keyTable.erase(currentKey);
            nodeIt = freqTable[tempMinFreq].erase(nodeIt);
            counter++;
        }
        if (freqTable[tempMinFreq].empty()) {
            freqTable.erase(tempMinFreq);
            // 删除频次列表时，若最小频次和临时最小频次相等，则最小频次+1
            minFreq = tempMinFreq == minFreq ? minFreq + 1 : minFreq;
        }
        if (counter == num || freqTable.empty()) {
            break;
        }
        tempMinFreq++;
        i++;
    }
}

/// 放入key，新增/更新(次数+1)次数
/// \param key key
void LFUCache::Put(emb_cache_key_t key)
{
    auto it = keyTable.find(key);
    if (it == keyTable.end()) {
        freqTable[1].emplace_front(key, 1);
        keyTable[key] = freqTable[1].begin();
        minFreq = 1;
        return;
    }
    auto& node = it->second;
    freq_num_t freq = node->freq;
    freqTable[freq].erase(node);
    if (freqTable[freq].empty()) {
        freqTable.erase(freq);
        if (minFreq == freq) {
            minFreq += 1;
        }
    }
    freqTable[freq + 1].emplace_front(key, freq + 1);
    keyTable[key] = freqTable[freq + 1].begin();
}

/// 直接放入指定次数；用于初始化场景
/// \param key key
/// \param freq 频次
void LFUCache::PutWithInit(emb_cache_key_t key, freq_num_t freq)
{
    if (keyTable.find(key) != keyTable.end()) {
        // 一般初始化时，key应该不存在已经被插入的情况；此处替换就的key频次信息
        LOG_WARN("key has exist when init process, key:{}", key);
        Pop(key);
    }
    freqTable[freq].emplace_front(key, freq);
    keyTable[key] = freqTable[freq].begin();
    if (minFreq == 0) {
        minFreq = freq;
    } else {
        minFreq = freq < minFreq ? freq : minFreq;
    }
}

/// 删除指定key
bool LFUCache::Pop(emb_cache_key_t key)
{
    auto it = keyTable.find(key);
    if (it == keyTable.end()) {
        return false;
    }
    auto& node = it->second;
    freq_num_t oldFreq = node->freq;
    freqTable[oldFreq].erase(node);
    if (freqTable[oldFreq].empty()) {
        freqTable.erase(oldFreq);
        if (minFreq == oldFreq) { minFreq += 1; }
    }
    keyTable.erase(it);
    return true;
}

/// 获取所有的key和次数信息
/// \return 频次数据map<key, freq>
std::unordered_map<emb_cache_key_t, freq_num_t> LFUCache::GetFreqTable()
{
    unordered_map<emb_cache_key_t, freq_num_t> freqMap(keyTable.size());
    for (const auto& it :keyTable) {
        freqMap[it.first] = it.second->freq;
    }
    return freqMap;
}

LFUCache::LFUCache(const string& cacheName)
{
    name = cacheName;
    minFreq = 0;
    keyTable.clear();
    freqTable.clear();
}

LFUCache::LFUCache()
{
    minFreq = 0;
    keyTable.clear();
    freqTable.clear();
}
