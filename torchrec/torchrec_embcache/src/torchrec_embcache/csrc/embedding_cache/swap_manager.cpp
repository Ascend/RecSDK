/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "swap_manager.h"

#include <iostream>
#include <memory>

#include <glog/logging.h>
#include <ATen/Parallel.h>

using namespace Embcache;

SwapManager::SwapManager(int64_t cacheSize, int64_t memStartOffset) : cacheSize(cacheSize)
{
    LOG(INFO) << "Init SwapManager, cacheSize:" << cacheSize << ", memStartOffset:" << memStartOffset;
    if (memStartOffset != 0) {
        this->memStartOffset = memStartOffset;
        occupiedNum = memStartOffset;
        swapIdx = memStartOffset;
    }
    cache.resize(cacheSize);
    key2off.reserve(cacheSize);
}

ComputeSwapRet SwapManager::ComputeSwapInfo(const std::vector<int64_t>& keys)
{
    std::vector<int64_t> swapoutKeys;
    std::vector<int64_t> swapinKeys;
    std::vector<int64_t> swapoutOffs;
    std::vector<int64_t> swapinOffs;
    std::vector<int64_t> batchOffs(keys.size());

    // 本 batch 的 key 不能被换出
    std::vector<int64_t> missedIdx;

    // 每个线程一个本地 missedIdx 缓冲
    std::vector<std::vector<int64_t>> missed_chunks(at::get_num_threads());
    at::parallel_for(0, keys.size(), std::ceil(keys.size() * 1.0 / at::get_num_threads()),
                     [&](int64_t begin, int64_t end) {
                         const int tid = at::get_thread_num();
                         auto& local_missed = missed_chunks[tid];
                         local_missed.reserve(end - begin);

                         for (int64_t i = begin; i < end; ++i) {
                             int64_t key = keys[i];
                             if (key == INVALID_KEY) {
                                 batchOffs[i] = OFFSET_OF_INVALID_KEY;
                                 continue;
                             }
                             auto it = key2off.find(key);
                             if (it != key2off.end()) {
                                 int64_t off = it->second;
                                 batchOffs[i] = off;
                                 cache[off].version = nowVersion;
                             } else {
                                 local_missed.push_back(i);
                             }
                         }
                     });

    // 合并各线程 missed 结果
    missedIdx.clear();
    size_t total = 0;
    for (auto& v : missed_chunks) {
        total += v.size();
    }
    missedIdx.reserve(total);

    for (auto& v : missed_chunks) {
        missedIdx.insert(missedIdx.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }

    for (int64_t i : missedIdx) {
        int64_t key = keys[i];
        auto it = key2off.find(key);
        if (it != key2off.end()) {
            batchOffs[i] = it->second;
            continue;
        }

        int64_t off;
        // cache 未满，直接在cache中新增
        if (occupiedNum < cacheSize) {
            off = occupiedNum++;
            swapinKeys.push_back(key);
            swapinOffs.push_back(off);
            // 更新状态
            cache[off] = {key, nowVersion};
            key2off[key] = off;
            batchOffs[i] = off;
            continue;
        }

        // cache 已满，需要替换
        // 找到可以被换出的位置，这一步和上一步正在用的key不能换出
        while (swapIdx < cacheSize && cache[swapIdx].version >= nowVersion - 1) {
            swapIdx++;
        }
        // 找到末尾了，重头开始找
        if (swapIdx == cacheSize) {
            swapIdx = memStartOffset;
            while (swapIdx < cacheSize && cache[swapIdx].version >= nowVersion - 1) {
                swapIdx++;
            }
            // 仍没找到，说明没有可以被换出的位置
            if (swapIdx == cacheSize) {
                throw std::runtime_error("cacheSize too small");
            }
        }
        bool isEvictedPos = cache[swapIdx].version == CAN_REUSE_KEY_VERSION;  // 是否是被淘汰的位置
        off = swapIdx++;
        swapinKeys.push_back(key);
        swapinOffs.push_back(off);
        int64_t swapoutKey = cache[off].key;
        key2off.erase(swapoutKey);
        if (!isEvictedPos) {
            // 非淘汰位置时，才需要将换出的emb/optimizer数据更新到DDR
            swapoutKeys.push_back(swapoutKey);
            swapoutOffs.push_back(off);
        }

        // 更新状态
        cache[off] = {key, nowVersion};
        key2off[key] = off;
        batchOffs[i] = off;
    }
    nowVersion++;
    return std::make_tuple(swapoutKeys, swapoutOffs, swapinKeys, swapinOffs, batchOffs);
}

void SwapManager::RemoveKeys(const std::vector<int64_t>& keys, std::vector<int64_t>& evictFeatures)
{
    for (auto key : keys) {
        auto iter = key2off.find(key);
        if (iter == key2off.end()) {
            continue;
        }

        // 删除 mem cache侧 key offset
        int64_t offset = iter->second;
        if (offset < cacheSize) {
            // version字段标记为可换出就代表已删除
            cache[offset].version = CAN_REUSE_KEY_VERSION;
        }
        // 删除host key offset
        key2off.erase(key);

        // 记录被删除的key, 用于延迟删除embTable中embedding数据
        evictFeatures.emplace_back(key);
    }
}

int64_t SwapManager::GetKey(int64_t off)
{
    TORCH_CHECK(off >= 0 && off < occupiedNum, "off is out of bounds: ", off, ", occupiedNum: ", occupiedNum)
    return cache[off].key;
}

int64_t SwapManager::GetMemStartOffset() const
{
    return memStartOffset;
}
