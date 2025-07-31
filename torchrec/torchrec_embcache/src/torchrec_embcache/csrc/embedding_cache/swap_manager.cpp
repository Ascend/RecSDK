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

#include <ATen/Parallel.h>

#include "common/constants.h"
#include "utils/logger.h"

using namespace Embcache;

SwapManager::SwapManager(int64_t cacheSize, int64_t memStartOffset) : cacheSize_(cacheSize)
{
    LOG_INFO("Init SwapManager, cacheSize: {}, memStartOffset: {}", cacheSize, memStartOffset);
    if (memStartOffset != 0) {
        this->memStartOffset_ = memStartOffset;
        occupiedNum_ = memStartOffset;
        swapIdx_ = memStartOffset;
    }
    cache_.resize(cacheSize);
    key2off_.reserve(cacheSize);
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
    at::parallel_for(
        0, keys.size(), std::ceil(keys.size() * 1.0 / at::get_num_threads()),
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
                auto it = key2off_.find(key);
                if (it != key2off_.end()) {
                    int64_t off = it->second;
                    batchOffs[i] = off;
                    cache_[off].version = nowVersion_;
                } else {
                    local_missed.push_back(i);
                }
            }
        }
    );

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
        auto it = key2off_.find(key);
        if (it != key2off_.end()) {
            batchOffs[i] = it->second;
            continue;
        }

        int64_t off;
        // cache 未满，直接在cache中新增
        if (occupiedNum_ < cacheSize_) {
            off = occupiedNum_++;
            swapinKeys.push_back(key);
            swapinOffs.push_back(off);
            // 更新状态
            cache_[off] = {key, nowVersion_};
            key2off_[key] = off;
            batchOffs[i] = off;
            continue;
        }

        // cache 已满，需要替换
        // 找到可以被换出的位置，这一步和上一步正在用的key不能换出
        while (swapIdx_ < cacheSize_ && cache_[swapIdx_].version >= nowVersion_ - 1) {
            swapIdx_++;
        }
        // 找到末尾了，重头开始找
        if (swapIdx_ == cacheSize_) {
            swapIdx_ = memStartOffset_;
            while (swapIdx_ < cacheSize_ && cache_[swapIdx_].version >= nowVersion_ - 1) {
                swapIdx_++;
            }
            // 仍没找到，说明没有可以被换出的位置
            if (swapIdx_ == cacheSize_) {
                throw std::runtime_error("cacheSize_ too small");
            }
        }
        bool isEvictedPos = cache_[swapIdx_].version == CAN_REUSE_KEY_VERSION;  // 是否是被淘汰的位置
        off = swapIdx_++;
        swapinKeys.push_back(key);
        swapinOffs.push_back(off);
        int64_t swapoutKey = cache_[off].key;
        if (swapoutKey != INVALID_KEY) {
            key2off_.erase(swapoutKey);
        }
        if (!isEvictedPos) {
            // 非淘汰位置时，才需要将换出的emb/optimizer数据更新到DDR
            swapoutKeys.push_back(swapoutKey);
            swapoutOffs.push_back(off);
        }

        // 更新状态
        cache_[off] = {key, nowVersion_};
        key2off_[key] = off;
        batchOffs[i] = off;
    }
    nowVersion_++;
    return std::make_tuple(swapoutKeys, swapoutOffs, swapinKeys, swapinOffs, batchOffs);
}

void SwapManager::RemoveKeys(const std::vector<int64_t>& keys)
{
    for (auto key : keys) {
        auto iter = key2off_.find(key);
        if (iter == key2off_.end()) {
            continue;
        }

        // 删除 mem cache侧 key offset
        int64_t offset = iter->second;
        // version字段标记为可换出就代表已删除
        cache_[offset].version = CAN_REUSE_KEY_VERSION;
        // 置为无效key，防止该offset在被淘汰到被重用期间进来相同的key, 在重用淘汰位置时，从key2off中误删
        cache_[offset].key = INVALID_KEY;

        // 删除host key offset
        key2off_.erase(key);
    }
}

int64_t SwapManager::GetKey(int64_t off)
{
    TORCH_CHECK(off >= 0 && off < occupiedNum_, "off is out of bounds: ", off, ", occupiedNum: ", occupiedNum_)
    return cache_[off].key;
}

int64_t SwapManager::GetMemStartOffset() const
{
    return memStartOffset_;
}
