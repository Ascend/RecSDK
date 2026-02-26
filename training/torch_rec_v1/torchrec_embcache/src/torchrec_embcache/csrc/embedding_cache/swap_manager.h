/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_SWAP_MANAGER_H
#define EMBEDDING_CACHE_SWAP_MANAGER_H

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>

#include "common/constants.h"

#include <c10/util/flat_hash_map.h>

namespace Embcache {

// 被淘汰的key的version给一个特殊标记，用以表示该位置可用；
constexpr int64_t OFFSET_OF_INVALID_KEY = 0;

using ComputeSwapRet = std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
                                  std::vector<int64_t>, std::vector<int64_t>>;

class SwapManager {
public:
    explicit SwapManager(int64_t cacheSize, int64_t memStartOffset = 0);

    ComputeSwapRet ComputeSwapInfo(const std::vector<int64_t>& keys);

    int64_t GetKey(int64_t off, int64_t embeddingUpdateVersion);
    int64_t GetOccupiedNum()
    {
        return occupiedNum_;
    };
    void RemoveKeys(const std::vector<int64_t>& keys, bool isEvict = false);
    void UpdateCache(int64_t key, int64_t off, int64_t version);
    int64_t GetMemStartOffset() const;
    struct KeyVersion {
        int64_t key = INVALID_KEY;
        int64_t version = INIT_VERSION;
    };

private:
    // device中的start offset；若开启准入，start offset将被初始化为1
    int64_t memStartOffset_ = 0;

    int64_t cacheSize_;
    int64_t occupiedNum_ = memStartOffset_;
    static constexpr int64_t maxVersionDiff_ = 3;  // cache中同一位置最多记录最近3轮的key版本信息；与pipeline的step和swap的最大值相关
    static constexpr int64_t INIT_VERSION = -1;
    ska::flat_hash_map<int64_t, int64_t> key2off_;
    struct CacheSlot {
        KeyVersion history[maxVersionDiff_];
        int64_t startIndex = 0;  // 循环队列的起始位置
    };
    std::vector<CacheSlot> cache_;
    int64_t nowVersion_ = 0;
    int64_t swapIdx_ = memStartOffset_;
    KeyVersion& FindLastKeyVersion(int64_t off);
};
}  // namespace Embcache

#endif  // EMBEDDING_CACHE_SWAP_MANAGER_H
