/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
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

#include <c10/util/flat_hash_map.h>

namespace Embcache {

// 被淘汰的key的version给一个特殊标记，用以表示该位置可用；
constexpr int64_t CAN_REUSE_KEY_VERSION = -2;
constexpr int64_t OFFSET_OF_INVALID_KEY = 0;

using ComputeSwapRet = std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
                                  std::vector<int64_t>, std::vector<int64_t>>;

class SwapManager {
public:
    explicit SwapManager(int64_t cacheSize, int64_t memStartOffset = 0);

    template<bool NewKeyReturn0>
    ComputeSwapRet ComputeSwapInfo(const std::vector<int64_t>& keys);

    int64_t GetKey(int64_t off);
    int64_t GetOccupiedNum()
    {
        return occupiedNum;
    };
    void RemoveKeys(const std::vector<int64_t>& keys);
    int64_t GetMemStartOffset() const;

private:
    // device中的start offset；若开启准入，start offset将被初始化为1
    int64_t memStartOffset = 0;

    int64_t cacheSize;
    int64_t occupiedNum = memStartOffset;
    ska::flat_hash_map<int64_t, int64_t> key2off;
    struct KeyVersion {
        int64_t key;
        int64_t version;
    };
    std::vector<KeyVersion> cache;
    int64_t nowVersion = 0;
    int64_t swapIdx = memStartOffset;
};
}  // namespace Embcache

#endif  // EMBEDDING_CACHE_SWAP_MANAGER_H
