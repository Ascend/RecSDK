/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EVICT_FEATURE_RECORD_H
#define EVICT_FEATURE_RECORD_H

#include <cstdint>
#include <ctime>
#include <unordered_map>
#include <vector>

namespace Embcache {
class EvictFeatureRecord {
public:
    EvictFeatureRecord() = default;
    bool CanRemoveFromEmbTable(uint64_t embUpdateCount) const;
    void ClearEvictInfo();
    void SetSwapCount(uint64_t swapCount);
    std::vector<int64_t>& GetEvictKeys();

private:
    // 触发淘汰时ComputeSwapInfo的执行步数，用于判断调用embTable删除接口的时机
    uint64_t executeSwapCount = 0;

    // ComputeSwapInfo于EmbeddingUpdate之间存在执行时间差异，记录embTable待删除的keys
    std::vector<int64_t> evictKeys;
};

}  // namespace Embcache

#endif  // EVICT_FEATURE_RECORD_H
