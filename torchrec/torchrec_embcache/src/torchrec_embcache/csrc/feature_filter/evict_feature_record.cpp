/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "evict_feature_record.h"

namespace Embcache {

void EvictFeatureRecord::RecordOneStep()
{
    executeSwapCount += 1;
}

bool EvictFeatureRecord::CanRemoveFromEmbTable(uint64_t lookupCount) const
{
    return lookupCount == executeSwapCount;
}

void EvictFeatureRecord::SetSwapCount(uint64_t swapCount)
{
    executeSwapCount = swapCount;
}

void EvictFeatureRecord::ClearEvictInfo()
{
    executeSwapCount = 0;
    evictKeys.clear();
}
std::vector<int64_t>& EvictFeatureRecord::GetEvictKeys()
{
    return evictKeys;
}

}  // namespace Embcache
