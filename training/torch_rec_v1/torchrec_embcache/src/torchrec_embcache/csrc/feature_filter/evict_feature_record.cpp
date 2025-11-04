/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "evict_feature_record.h"

namespace Embcache {

bool EvictFeatureRecord::CanRemoveFromEmbTable(uint64_t embUpdateCount) const
{
    return embUpdateCount == executeSwapCount_;
}

void EvictFeatureRecord::SetSwapCount(uint64_t swapCount)
{
    executeSwapCount_ = swapCount;
}

void EvictFeatureRecord::ClearEvictInfo()
{
    executeSwapCount_ = 0;
    evictKeys_.clear();
}

const std::vector<int64_t>& EvictFeatureRecord::GetEvictKeys() const
{
    return evictKeys_;
}

std::vector<int64_t>& EvictFeatureRecord::GetEvictKeys()
{
    return evictKeys_;
}

}  // namespace Embcache
