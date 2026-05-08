/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef BLOCK_BUCKETIZE_SPARSE_FEATURES_COMMON_H
#define BLOCK_BUCKETIZE_SPARSE_FEATURES_COMMON_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace BlockBucketizeSparseFeaturesCommon {

struct WorkRange {
    int32_t begin;
    int32_t end;
};

__aicore__ inline void ComputeWorkRange(int32_t totalSize, int32_t coreId, int32_t coreNum, WorkRange& range)
{
    if (totalSize <= 0 || coreNum <= 0 || coreId < 0 || coreId >= coreNum) {
        range = {0, 0};
        return;
    }
    const int32_t base = totalSize / coreNum;
    const int32_t remainder = totalSize % coreNum;
    const int32_t begin = coreId * base + ((coreId < remainder) ? coreId : remainder);
    range = {begin, begin + base + ((coreId < remainder) ? 1 : 0)};
}

} // namespace BlockBucketizeSparseFeaturesCommon

#endif // BLOCK_BUCKETIZE_SPARSE_FEATURES_COMMON_H
