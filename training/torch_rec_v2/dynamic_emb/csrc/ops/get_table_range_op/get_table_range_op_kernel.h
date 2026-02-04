/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GET_TABLE_RANGE_OP_KERNEL_H
#define GET_TABLE_RANGE_OP_KERNEL_H

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace DynamicEmbeddingTableRangeOPSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 1;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t CACHE_ALIGN = 64;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void GetTableRangeKernel(
    __gm__ T* offsets, __gm__ T* featureOffsets, __gm__ T* tableRange,
    int64_t numTable,         // 表的个数
    int64_t featureNumXBatch  // 一个batch的特征总数
)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int64_t globalTid = coreId * threadNumPerCore + tid;

    if (numTable == 0) {
        tableRange[0] = 0;
        return;
    }

    if (globalTid < numTable + 1) {
        T numFeature = featureOffsets[numTable];
        int64_t batch = featureNumXBatch / numFeature;
        T featureOffset = featureOffsets[globalTid];
        int64_t featurePerBatchOffset = static_cast<int64_t>(featureOffset * batch);
        tableRange[globalTid] = offsets[featurePerBatchOffset];
    }
}
}  // namespace DynamicEmbeddingTableRangeOPSimt

#endif  // GET_TABLE_RANGE_OP_KERNEL_H
