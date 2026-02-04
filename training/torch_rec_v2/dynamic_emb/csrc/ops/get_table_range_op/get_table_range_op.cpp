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

#include <type_traits>

#include "get_table_range_op_kernel.h"

#define DTYPE_DISPATCH(isInt32, DTYPE, ...) \
    do {                                    \
        if (isInt32) {                      \
            using DTYPE = int32_t;          \
            __VA_ARGS__;                    \
        } else {                            \
            using DTYPE = int64_t;          \
            __VA_ARGS__;                    \
        }                                   \
    } while (0)

extern "C" __global__ __aicore__ void get_table_range_op(GM_ADDR offsets, GM_ADDR featureOffsets, GM_ADDR tableRange,
                                                         int64_t featureNumXBatch, int64_t tableNum, int32_t isInt32)
{
    DTYPE_DISPATCH(isInt32 > 0, DTYPE_X, {
        using namespace AscendC;
        __gm__ DTYPE_X* offsetsGm = reinterpret_cast<__gm__ DTYPE_X*>(offsets);
        __gm__ DTYPE_X* featureOffsetsGm = reinterpret_cast<__gm__ DTYPE_X*>(featureOffsets);
        __gm__ DTYPE_X* tableRangeGm = reinterpret_cast<__gm__ DTYPE_X*>(tableRange);
        int32_t coreNum = AscendC::GetBlockNum();
        int32_t totalThreads = coreNum * DynamicEmbeddingTableRangeOPSimt::MAX_THREADS_PER_BLOCK;

        Simt::VF_CALL<DynamicEmbeddingTableRangeOPSimt::GetTableRangeKernel<DTYPE_X>>(
            Simt::Dim3{DynamicEmbeddingTableRangeOPSimt::MAX_THREADS_PER_BLOCK, 1, 1}, offsetsGm, featureOffsetsGm,
            tableRangeGm, tableNum, featureNumXBatch);
        SyncAll();
    });
}