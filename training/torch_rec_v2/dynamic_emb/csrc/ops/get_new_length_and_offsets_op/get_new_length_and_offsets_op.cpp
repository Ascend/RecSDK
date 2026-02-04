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

#include "get_new_length_and_offsets_op_kernel.h"

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

extern "C" __global__ __aicore__ void get_new_length_and_offsets_op(GM_ADDR dUniqueOffsets,
                                                                    GM_ADDR dTableOffsetsInFeature, GM_ADDR newOffsets,
                                                                    GM_ADDR newLenghths, int tableNum,
                                                                    int64_t newLengthsSize, int localBatchSize,
                                                                    int32_t isInt32)
{
    DTYPE_DISPATCH(isInt32 > 0, DTYPE_X, {
        using namespace AscendC;
        using namespace DynamicEmbeddingGetNewLengthAndOffsetsOPSimt;

        __gm__ uint64_t* dUniqueOffsetsGm = reinterpret_cast<__gm__ uint64_t*>(dUniqueOffsets);
        __gm__ int64_t* dTableOffsetsInFeatureGm = reinterpret_cast<__gm__ int64_t*>(dTableOffsetsInFeature);
        __gm__ DTYPE_X* newOffsetsGm = reinterpret_cast<__gm__ DTYPE_X*>(newOffsets);
        __gm__ DTYPE_X* newLengthsGm = reinterpret_cast<__gm__ DTYPE_X*>(newLenghths);

        Simt::VF_CALL<GetNewLengthAndOffsetsKernel<DTYPE_X>>(Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, dUniqueOffsetsGm,
                                                             dTableOffsetsInFeatureGm, newOffsetsGm, newLengthsGm,
                                                             tableNum, newLengthsSize, localBatchSize);

        SyncAll();
    });
}