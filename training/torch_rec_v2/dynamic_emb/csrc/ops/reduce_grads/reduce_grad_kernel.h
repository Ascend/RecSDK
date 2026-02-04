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

#ifndef REDUCE_GRAD_KERNEL_H
#define REDUCE_GRAD_KERNEL_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernel_operator.h"

namespace DynamicEmbeddingReduceGradOPSimt {

constexpr int MAX_THREADS_PER_BLOCK = 1024;

__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void ReduceGradCompute(
    __gm__ const float* grad, __gm__ const int64_t* inverse,
    int len, int dim, int blkStart, int blkCnt, __gm__ float* output)
{
    int threadIdx = AscendC::Simt::GetThreadIdx<0>();

#pragma unroll
    for (int i = 0; i < blkCnt; ++i) {
        int index = (blkStart + i) * MAX_THREADS_PER_BLOCK + threadIdx;
        int keyIdx = index / dim;
        if (keyIdx >= len) {
            continue;
        }

        int dimIdx = index % dim;
        float src = *(grad + dim * keyIdx + dimIdx);
        __gm__ float* dst = output + inverse[keyIdx] * dim + dimIdx;
        AscendC::Simt::AtomicAdd<float>(dst, src);
    }
}

}

#endif