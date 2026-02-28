/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#pragma once

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

namespace PoolingEmbeddingsSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;

// SIMT VF函数 - 小数据模式
template <typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ T1* src, __gm__ T2* dst, __gm__ T3* offset, __gm__ T3* inverse, int32_t combiner,
    int32_t total_dims, int32_t accum_dims, int32_t ev_size, int32_t num_vec, int32_t batch_size)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t outLength = num_vec * ev_size;

    int32_t blockBase = blockIdx * blockElementCapacity;

    if (blockBase >= outLength) {
        return;
    }

    int32_t elementsRemaining = outLength - blockBase;
    int32_t elementsThisBlock = (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
    if (elementsThisBlock <= 0) {
        return;
    }

    int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;

    if (threadElementBase >= outLength) {
        return;
    }

    int32_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
    int32_t elementsForThread =
        (elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD : elementsThisBlockRemaining;

    // 2. 实际计算
#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
        if (i >= elementsForThread) {
            return;
        }
        int32_t globalIdx = threadElementBase + i;
        if (globalIdx >= outLength) {
            return;
        }
        int32_t indices_index = globalIdx / ev_size;
        int32_t indices_dim = globalIdx % ev_size;

        int32_t start = offset[indices_index] - offset[0];
        int32_t vectorNum = offset[indices_index + 1] - offset[indices_index];

        float accum{0.0f};
        for (int32_t j = 0; j < vectorNum; j++) {
            int32_t src_index = inverse[j + start];
            int32_t data_index = src_index * ev_size + indices_dim;
            accum += src[data_index];
        }

        if (combiner > 0) {
            accum /= vectorNum;
        }

        int32_t dstRowIndex = indices_index % batch_size;
        int32_t dstColIndex = indices_index / batch_size;
        dst[dstRowIndex * total_dims + accum_dims + dstColIndex * ev_size + indices_dim] = accum;
    }
}

// SIMT VF函数 - 大数据模式
template <typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ T1* src, __gm__ T2* dst, __gm__ T3* offset, __gm__ T3* inverse, int32_t combiner,
    int32_t total_dims, int32_t accum_dims, int32_t ev_size, int32_t num_vec, int32_t batch_size,
    int32_t totalBlocks, int32_t blockStartIdx, int32_t curBlocksCount)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();

    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();

    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

    int32_t outLength = num_vec * ev_size;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        // 1.位置计算
        int32_t globalBlockIdx = blockStartIdx + iter;
        if (globalBlockIdx >= totalBlocks) {
            break;
        }

        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= outLength) {
            break;
        }

        int32_t elementsRemaining = outLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;

        if (threadElementBase >= outLength) {
            break;
        }

        int32_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = (elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD
                                                                                           : elementsThisBlockRemaining;

        // 2. 实际计算
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i >= elementsForThread) {
                break;
            }
            int32_t globalIdx = threadElementBase + i;
            if (globalIdx >= outLength) {
                break;
            }

            int32_t indices_index = globalIdx / ev_size;
            int32_t indices_dim = globalIdx % ev_size;

            int32_t start = offset[indices_index] - offset[0];
            int32_t vectorNum = offset[indices_index + 1] - offset[indices_index];

            float accum{0.0f};
            for (int32_t j = 0; j < vectorNum; j++) {
                int32_t src_index = inverse[j + start];
                int32_t data_index = src_index * ev_size + indices_dim;
                accum += src[data_index];
            }

            if (combiner > 0) {
                accum /= vectorNum;
            }

            int32_t dstRowIndex = indices_index % batch_size;
            int32_t dstColIndex = indices_index / batch_size;
            dst[dstRowIndex * total_dims + accum_dims + dstColIndex * ev_size + indices_dim] = accum;
        }
    }
}
}  // namespace PoolingEmbeddingsSimt