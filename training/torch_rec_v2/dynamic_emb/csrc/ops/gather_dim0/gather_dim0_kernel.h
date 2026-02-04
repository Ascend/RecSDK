/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

namespace GatherDim0Simt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t CACHE_ALIGN = 64;

// SIMT VF函数 - 小数据模式
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ float* input, __gm__ T* indices, __gm__ float* output, int32_t dataDim, int32_t inLength, int32_t outLength)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

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
        int32_t indices_index = globalIdx / dataDim;
        int32_t indices_dim = globalIdx % dataDim;
        int64_t index = indices[indices_index];
        int64_t data_index = index * dataDim + indices_dim;
        output[globalIdx] = input[data_index];
    }
}

// SIMT VF函数 - 大数据模式
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ float* input, __gm__ T* indices, __gm__ float* output, int32_t dataDim, int32_t inLength, int32_t outLength,
    int32_t totalBlocks, int32_t blockStartIdx, int32_t curBlocksCount)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();

    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();

    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

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
            int32_t indices_index = globalIdx / dataDim;
            int32_t indices_dim = globalIdx % dataDim;
            int64_t index = indices[indices_index];
            int64_t data_index = index * dataDim + indices_dim;
            output[globalIdx] = input[data_index];
        }
    }
}

}  // namespace GatherDim0Simt
