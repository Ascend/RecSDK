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

namespace LoadFromPointerSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;

// SIMT VF函数 - 小数据模式
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SmallDataLoadFromPointerCompute(
    __gm__ float* __gm__* input, __gm__ float* output, int32_t dataDim, int64_t inLength, int64_t outLength)
{
    // 1. 线程信息计算
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

    int64_t blockBase = blockIdx * blockElementCapacity;

    if (blockBase >= outLength) {
        return;
    }

    int64_t elementsRemaining = outLength - blockBase;
    int64_t elementsThisBlock = (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
    if (elementsThisBlock <= 0) {
        return;
    }

    int64_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;

    if (threadElementBase >= outLength) {
        return;
    }

    int64_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
    int64_t elementsForThread =
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

        if (input[indices_index] == 0) {
            continue;
        }

        __gm__ float* first_pointer = reinterpret_cast<__gm__ float*>(input[indices_index]);
        output[globalIdx] = first_pointer[indices_dim];
    }
}

// SIMT VF函数 - 大数据模式
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void LargeDataLoadFromPointerCompute(
    __gm__ float* __gm__* input, __gm__ float* output, int32_t dataDim, int64_t inLength, int64_t outLength,
    int64_t totalBlocks, int64_t blockStartIdx, int64_t curBlocksCount)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        // 1.位置计算
        int64_t globalBlockIdx = blockStartIdx + iter;
        if (globalBlockIdx >= totalBlocks) {
            break;
        }

        int64_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= outLength) {
            break;
        }

        int64_t elementsRemaining = outLength - blockBase;
        int64_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int64_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;

        if (threadElementBase >= outLength) {
            break;
        }

        int64_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
        int64_t elementsForThread = (elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD
                                                                                           : elementsThisBlockRemaining;

        // 2. 实际计算
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i >= elementsForThread) {
                break;
            }
            int64_t globalIdx = threadElementBase + i;
            if (globalIdx >= outLength) {
                break;
            }

            int64_t indices_index = globalIdx / dataDim;
            int64_t indices_dim = globalIdx % dataDim;

            if (input[indices_index] == 0) {
                continue;
            }

            __gm__ float* first_pointer = reinterpret_cast<__gm__ float*>(input[indices_index]);
            output[globalIdx] = first_pointer[indices_dim];
        }
    }
}
}  // namespace LoadFromPointerSimt