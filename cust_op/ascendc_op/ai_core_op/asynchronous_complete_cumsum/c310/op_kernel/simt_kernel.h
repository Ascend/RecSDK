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

#ifndef SIMT_KERNEL_H
#define SIMT_KERNEL_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t DATA_ALIGN_BYTES = 32;

namespace CumsumSimt {

// Warp级前缀和计算
template <typename T>
__aicore__ inline T WarpPrefixSum(T val)
{
    int32_t laneId = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        T temp = asc_shfl_up(val, offset);
        if (laneId >= offset) {
            val += temp;
        }
    }
    return val;
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
    inline void SmallDataCompute(__local_mem__ T* input, __local_mem__ T* output,
                                 __gm__ T* blockSums, __ubuf__ T* sharedMemory,
                                 int32_t elementsThisBlock, int64_t blockIdx)
{
    // 线程信息计算
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    const int32_t laneId = threadIdx.x % WARP_SIZE;

    int32_t activeWarpCount = (elementsThisBlock + WARP_SIZE - 1) / WARP_SIZE;

    T currentValue = (threadIdx.x < elementsThisBlock) ? input[threadIdx.x] : static_cast<T>(0);
    T warpPrefixSum = WarpPrefixSum(currentValue);

    int32_t elementsInThisWarp = (warpId < activeWarpCount - 1) ? WARP_SIZE :
                                                                (elementsThisBlock - warpId * WARP_SIZE);
    elementsInThisWarp = (warpId >= activeWarpCount) ? 0 : elementsInThisWarp;
    if (laneId == elementsInThisWarp - 1 && warpId < activeWarpCount) {
        sharedMemory[warpId] = warpPrefixSum;
    }
    asc_syncthreads();

    if (threadIdx.x < activeWarpCount) {
        T warpSumValue = sharedMemory[threadIdx.x];
        T warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedMemory[threadIdx.x] = warpSumPrefix;
    }
    asc_syncthreads();

    T warpExclusive = static_cast<T>(0);
    if (warpId > 0 && warpId < activeWarpCount) {
        warpExclusive = sharedMemory[warpId - 1];
    }

    T finalPrefixSum = warpExclusive + warpPrefixSum;

    if (threadIdx.x < elementsThisBlock) {
        output[threadIdx.x] = finalPrefixSum;
    }

    // 保存block结果
    if (threadIdx.x == 0) {
        blockSums[blockIdx] = sharedMemory[activeWarpCount - 1];
    }
}


// SIMT VF函数 - 大数据模式（每线程处理最多4个元素）
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
    inline void LargeDataCompute(__local_mem__ T *input, __local_mem__ T *output,
                                 __gm__ T *blockSums, __ubuf__ T *sharedMemory,
                                 int32_t elementsThisBlock, int64_t blockIdx)
{
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    const int32_t laneId = threadIdx.x % WARP_SIZE;

    int32_t threadElementBase = threadIdx.x * MAX_ELEMENTS_PER_THREAD;
    int32_t elementsForThread = Std::max(0, Std::min(elementsThisBlock - threadElementBase, MAX_ELEMENTS_PER_THREAD));

    T prefixSums[MAX_ELEMENTS_PER_THREAD] = {static_cast<T>(0)};
    T threadSum = static_cast<T>(0);
#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
        if (i < elementsForThread) {
            T value = input[threadElementBase + i];
            threadSum += value;
            prefixSums[i] = threadSum;
        }
    }

    int32_t activeThreads = (elementsThisBlock + MAX_ELEMENTS_PER_THREAD - 1) / MAX_ELEMENTS_PER_THREAD;
    int32_t activeWarpCount = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
    int32_t threadsInWarp = Std::max(0, Std::min(activeThreads - warpId * WARP_SIZE, WARP_SIZE));

    T warpPrefixSum = WarpPrefixSum(threadSum);

    if (warpId < activeWarpCount && laneId == (threadsInWarp - 1)) {
        sharedMemory[warpId] = warpPrefixSum;
    }
    asc_syncthreads();

    if (threadIdx.x < activeWarpCount) {
        T warpSumValue = sharedMemory[threadIdx.x];
        T warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedMemory[threadIdx.x] = warpSumPrefix;
    }
    asc_syncthreads();

    T warpExclusive = static_cast<T>(0);
    if (warpId > 0 && warpId < activeWarpCount) {
        warpExclusive = sharedMemory[warpId - 1];
    }
    T blockOffset = warpExclusive + warpPrefixSum - threadSum;

#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
        if (i < elementsForThread) {
            int32_t localIdx = threadElementBase + i;
            if (localIdx < elementsThisBlock) {
                output[localIdx] = blockOffset + prefixSums[i];
            }
        }
    }

    asc_syncthreads();
    if (threadIdx.x == 0) {
        blockSums[blockIdx] = sharedMemory[activeWarpCount - 1];
    }
}

}  // namespace CumsumSimt

#endif  // SIMT_KERNEL_H
