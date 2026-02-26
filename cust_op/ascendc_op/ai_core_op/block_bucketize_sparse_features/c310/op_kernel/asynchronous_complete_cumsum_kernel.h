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

#ifndef ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H
#define ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace AsynchronousCompleteCumsumSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t CACHE_ALIGN = 64;

// Warp级前缀和计算
template<typename T>
__aicore__ inline T WarpPrefixSum(T val)
{
    int32_t laneId = AscendC::Simt::GetThreadIdx<0>() % WARP_SIZE;
#pragma unroll
    for (int32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        T temp = AscendC::Simt::WarpShflUpSync(val, offset);
        if (laneId >= offset) {
            val += temp;
        }
    }
    return val;
}

// 小数据模式中与Warp聚合相关的公共逻辑
template<typename T>
__aicore__ inline bool PrepareWarpAggregates(__gm__ T* input, __ubuf__ T* sharedMemory,
                                             int64_t totalLength, int32_t blockIdx,
                                             int32_t blockDim, int32_t threadIdx,
                                             int32_t warpId, int32_t laneId, int32_t globalIdx,
                                             int32_t& activeWarpCount, T& currentSum,
                                             T& warpPrefixSum)
{
    if (globalIdx > totalLength) {
        return false;
    }

    int32_t blockStart = blockIdx * blockDim;
    int32_t elementsRemaining = totalLength - blockStart;
    elementsRemaining = elementsRemaining < 0 ? 0 : elementsRemaining;
    int32_t elementsThisBlock = (elementsRemaining < blockDim) ? elementsRemaining : blockDim;
    activeWarpCount = (elementsThisBlock + WARP_SIZE - 1) / WARP_SIZE;

    // 1. 读取输入数据
    currentSum = (globalIdx < totalLength) ? input[globalIdx] : static_cast<T>(0);
    // 2. Warp级前缀和计算
    warpPrefixSum = WarpPrefixSum(currentSum);

    // 3. Block级同步
    int32_t elementsInThisWarp = (warpId < activeWarpCount - 1) ? WARP_SIZE :
                                                                (elementsThisBlock - warpId * WARP_SIZE);
    if (laneId == elementsInThisWarp - 1 && warpId < activeWarpCount && warpId < MAX_WARPS) {
        sharedMemory[warpId] = warpPrefixSum;
    }
    AscendC::Simt::ThreadBarrier();

    // 4. Block级前缀和计算
    if (threadIdx < activeWarpCount && threadIdx < MAX_WARPS) {
        T warpSumValue = sharedMemory[threadIdx];
        T warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedMemory[threadIdx] = warpSumPrefix;
    }
    AscendC::Simt::ThreadBarrier();

    return true;
}

// SIMT VF函数 - 小数据模式第一阶段
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
    inline void SimtSmallDataCompute(__gm__ T* input, __gm__ T* output,
                                     volatile __gm__ T* blockSums, __ubuf__ T* sharedMemory,
                                     int64_t totalLength, int32_t activeBlockNum)
{
    // 线程信息计算
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    int32_t globalIdx = blockIdx * blockDim + threadIdx;
    int32_t warpId = threadIdx / WARP_SIZE;
    int32_t laneId = threadIdx % WARP_SIZE;

    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);
    int32_t activeWarpCount = 0;
    T currentSum = static_cast<T>(0);
    T warpPrefixSum = static_cast<T>(0);

    int32_t blockNum = (activeBlockNum <= 0) ? 0 : activeBlockNum;
    if (blockIdx >= blockNum) {
        return;
    }
    if (!PrepareWarpAggregates(input, sharedMemory, totalLength, blockIdx, blockDim, threadIdx,
                               warpId, laneId, globalIdx, activeWarpCount, currentSum, warpPrefixSum)) {
        return;
    }

    // 5. 计算最终前缀和
    T blockOffset = static_cast<T>(0);
    if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MAX_WARPS) {
        blockOffset = sharedMemory[warpId - 1];
    }
    T finalPrefixSum = blockOffset + warpPrefixSum - currentSum;

    // 6. 写入输出
    if (globalIdx < totalLength) {
        output[globalIdx] = finalPrefixSum;
    }

    // 7. 保存block结果
    if (threadIdx == 0) {
        if (activeWarpCount > 0) {
            int32_t lastWarpIdx = activeWarpCount - 1;
            if (lastWarpIdx >= 0 && lastWarpIdx < MAX_WARPS) {
                T blockSum = sharedMemory[lastWarpIdx];
                blockSums[blockIdx * stride] = blockSum;
            }
        } else {
            blockSums[blockIdx * stride] = static_cast<T>(0);
        }
    }

    AscendC::Simt::ThreadBarrier();

    // 8. 计算总和（仅单块场景）
    if (blockNum == 1 && threadIdx == 0) {
        T totalSum = static_cast<T>(0);
        for (int32_t i = 0; i < blockNum; ++i) {
            totalSum += blockSums[i * stride];
        }
        output[totalLength] = totalSum;
    }
}

// SIMT VF函数 - 小数据模式第二阶段
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
    inline void SimtSmallDataUpdate(__gm__ T* output, volatile __gm__ T* blockSums,
                                    int64_t totalLength, int32_t activeBlockNum)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    int32_t globalIdx = blockIdx * blockDim + threadIdx;

    int32_t blockNum = (activeBlockNum <= 0) ? 0 : activeBlockNum;
    if ((blockIdx >= blockNum) || (globalIdx >= totalLength)) {
        return;
    }

    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);

    // 计算前面所有block的偏移
    T blockOffset = static_cast<T>(0);
    for (int32_t i = 0; i < blockIdx; ++i) {
        blockOffset += blockSums[i * stride];
    }

    // 更新输出
    output[globalIdx] += blockOffset;

    // 最后一个线程计算总和
    if (globalIdx == totalLength - 1 && blockIdx == blockNum - 1) {
        T totalSum = static_cast<T>(0);
        for (int32_t i = 0; i < blockNum; ++i) {
            totalSum += blockSums[i * stride];
        }
        output[totalLength] = totalSum;
    }
}

template<typename T>
__aicore__ inline void FinalizeLargeDataBlock(__gm__ T* output, volatile __gm__ T* blockSums,
                                              __ubuf__ T* sharedMemory, int64_t totalLength,
                                              int32_t globalBlockIdx, int32_t threadElementBase,
                                              int32_t elementsForThread, int32_t threadIdx,
                                              int32_t warpId, int32_t laneId,
                                              int32_t activeWarpCount, int32_t threadsInWarp,
                                              T warpPrefixSum, T threadSum, T* prefixSums,
                                              int32_t stride)
{
    if (threadsInWarp > 0 && warpId < activeWarpCount && warpId < MAX_WARPS &&
        laneId == (threadsInWarp - 1)) {
        sharedMemory[warpId] = warpPrefixSum;
    }
    AscendC::Simt::ThreadBarrier();

    if (threadIdx < activeWarpCount && threadIdx < MAX_WARPS) {
        T warpSumValue = sharedMemory[threadIdx];
        T warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedMemory[threadIdx] = warpSumPrefix;
    }
    AscendC::Simt::ThreadBarrier();

    T blockOffset = static_cast<T>(0);
    if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MAX_WARPS) {
        blockOffset = sharedMemory[warpId - 1];
    }

    T finalOffset = blockOffset + warpPrefixSum - threadSum;

#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
        if (i < elementsForThread) {
            int32_t globalIdx = threadElementBase + i;
            if (globalIdx < totalLength) {
                output[globalIdx] = finalOffset + prefixSums[i];
            }
        }
    }

    AscendC::Simt::ThreadBarrier();
    if (threadIdx == 0) {
        T blockSum = static_cast<T>(0);
        if (activeWarpCount > 0) {
            blockSum = sharedMemory[activeWarpCount - 1];
        }
        blockSums[globalBlockIdx * stride] = blockSum;
    }
    AscendC::Simt::ThreadBarrier();
}

// SIMT VF函数 - 大数据模式第一阶段
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
    inline void SimtLargeDataCompute(__gm__ T* input, __gm__ T* output,
                                     __gm__ T* blockSums, __ubuf__ T* sharedMemory,
                                     int64_t totalLength, int64_t totalBlocks,
                                     int64_t blockStartIdx, int64_t curBlocksCount)
{
    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);

    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    int32_t warpId = threadIdx / WARP_SIZE;
    int32_t laneId = threadIdx % WARP_SIZE;
    int32_t blockElementCapacity = blockDim * MAX_ELEMENTS_PER_THREAD;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        if (globalBlockIdx >= totalBlocks) {
            break;
        }

        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= totalLength) {
            break;
        }

        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock = (elementsRemaining < blockElementCapacity) ?
                                                                               elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
        elementsForThread = (elementsForThread < 0) ? 0 : elementsForThread;
        elementsForThread = (elementsForThread > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD : elementsForThread;

        if (threadElementBase >= totalLength) {
            elementsForThread = 0;
        }

        T threadSum = static_cast<T>(0);
        T prefixSums[MAX_ELEMENTS_PER_THREAD] = {static_cast<T>(0)};
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                T value = input[threadElementBase + i];
                prefixSums[i] = threadSum;
                threadSum += value;
            }
        }

        T warpPrefixSum = WarpPrefixSum(threadSum);

        int32_t activeThreads = (elementsThisBlock + MAX_ELEMENTS_PER_THREAD - 1) / MAX_ELEMENTS_PER_THREAD;
        int32_t activeWarpCount = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
        int32_t threadsInWarp = activeThreads - warpId * WARP_SIZE;
        threadsInWarp = (threadsInWarp > WARP_SIZE) ? WARP_SIZE : threadsInWarp;
        threadsInWarp = (threadsInWarp < 0) ? 0 : threadsInWarp;

        FinalizeLargeDataBlock(output, blockSums, sharedMemory, totalLength,
                               globalBlockIdx, threadElementBase, elementsForThread,
                               threadIdx, warpId, laneId, activeWarpCount, threadsInWarp,
                               warpPrefixSum, threadSum, prefixSums, stride);
    }
}

// SIMT VF函数 - 大数据模式第二阶段
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
    inline void SimtLargeDataUpdate(__gm__ T* output, volatile __gm__ T* blockSums,
                                    int64_t totalLength, int64_t totalBlocks,
                                    int64_t blockStartIdx, int64_t curBlocksCount)
{
    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);

    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    (void)blockIdx;  // 保持接口一致

    int32_t blockElementCapacity = blockDim * MAX_ELEMENTS_PER_THREAD;

    T blockPrefix = static_cast<T>(0);
    for (int32_t i = 0; i < blockStartIdx && i < totalBlocks; ++i) {
        blockPrefix += blockSums[i * stride];
    }

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        if (globalBlockIdx >= totalBlocks) {
            break;
        }

        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= totalLength) {
            break;
        }

        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock = (elementsRemaining < blockElementCapacity) ?
                                                                               elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
        if (elementsForThread < 0) {
            elementsForThread = 0;
        }
        if (elementsForThread > MAX_ELEMENTS_PER_THREAD) {
            elementsForThread = MAX_ELEMENTS_PER_THREAD;
        }

        for (int32_t i = 0; i < elementsForThread; ++i) {
            int32_t globalIdx = threadElementBase + i;
            if (globalIdx < totalLength) {
                output[globalIdx] += blockPrefix;
            }
        }

        if (threadIdx == 0 && globalBlockIdx == totalBlocks - 1) {
            output[totalLength] = blockPrefix + blockSums[globalBlockIdx * stride];
        }

        blockPrefix += blockSums[globalBlockIdx * stride];
    }
}

} // namespace AsynchronousCompleteCumsumSimt

#endif // ASYNCHRONOUS_COMPLETE_CUMSUM_KERNEL_H
