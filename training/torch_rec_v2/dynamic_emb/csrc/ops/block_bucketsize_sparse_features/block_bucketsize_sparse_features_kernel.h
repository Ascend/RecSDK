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

#ifndef BLOCK_BUCKETSIZE_SPARSE_FEATURES_KERNEL_H
#define BLOCK_BUCKETSIZE_SPARSE_FEATURES_KERNEL_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
namespace BlockBucketSizeSparseSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t CACHE_ALIGN = 64;
constexpr int32_t SMALL_DATA_THRESHOLD_32 = 24 * MAX_THREADS_PER_BLOCK;  // 24576
constexpr int32_t SMALL_DATA_THRESHOLD_64 = 44 * MAX_THREADS_PER_BLOCK;  // 45056

template <typename T>
__aicore__ inline T Min(const T& a, const T& b)
{
    return (a < b) ? a : b;
}

template <typename T>
__aicore__ inline T WarpPrefixSum(T val)
{
    int32_t laneId = AscendC::Simt::GetThreadIdx<0>() % WARP_SIZE;
    if (laneId >= WARP_SIZE)
        return val;

#pragma unroll
    for (int32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        T temp = AscendC::Simt::WarpShflUpSync(val, offset);
        if (laneId >= offset) {
            val += temp;
        }
    }
    return val;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ T* input, __gm__ T* output, __gm__ T* blockSums, __ubuf__ T* sharedMemory, const int32_t totalLength,
    int32_t coreNum)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t globalTid = coreId * threadNumPerCore + tid;
    int32_t warpId = tid / WARP_SIZE;
    int32_t laneId = tid % WARP_SIZE;

    if (globalTid >= totalLength) {
        return;
    }

    T currentVal = input[globalTid];
    T threadSum = currentVal;
    // Warp级
    T warpPrefixSum = WarpPrefixSum(threadSum);
    int32_t elementsThisBlock = threadNumPerCore;
    int32_t activeWarpCount = (elementsThisBlock + WARP_SIZE - 1) / WARP_SIZE;
    int32_t elementsInThisWarp = (warpId < activeWarpCount - 1) ? WARP_SIZE : (elementsThisBlock - warpId * WARP_SIZE);
    // Warp级写入
    if (laneId == elementsInThisWarp - 1 && warpId < activeWarpCount && warpId < MAX_WARPS) {
        sharedMemory[warpId] = warpPrefixSum;
    }
    AscendC::Simt::ThreadBarrier();

    // Block级
    if (tid < activeWarpCount && tid < MAX_WARPS) {
        T warpSumValue = sharedMemory[tid];
        T warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedMemory[tid] = warpSumPrefix;
    }
    AscendC::Simt::ThreadBarrier();

    // Exclusive前缀和
    T blockOffset = static_cast<T>(0);
    if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MAX_WARPS) {
        blockOffset = sharedMemory[warpId - 1];
    }
    T finalPrefixSum = blockOffset + warpPrefixSum - currentVal;

    output[globalTid] = finalPrefixSum;

    // 当前核心总和
    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);
    if (tid == 0) {
        T coreSum = (activeWarpCount > 0 && activeWarpCount - 1 < MAX_WARPS) ? sharedMemory[activeWarpCount - 1] : 0;
        if (coreId * stride < coreNum * stride) {
            blockSums[coreId * stride] = coreSum;
        }
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataUpdate(__gm__ T* output,
                                                                                           __gm__ T* blockSums,
                                                                                           const int32_t totalLength,
                                                                                           int32_t coreNum)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t globalTid = coreId * threadNumPerCore + tid;

    if (globalTid >= totalLength) {
        return;
    }

    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);
    T coreOffset = static_cast<T>(0);
    int32_t totalBlocks = (totalLength + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    for (int32_t c = 0; c < coreId && c < totalBlocks; ++c) {
        coreOffset += blockSums[c * stride];
    }

    // 更新输出
    output[globalTid] += coreOffset;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ T* input, __gm__ T* output, __gm__ T* blockSums, __ubuf__ T* sharedMemory, const int32_t totalLength,
    int32_t coreNum, int32_t blockStartIdx, int32_t curBlocksCount)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadIdxInt = static_cast<int32_t>(tid);
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t warpId = threadIdxInt / WARP_SIZE;
    int32_t laneId = threadIdxInt % WARP_SIZE;
    int32_t blockElementCapacity = threadNumPerCore * MAX_ELEMENTS_PER_THREAD;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;

        // 剩余元素数
        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0)
            continue;

        // 线程内处理范围
        int32_t threadElementBase = blockBase + threadIdxInt * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = elementsThisBlock - threadIdxInt * MAX_ELEMENTS_PER_THREAD;
        if (elementsForThread < 0)
            elementsForThread = 0;
        if (elementsForThread > MAX_ELEMENTS_PER_THREAD)
            elementsForThread = MAX_ELEMENTS_PER_THREAD;
        if (elementsForThread <= 0)
            continue;

        // 线程内累加
        T threadSum = 0;
        T prefixSums[MAX_ELEMENTS_PER_THREAD] = {0};
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t elemIdx = threadElementBase + i;
                if (elemIdx >= 0 && elemIdx < totalLength) {
                    T value = input[elemIdx];
                    prefixSums[i] = threadSum;
                    threadSum += value;
                }
            }
        }

        // Warp级
        T warpPrefixSum = WarpPrefixSum(threadSum);

        int32_t activeThreads = (elementsThisBlock + MAX_ELEMENTS_PER_THREAD - 1) / MAX_ELEMENTS_PER_THREAD;
        int32_t activeWarpCount = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
        int32_t threadsInWarp = activeThreads - warpId * WARP_SIZE;
        if (threadsInWarp < 0)
            threadsInWarp = 0;
        if (threadsInWarp > WARP_SIZE)
            threadsInWarp = WARP_SIZE;

        if (threadsInWarp > 0 && warpId < activeWarpCount && warpId < MAX_WARPS && laneId == threadsInWarp - 1) {
            sharedMemory[warpId] = warpPrefixSum;
        }
        AscendC::Simt::ThreadBarrier();

        // Block级
        if (threadIdxInt < activeWarpCount && threadIdxInt < MAX_WARPS) {
            T warpSumValue = sharedMemory[threadIdxInt];
            T warpSumPrefix = WarpPrefixSum(warpSumValue);
            sharedMemory[threadIdxInt] = warpSumPrefix;
        }
        AscendC::Simt::ThreadBarrier();

        T blockOffset = 0;
        if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MAX_WARPS) {
            blockOffset = sharedMemory[warpId - 1];
        }
        T finalOffset = blockOffset + warpPrefixSum - threadSum;

        // 输出
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t globalIdx = threadElementBase + i;
                if (globalIdx >= 0 && globalIdx < totalLength) {
                    output[globalIdx] = finalOffset + prefixSums[i];
                }
            }
        }

        constexpr int32_t stride = CACHE_ALIGN / sizeof(T);
        if (threadIdxInt == 0) {
            T blockSum =
                (activeWarpCount > 0 && activeWarpCount - 1 < MAX_WARPS) ? sharedMemory[activeWarpCount - 1] : 0;
            blockSums[globalBlockIdx * stride] = blockSum;
        }
        AscendC::Simt::ThreadBarrier();
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataUpdate(
    __gm__ T* output, __gm__ T* blockSums, const int32_t totalLength, int32_t coreNum, int32_t blockStartIdx,
    int32_t curBlocksCount)
{
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadIdxInt = static_cast<int32_t>(tid);
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = threadNumPerCore * MAX_ELEMENTS_PER_THREAD;

    constexpr int32_t stride = CACHE_ALIGN / sizeof(T);
    T blockPrefix = 0;
    // 累加当前Block之前的所有BlockSum
    for (int32_t i = 0; i < blockStartIdx; ++i) {
        blockPrefix += blockSums[i * stride];
    }

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= totalLength)
            break;

        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0)
            continue;

        // 线程内处理范围
        int32_t threadElementBase = blockBase + threadIdxInt * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = elementsThisBlock - threadIdxInt * MAX_ELEMENTS_PER_THREAD;
        if (elementsForThread < 0)
            elementsForThread = 0;
        if (elementsForThread > MAX_ELEMENTS_PER_THREAD)
            elementsForThread = MAX_ELEMENTS_PER_THREAD;
        if (elementsForThread <= 0)
            continue;

        // 更新
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t globalIdx = threadElementBase + i;
                if (globalIdx >= 0 && globalIdx < totalLength) {
                    output[globalIdx] += blockPrefix;
                }
            }
        }

        // 更新前缀和（包含当前Block）
        blockPrefix += blockSums[globalBlockIdx * stride];
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtComputeNewLengths(
    const __gm__ int64_t* offsets, const __gm__ T* indices, const __gm__ T* blockSizes,
    const __gm__ T* distTypePerFeature, __gm__ T* newLengths, int32_t lengthSize, int32_t B, int32_t mySize)
{
    using uindex_t = typename std::make_unsigned<T>::type;
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t coreNum = AscendC::Simt::GetBlockNum();

    for (int32_t feature = coreId * threadNumPerCore + threadIdx; feature < lengthSize;
         feature += coreNum * threadNumPerCore) {
        const auto t = feature / B;

        bool useRoundRobin = distTypePerFeature ? (distTypePerFeature[t] != 0) : false;
        uindex_t blkSize = blockSizes[t];

        T rowstart = (feature == 0) ? 0 : offsets[feature - 1];
        T rowend = offsets[feature];

        for (auto i = rowstart; i < rowend; i++) {
            uindex_t idx = static_cast<uindex_t>(indices[i]);
            uindex_t p = 0;

            if (useRoundRobin) {
                p = idx % mySize;
            } else {
                p = (idx < blkSize * mySize) ? (idx / blkSize) : (idx % mySize);
            }

            newLengths[p * lengthSize + feature] += 1;
        }
    }

    AscendC::Simt::ThreadBarrier();
}

// 得到newIndices
template <bool sequence, bool hasWeight, bool bucketizePos, typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtRearrangeData(
    const __gm__ int64_t* offsets, const __gm__ T* indices, const __gm__ float* weights, const __gm__ T* blockSizes,
    const __gm__ T* distTypePerFeature, __gm__ T* newOffsets, __gm__ T* newIndices, __gm__ float* newWeights,
    __gm__ T* newPos, __gm__ T* unbucketizePermute, int32_t lengthSize, int32_t B, int32_t mySize)
{
    using uindex_t = typename std::make_unsigned<T>::type;
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t coreNum = AscendC::Simt::GetBlockNum();

    for (int32_t feature = coreId * threadNumPerCore + threadIdx; feature < lengthSize;
         feature += coreNum * threadNumPerCore) {
        const auto t = feature / B;
        bool useRoundRobin = distTypePerFeature ? (distTypePerFeature[t] != 0) : false;
        T blkSize = blockSizes[t];

        uindex_t rowstart = (feature == 0) ? 0 : offsets[feature - 1];
        uindex_t rowend = offsets[feature];

        for (uindex_t i = rowstart; i < rowend; ++i) {
            uindex_t idx = static_cast<uindex_t>(indices[i]);
            uindex_t p = 0;
            uindex_t newIdx = 0;

            if (useRoundRobin) {
                p = idx % mySize;
                newIdx = idx;
            } else {
                p = (idx < blkSize * mySize) ? (idx / blkSize) : (idx % mySize);
                newIdx = (idx < blkSize * mySize) ? (idx % blkSize) : (idx / mySize);
            }

            uindex_t pos = newOffsets[p * lengthSize + feature];

            newIndices[pos] = newIdx;

            newOffsets[p * lengthSize + feature]++;

            if (sequence) {
                unbucketizePermute[i] = pos;
            }

            if (hasWeight) {
                newWeights[pos] = weights[i];
            }

            if (bucketizePos) {
                newPos[pos] = i - rowstart;
            }
        }
    }

    Simt::ThreadBarrier();
}
}  // namespace BlockBucketSizeSparseSimt

#endif  // BLOCK_BUCKETSIZE_SPARSE_FEATURES_KERNEL_H
