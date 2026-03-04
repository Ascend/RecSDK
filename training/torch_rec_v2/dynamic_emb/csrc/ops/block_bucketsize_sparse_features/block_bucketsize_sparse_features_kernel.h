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
constexpr int32_t UNROLL_FACTOR = 4;
constexpr int32_t MAX_FEATURE_NUM_USE_QUICK_DIVIDE = 500;

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
    int32_t coreNum, int32_t stride)
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
                                                                                           int32_t coreNum,
                                                                                           int32_t stride)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t globalTid = coreId * threadNumPerCore + tid;

    if (globalTid >= totalLength) {
        return;
    }

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
    int32_t coreNum, int32_t blockStartIdx, int32_t curBlocksCount, int32_t stride)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadIdxInt = static_cast<int32_t>(tid);
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t warpId = threadIdxInt / WARP_SIZE;
    int32_t laneId = threadIdxInt % WARP_SIZE;
    int32_t blockElementCapacity = threadNumPerCore * MAX_ELEMENTS_PER_THREAD;
    int32_t threadElementOffset = threadIdxInt * MAX_ELEMENTS_PER_THREAD;

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
        int32_t threadElementBase = blockBase + threadElementOffset;
        int32_t elementsForThread = elementsThisBlock - threadElementOffset;
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
    int32_t curBlocksCount, int32_t stride)
{
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadIdxInt = static_cast<int32_t>(tid);
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = threadNumPerCore * MAX_ELEMENTS_PER_THREAD;
    int32_t threadElementOffset = threadIdxInt * MAX_ELEMENTS_PER_THREAD;

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
        int32_t threadElementBase = blockBase + threadElementOffset;
        int32_t elementsForThread = elementsThisBlock - threadElementOffset;
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

/* 本函数原公式为：bucket = idx % mySize
 * 现由快除法计算出div_res = idx / mySize，再由bucket = idx - div_res * mySize
 * uint64_t快除法：
 * r = x / y，如果y的值固定，则除法可以等效替换为如下公式
 * 预计算部分（host）：
 * 1. shift = ceil(log2(y))
 * 2. magic = ceil(2 ^ (64 + shift) / y)
 * 运行时计算部分（本函数内）：
 * 3. q = (x * magic) >> 64 由__umul64hi完成
 * 4. t = ((x - q) >> 1) + q
 * 5. r = t >> (shift - 1)
 */
__aicore__ inline uint64_t QuickRem(
    const uint64_t& x, const uint64_t& divisorMagic, const uint64_t& divisorShift, const uint64_t& y)
{
    uint64_t divTmp = __umul64hi(x, divisorMagic);
    divTmp = ((x - divTmp) >> 1) + divTmp;
    uint64_t divResult = divTmp >> divisorShift;
    return x - divResult * y;
}

__aicore__ inline uint64_t QuickDiv(
    const uint64_t& x, const uint64_t& divisorMagic, const uint64_t& divisorShift)
{
    uint64_t divTmp = __umul64hi(x, divisorMagic);
    divTmp = ((x - divTmp) >> 1) + divTmp;
    uint64_t divResult = divTmp >> divisorShift;
    return divResult;
}

template <typename T, bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtComputeNewLengths(
    const __gm__ int64_t* offsets, const __gm__ T* indices, const __gm__ T* blockSizes,
    const __gm__ T* distTypePerFeature, __gm__ T* newLengths, int32_t lengthSize, int32_t B, const int32_t mySize)
{
    using uindex_t = typename std::make_unsigned<T>::type;
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t coreNum = AscendC::Simt::GetBlockNum();
    const uindex_t mySizeMask = (isPowerOfTwo) ? (mySize - 1) : 0;

    for (int32_t feature = coreId * threadNumPerCore + threadIdx; feature < lengthSize;
         feature += coreNum * threadNumPerCore) {
        const auto t = feature / B;

        bool useRoundRobin = distTypePerFeature ? (distTypePerFeature[t] != 0) : false;
        int32_t rowstart = (feature == 0) ? 0 : offsets[feature - 1];
        int32_t rowend = offsets[feature];

        if (useRoundRobin) {
            uindex_t i = rowstart;

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                uindex_t p0 = isPowerOfTwo ? (idx0 & mySizeMask) : (idx0 % mySize);
                uindex_t p1 = isPowerOfTwo ? (idx1 & mySizeMask) : (idx1 % mySize);
                uindex_t p2 = isPowerOfTwo ? (idx2 & mySizeMask) : (idx2 % mySize);
                uindex_t p3 = isPowerOfTwo ? (idx3 & mySizeMask) : (idx3 % mySize);

                newLengths[p0 * lengthSize + feature] += 1;
                newLengths[p1 * lengthSize + feature] += 1;
                newLengths[p2 * lengthSize + feature] += 1;
                newLengths[p3 * lengthSize + feature] += 1;
            }

            for (; i < rowend; i++) {
                uindex_t idx = static_cast<uindex_t>(indices[i]);
                uindex_t p = isPowerOfTwo ? (idx & mySizeMask) : (idx % mySize);
                newLengths[p * lengthSize + feature] += 1;
            }

        } else {
            uindex_t blkSize = blockSizes[t];
            const uindex_t blkSizeMulMySize = blkSize * mySize;
            uindex_t i = rowstart;

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                const uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                const uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                const uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                const uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                const uindex_t p0 = (idx0 < blkSizeMulMySize) ? (idx0 / blkSize) : (idx0 % mySize);
                const uindex_t p1 = (idx1 < blkSizeMulMySize) ? (idx1 / blkSize) : (idx1 % mySize);
                const uindex_t p2 = (idx2 < blkSizeMulMySize) ? (idx2 / blkSize) : (idx2 % mySize);
                const uindex_t p3 = (idx3 < blkSizeMulMySize) ? (idx3 / blkSize) : (idx3 % mySize);

                newLengths[p0 * lengthSize + feature] += 1;
                newLengths[p1 * lengthSize + feature] += 1;
                newLengths[p2 * lengthSize + feature] += 1;
                newLengths[p3 * lengthSize + feature] += 1;
            }

            for (; i < rowend; i++) {
                const uindex_t idx = static_cast<uindex_t>(indices[i]);
                const uindex_t p = (idx < blkSizeMulMySize) ? (idx / blkSize) : (idx % mySize);
                newLengths[p * lengthSize + feature] += 1;
            }
        }
    }

    AscendC::Simt::ThreadBarrier();
}

template <typename T, bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtComputeNewLengthsQuickDiv(
    const __gm__ int64_t* offsets, const __gm__ T* indices, const __gm__ T* blockSizes,
    const __gm__ T* distTypePerFeature, __gm__ T* newLengths, int32_t lengthSize, int32_t B, const uint64_t mySize,
    const uint64_t mySizeMagic, const uint64_t mySizeShift,
    __ubuf__ typename std::make_unsigned<T>::type* blkSizeMagicShifts)
{
    using uindex_t = typename std::make_unsigned<T>::type;
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t coreNum = AscendC::Simt::GetBlockNum();
    const uindex_t mySizeMask = (isPowerOfTwo) ? (mySize - 1) : 0;

    for (int32_t feature = coreId * threadNumPerCore + threadIdx; feature < lengthSize;
         feature += coreNum * threadNumPerCore) {
        const auto t = feature / B;

        bool useRoundRobin = distTypePerFeature ? (distTypePerFeature[t] != 0) : false;
        int32_t rowstart = (feature == 0) ? 0 : offsets[feature - 1];
        int32_t rowend = offsets[feature];

        if (useRoundRobin) {
            uindex_t i = rowstart;

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                uindex_t p0 = isPowerOfTwo ? (idx0 & mySizeMask) : (idx0 % mySize);
                uindex_t p1 = isPowerOfTwo ? (idx1 & mySizeMask) : (idx1 % mySize);
                uindex_t p2 = isPowerOfTwo ? (idx2 & mySizeMask) : (idx2 % mySize);
                uindex_t p3 = isPowerOfTwo ? (idx3 & mySizeMask) : (idx3 % mySize);

                newLengths[p0 * lengthSize + feature] += 1;
                newLengths[p1 * lengthSize + feature] += 1;
                newLengths[p2 * lengthSize + feature] += 1;
                newLengths[p3 * lengthSize + feature] += 1;
            }

            for (; i < rowend; i++) {
                uindex_t idx = static_cast<uindex_t>(indices[i]);
                uindex_t p = isPowerOfTwo ? (idx & mySizeMask) : (idx % mySize);
                newLengths[p * lengthSize + feature] += 1;
            }

        } else {
            uindex_t blkSize = blockSizes[t];
            const uindex_t blkSizeMulMySize = blkSize * mySize;
            uindex_t i = rowstart;

            uindex_t blkSizeMagic = blkSizeMagicShifts[t * 2];
            uindex_t blkSizeShift = blkSizeMagicShifts[t * 2 + 1];

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                const uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                const uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                const uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                const uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                const uindex_t p0 = (idx0 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx0), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx0), mySizeMagic, mySizeShift, mySize));
                const uindex_t p1 = (idx1 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx1), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx1), mySizeMagic, mySizeShift, mySize));
                const uindex_t p2 = (idx2 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx2), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx2), mySizeMagic, mySizeShift, mySize));
                const uindex_t p3 = (idx3 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx3), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx3), mySizeMagic, mySizeShift, mySize));

                newLengths[p0 * lengthSize + feature] += 1;
                newLengths[p1 * lengthSize + feature] += 1;
                newLengths[p2 * lengthSize + feature] += 1;
                newLengths[p3 * lengthSize + feature] += 1;
            }

            for (; i < rowend; i++) {
                const uindex_t idx = static_cast<uindex_t>(indices[i]);
                const uindex_t p = (idx < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx), mySizeMagic, mySizeShift, mySize));
                newLengths[p * lengthSize + feature] += 1;
            }
        }
    }

    AscendC::Simt::ThreadBarrier();
}

// 得到newIndices
template <bool sequence, bool hasWeight, bool bucketizePos, typename T, bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtRearrangeData(
    const __gm__ int64_t* offsets, const __gm__ T* indices, const __gm__ float* weights, const __gm__ T* blockSizes,
    const __gm__ T* distTypePerFeature, __gm__ T* newOffsets, __gm__ T* newIndices, __gm__ float* newWeights,
    __gm__ T* newPos, __gm__ T* unbucketizePermute, int32_t lengthSize, int32_t B, const int32_t mySize)
{
    using uindex_t = typename std::make_unsigned<T>::type;
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t coreNum = AscendC::Simt::GetBlockNum();
    const uindex_t mySizeMask = (isPowerOfTwo) ? (mySize - 1) : 0;

    for (int32_t feature = coreId * threadNumPerCore + threadIdx; feature < lengthSize;
         feature += coreNum * threadNumPerCore) {
        const auto t = feature / B;
        bool useRoundRobin = distTypePerFeature ? (distTypePerFeature[t] != 0) : false;

        int32_t rowstart = (feature == 0) ? 0 : offsets[feature - 1];
        int32_t rowend = offsets[feature];
        if (useRoundRobin) {
            uindex_t i = rowstart;

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                const uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                const uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                const uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                const uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                const uindex_t p0 = isPowerOfTwo ? (idx0 & mySizeMask) : (idx0 % mySize);
                const uindex_t p1 = isPowerOfTwo ? (idx1 & mySizeMask) : (idx1 % mySize);
                const uindex_t p2 = isPowerOfTwo ? (idx2 & mySizeMask) : (idx2 % mySize);
                const uindex_t p3 = isPowerOfTwo ? (idx3 & mySizeMask) : (idx3 % mySize);

                const uindex_t offset0 = p0 * lengthSize + feature;
                const uindex_t offset1 = p1 * lengthSize + feature;
                const uindex_t offset2 = p2 * lengthSize + feature;
                const uindex_t offset3 = p3 * lengthSize + feature;

                const uindex_t pos0 = newOffsets[offset0];
                newOffsets[offset0]++;
                const uindex_t pos1 = newOffsets[offset1];
                newOffsets[offset1]++;
                const uindex_t pos2 = newOffsets[offset2];
                newOffsets[offset2]++;
                const uindex_t pos3 = newOffsets[offset3];
                newOffsets[offset3]++;

                newIndices[pos0] = static_cast<T>(idx0);
                newIndices[pos1] = static_cast<T>(idx1);
                newIndices[pos2] = static_cast<T>(idx2);
                newIndices[pos3] = static_cast<T>(idx3);

                // 更新newOffsets
                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos0);
                    unbucketizePermute[i + 1] = static_cast<T>(pos1);
                    unbucketizePermute[i + 2] = static_cast<T>(pos2);
                    unbucketizePermute[i + 3] = static_cast<T>(pos3);
                }

                if (hasWeight) {
                    newWeights[pos0] = weights[i];
                    newWeights[pos1] = weights[i + 1];
                    newWeights[pos2] = weights[i + 2];
                    newWeights[pos3] = weights[i + 3];
                }

                if (bucketizePos) {
                    newPos[pos0] = static_cast<T>(i - rowstart);
                    newPos[pos1] = static_cast<T>(i + 1 - rowstart);
                    newPos[pos2] = static_cast<T>(i + 2 - rowstart);
                    newPos[pos3] = static_cast<T>(i + 3 - rowstart);
                }
            }

            for (; i < rowend; ++i) {
                const uindex_t idx = static_cast<uindex_t>(indices[i]);
                const uindex_t p = isPowerOfTwo ? (idx & mySizeMask) : (idx % mySize);
                const uindex_t offset = p * lengthSize + feature;
                const uindex_t pos = newOffsets[offset];

                newIndices[pos] = static_cast<T>(idx);
                newOffsets[offset]++;

                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos);
                }

                if (hasWeight) {
                    newWeights[pos] = weights[i];
                }

                if (bucketizePos) {
                    newPos[pos] = static_cast<T>(i - rowstart);
                }
            }
        } else {
            uindex_t blkSize = blockSizes[t];
            const uindex_t blkSizeMulMySize = blkSize * mySize;
            uindex_t i = rowstart;

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                const uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                const uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                const uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                const uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                const uindex_t p0 = (idx0 < blkSizeMulMySize) ? (idx0 / blkSize) : (idx0 % mySize);
                const uindex_t p1 = (idx1 < blkSizeMulMySize) ? (idx1 / blkSize) : (idx1 % mySize);
                const uindex_t p2 = (idx2 < blkSizeMulMySize) ? (idx2 / blkSize) : (idx2 % mySize);
                const uindex_t p3 = (idx3 < blkSizeMulMySize) ? (idx3 / blkSize) : (idx3 % mySize);

                const uindex_t newIdx0 = (idx0 < blkSizeMulMySize) ? (idx0 % blkSize) : (idx0 / mySize);
                const uindex_t newIdx1 = (idx1 < blkSizeMulMySize) ? (idx1 % blkSize) : (idx1 / mySize);
                const uindex_t newIdx2 = (idx2 < blkSizeMulMySize) ? (idx2 % blkSize) : (idx2 / mySize);
                const uindex_t newIdx3 = (idx3 < blkSizeMulMySize) ? (idx3 % blkSize) : (idx3 / mySize);

                const uindex_t offset0 = p0 * lengthSize + feature;
                const uindex_t offset1 = p1 * lengthSize + feature;
                const uindex_t offset2 = p2 * lengthSize + feature;
                const uindex_t offset3 = p3 * lengthSize + feature;

                const uindex_t pos0 = newOffsets[offset0];
                newOffsets[offset0]++;
                const uindex_t pos1 = newOffsets[offset1];
                newOffsets[offset1]++;
                const uindex_t pos2 = newOffsets[offset2];
                newOffsets[offset2]++;
                const uindex_t pos3 = newOffsets[offset3];
                newOffsets[offset3]++;

                newIndices[pos0] = static_cast<T>(newIdx0);
                newIndices[pos1] = static_cast<T>(newIdx1);
                newIndices[pos2] = static_cast<T>(newIdx2);
                newIndices[pos3] = static_cast<T>(newIdx3);

                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos0);
                    unbucketizePermute[i + 1] = static_cast<T>(pos1);
                    unbucketizePermute[i + 2] = static_cast<T>(pos2);
                    unbucketizePermute[i + 3] = static_cast<T>(pos3);
                }

                if (hasWeight) {
                    newWeights[pos0] = weights[i];
                    newWeights[pos1] = weights[i + 1];
                    newWeights[pos2] = weights[i + 2];
                    newWeights[pos3] = weights[i + 3];
                }

                if (bucketizePos) {
                    newPos[pos0] = static_cast<T>(i - rowstart);
                    newPos[pos1] = static_cast<T>(i + 1 - rowstart);
                    newPos[pos2] = static_cast<T>(i + 2 - rowstart);
                    newPos[pos3] = static_cast<T>(i + 3 - rowstart);
                }
            }

            for (; i < rowend; ++i) {
                const uindex_t idx = static_cast<uindex_t>(indices[i]);
                const uindex_t p = (idx < blkSizeMulMySize) ? (idx / blkSize) : (idx % mySize);
                const uindex_t newIdx = (idx < blkSizeMulMySize) ? (idx % blkSize) : (idx / mySize);

                const uindex_t offset = p * lengthSize + feature;
                const uindex_t pos = newOffsets[offset];

                newIndices[pos] = static_cast<T>(newIdx);
                newOffsets[offset]++;

                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos);
                }

                if (hasWeight) {
                    newWeights[pos] = weights[i];
                }

                if (bucketizePos) {
                    newPos[pos] = static_cast<T>(i - rowstart);
                }
            }
        }
    }

    Simt::ThreadBarrier();
}

template <bool sequence, bool hasWeight, bool bucketizePos, typename T, bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtRearrangeDataQuickDiv(
    const __gm__ int64_t* offsets, const __gm__ T* indices, const __gm__ float* weights, const __gm__ T* blockSizes,
    const __gm__ T* distTypePerFeature, __gm__ T* newOffsets, __gm__ T* newIndices, __gm__ float* newWeights,
    __gm__ T* newPos, __gm__ T* unbucketizePermute, int32_t lengthSize, int32_t B, const uint64_t mySize,
    const uint64_t mySizeMagic, const uint64_t mySizeShift,
    __ubuf__ typename std::make_unsigned<T>::type* blkSizeMagicShifts)
{
    using uindex_t = typename std::make_unsigned<T>::type;
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t coreNum = AscendC::Simt::GetBlockNum();
    const uindex_t mySizeMask = (isPowerOfTwo) ? (mySize - 1) : 0;

    for (int32_t feature = coreId * threadNumPerCore + threadIdx; feature < lengthSize;
         feature += coreNum * threadNumPerCore) {
        const auto t = feature / B;
        bool useRoundRobin = distTypePerFeature ? (distTypePerFeature[t] != 0) : false;

        int32_t rowstart = (feature == 0) ? 0 : offsets[feature - 1];
        int32_t rowend = offsets[feature];
        if (useRoundRobin) {
            uindex_t i = rowstart;

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                const uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                const uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                const uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                const uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                const uindex_t p0 = isPowerOfTwo ? (idx0 & mySizeMask) : (idx0 % mySize);
                const uindex_t p1 = isPowerOfTwo ? (idx1 & mySizeMask) : (idx1 % mySize);
                const uindex_t p2 = isPowerOfTwo ? (idx2 & mySizeMask) : (idx2 % mySize);
                const uindex_t p3 = isPowerOfTwo ? (idx3 & mySizeMask) : (idx3 % mySize);

                const uindex_t offset0 = p0 * lengthSize + feature;
                const uindex_t offset1 = p1 * lengthSize + feature;
                const uindex_t offset2 = p2 * lengthSize + feature;
                const uindex_t offset3 = p3 * lengthSize + feature;

                const uindex_t pos0 = newOffsets[offset0];
                newOffsets[offset0]++;
                const uindex_t pos1 = newOffsets[offset1];
                newOffsets[offset1]++;
                const uindex_t pos2 = newOffsets[offset2];
                newOffsets[offset2]++;
                const uindex_t pos3 = newOffsets[offset3];
                newOffsets[offset3]++;

                newIndices[pos0] = static_cast<T>(idx0);
                newIndices[pos1] = static_cast<T>(idx1);
                newIndices[pos2] = static_cast<T>(idx2);
                newIndices[pos3] = static_cast<T>(idx3);

                // 更新newOffsets
                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos0);
                    unbucketizePermute[i + 1] = static_cast<T>(pos1);
                    unbucketizePermute[i + 2] = static_cast<T>(pos2);
                    unbucketizePermute[i + 3] = static_cast<T>(pos3);
                }

                if (hasWeight) {
                    newWeights[pos0] = weights[i];
                    newWeights[pos1] = weights[i + 1];
                    newWeights[pos2] = weights[i + 2];
                    newWeights[pos3] = weights[i + 3];
                }

                if (bucketizePos) {
                    newPos[pos0] = static_cast<T>(i - rowstart);
                    newPos[pos1] = static_cast<T>(i + 1 - rowstart);
                    newPos[pos2] = static_cast<T>(i + 2 - rowstart);
                    newPos[pos3] = static_cast<T>(i + 3 - rowstart);
                }
            }

            for (; i < rowend; ++i) {
                const uindex_t idx = static_cast<uindex_t>(indices[i]);
                const uindex_t p = isPowerOfTwo ? (idx & mySizeMask) : (idx % mySize);
                const uindex_t offset = p * lengthSize + feature;
                const uindex_t pos = newOffsets[offset];

                newIndices[pos] = static_cast<T>(idx);
                newOffsets[offset]++;

                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos);
                }

                if (hasWeight) {
                    newWeights[pos] = weights[i];
                }

                if (bucketizePos) {
                    newPos[pos] = static_cast<T>(i - rowstart);
                }
            }
        } else {
            uindex_t blkSize = blockSizes[t];
            const uindex_t blkSizeMulMySize = blkSize * mySize;
            uindex_t i = rowstart;

            uindex_t blkSizeMagic = blkSizeMagicShifts[t * 2];
            uindex_t blkSizeShift = blkSizeMagicShifts[t * 2 + 1];

            for (; i + (UNROLL_FACTOR - 1) < rowend; i += UNROLL_FACTOR) {
                const uindex_t idx0 = static_cast<uindex_t>(indices[i]);
                const uindex_t idx1 = static_cast<uindex_t>(indices[i + 1]);
                const uindex_t idx2 = static_cast<uindex_t>(indices[i + 2]);
                const uindex_t idx3 = static_cast<uindex_t>(indices[i + 3]);

                const uindex_t p0 = (idx0 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx0), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx0), mySizeMagic, mySizeShift, mySize));
                const uindex_t p1 = (idx1 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx1), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx1), mySizeMagic, mySizeShift, mySize));
                const uindex_t p2 = (idx2 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx2), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx2), mySizeMagic, mySizeShift, mySize));
                const uindex_t p3 = (idx3 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx3), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx3), mySizeMagic, mySizeShift, mySize));

                const uindex_t newIdx0 = (idx0 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx0), blkSizeMagic, blkSizeShift, blkSize)) :
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx0), mySizeMagic, mySizeShift));
                const uindex_t newIdx1 = (idx1 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx1), blkSizeMagic, blkSizeShift, blkSize)) :
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx1), mySizeMagic, mySizeShift));
                const uindex_t newIdx2 = (idx2 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx2), blkSizeMagic, blkSizeShift, blkSize)) :
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx2), mySizeMagic, mySizeShift));
                const uindex_t newIdx3 = (idx3 < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx3), blkSizeMagic, blkSizeShift, blkSize)) :
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx3), mySizeMagic, mySizeShift));

                const uindex_t offset0 = p0 * lengthSize + feature;
                const uindex_t offset1 = p1 * lengthSize + feature;
                const uindex_t offset2 = p2 * lengthSize + feature;
                const uindex_t offset3 = p3 * lengthSize + feature;

                const uindex_t pos0 = newOffsets[offset0];
                newOffsets[offset0]++;
                const uindex_t pos1 = newOffsets[offset1];
                newOffsets[offset1]++;
                const uindex_t pos2 = newOffsets[offset2];
                newOffsets[offset2]++;
                const uindex_t pos3 = newOffsets[offset3];
                newOffsets[offset3]++;

                newIndices[pos0] = static_cast<T>(newIdx0);
                newIndices[pos1] = static_cast<T>(newIdx1);
                newIndices[pos2] = static_cast<T>(newIdx2);
                newIndices[pos3] = static_cast<T>(newIdx3);

                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos0);
                    unbucketizePermute[i + 1] = static_cast<T>(pos1);
                    unbucketizePermute[i + 2] = static_cast<T>(pos2);
                    unbucketizePermute[i + 3] = static_cast<T>(pos3);
                }

                if (hasWeight) {
                    newWeights[pos0] = weights[i];
                    newWeights[pos1] = weights[i + 1];
                    newWeights[pos2] = weights[i + 2];
                    newWeights[pos3] = weights[i + 3];
                }

                if (bucketizePos) {
                    newPos[pos0] = static_cast<T>(i - rowstart);
                    newPos[pos1] = static_cast<T>(i + 1 - rowstart);
                    newPos[pos2] = static_cast<T>(i + 2 - rowstart);
                    newPos[pos3] = static_cast<T>(i + 3 - rowstart);
                }
            }

            for (; i < rowend; ++i) {
                const uindex_t idx = static_cast<uindex_t>(indices[i]);
                const uindex_t p = (idx < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx), blkSizeMagic, blkSizeShift)) :
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx), mySizeMagic, mySizeShift, mySize));
                const uindex_t newIdx = (idx < blkSizeMulMySize) ?
                    static_cast<uindex_t>(QuickRem(static_cast<uint64_t>(idx), blkSizeMagic, blkSizeShift, blkSize)) :
                    static_cast<uindex_t>(QuickDiv(static_cast<uint64_t>(idx), mySizeMagic, mySizeShift));

                const uindex_t offset = p * lengthSize + feature;
                const uindex_t pos = newOffsets[offset];

                newIndices[pos] = static_cast<T>(newIdx);
                newOffsets[offset]++;

                if (sequence) {
                    unbucketizePermute[i] = static_cast<T>(pos);
                }

                if (hasWeight) {
                    newWeights[pos] = weights[i];
                }

                if (bucketizePos) {
                    newPos[pos] = static_cast<T>(i - rowstart);
                }
            }
        }
    }

    Simt::ThreadBarrier();
}
}  // namespace BlockBucketSizeSparseSimt

#endif  // BLOCK_BUCKETSIZE_SPARSE_FEATURES_KERNEL_H
