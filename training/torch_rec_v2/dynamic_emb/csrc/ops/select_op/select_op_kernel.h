/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef SELECT_OP_KERNEL_H
#define SELECT_OP_KERNEL_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

namespace DynamicEmbeddingSelectOPSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / WARP_SIZE;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t CACHE_ALIGN = 64;
constexpr int32_t SMALL_DATA_THRESHOLD = 44 * MAX_THREADS_PER_BLOCK;

template <typename T>
__simt_callee__ inline T Min(const T& a, const T& b)
{
    return (a < b) ? a : b;
}

template <typename T>
__simt_callee__ inline T WarpPrefixSum(T val)
{
    int32_t laneId = AscendC::Simt::GetThreadIdx<0>() % WARP_SIZE;
    if (laneId >= WARP_SIZE) {
        return val;
    }

#pragma unroll
    for (uint32_t offset = 1; offset < static_cast<uint32_t>(WARP_SIZE); offset <<= 1) {
        T temp = AscendC::Simt::WarpShflUpSync(val, offset);
        if (laneId >= static_cast<int32_t>(offset)) {
            val += temp;
        }
    }
    return val;
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void FlagPrefixSumSmall(
    __gm__ bool* flags, __gm__ OffsetT* prefix, __gm__ OffsetT* blockSums, __ubuf__ OffsetT* sharedMemory,
    const int32_t totalLength, int32_t coreNum, int32_t stride)
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

    OffsetT currentVal = flags[globalTid] ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
    OffsetT threadSum = currentVal;
    OffsetT warpPrefixSum = WarpPrefixSum(threadSum);

    int32_t elementsThisBlock = threadNumPerCore;
    int32_t activeWarpCount = (elementsThisBlock + WARP_SIZE - 1) / WARP_SIZE;
    int32_t elementsInThisWarp = (warpId < activeWarpCount - 1) ? WARP_SIZE : (elementsThisBlock - warpId * WARP_SIZE);

    if (laneId == elementsInThisWarp - 1 && warpId < activeWarpCount && warpId < MAX_WARPS) {
        sharedMemory[warpId] = warpPrefixSum;
    }
    AscendC::Simt::ThreadBarrier();

    if (tid < activeWarpCount && tid < MAX_WARPS) {
        OffsetT warpSumValue = sharedMemory[tid];
        OffsetT warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedMemory[tid] = warpSumPrefix;
    }
    AscendC::Simt::ThreadBarrier();

    OffsetT blockOffset = static_cast<OffsetT>(0);
    if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MAX_WARPS) {
        blockOffset = sharedMemory[warpId - 1];
    }
    prefix[globalTid] = blockOffset + warpPrefixSum - currentVal;

    if (tid == 0) {
        OffsetT coreSum =
            (activeWarpCount > 0 && activeWarpCount - 1 < MAX_WARPS) ? sharedMemory[activeWarpCount - 1] : 0;
        if (coreId * stride < coreNum * stride) {
            blockSums[coreId * stride] = coreSum;
        }
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void FlagPrefixSumSmallUpdate(
    __gm__ OffsetT* prefix, __gm__ OffsetT* blockSums, const int32_t totalLength, int32_t coreNum, int32_t stride)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t globalTid = coreId * threadNumPerCore + tid;

    if (globalTid >= totalLength) {
        return;
    }

    OffsetT coreOffset = static_cast<OffsetT>(0);
    int32_t totalBlocks = (totalLength + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    for (int32_t c = 0; c < coreId && c < totalBlocks; ++c) {
        coreOffset += blockSums[c * stride];
    }
    prefix[globalTid] += coreOffset;
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void FlagPrefixSumLarge(
    __gm__ bool* flags, __gm__ OffsetT* prefix, __gm__ OffsetT* blockSums, __ubuf__ OffsetT* sharedMemory,
    const int32_t totalLength, int32_t blockStartIdx, int32_t curBlocksCount, int32_t stride)
{
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

        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadElementOffset;
        int32_t elementsForThread = elementsThisBlock - threadElementOffset;
        if (elementsForThread < 0) {
            elementsForThread = 0;
        }
        if (elementsForThread > MAX_ELEMENTS_PER_THREAD) {
            elementsForThread = MAX_ELEMENTS_PER_THREAD;
        }
        if (elementsForThread <= 0) {
            continue;
        }

        OffsetT threadSum = 0;
        OffsetT prefixSums[MAX_ELEMENTS_PER_THREAD] = {0};
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t elemIdx = threadElementBase + i;
                if (elemIdx >= 0 && elemIdx < totalLength) {
                    OffsetT value = flags[elemIdx] ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
                    prefixSums[i] = threadSum;
                    threadSum += value;
                }
            }
        }

        OffsetT warpPrefixSum = WarpPrefixSum(threadSum);

        int32_t activeThreads = (elementsThisBlock + MAX_ELEMENTS_PER_THREAD - 1) / MAX_ELEMENTS_PER_THREAD;
        int32_t activeWarpCount = (activeThreads + WARP_SIZE - 1) / WARP_SIZE;
        int32_t threadsInWarp = activeThreads - warpId * WARP_SIZE;
        if (threadsInWarp < 0) {
            threadsInWarp = 0;
        }
        if (threadsInWarp > WARP_SIZE) {
            threadsInWarp = WARP_SIZE;
        }

        if (threadsInWarp > 0 && warpId < activeWarpCount && warpId < MAX_WARPS && laneId == threadsInWarp - 1) {
            sharedMemory[warpId] = warpPrefixSum;
        }
        AscendC::Simt::ThreadBarrier();

        if (threadIdxInt < activeWarpCount && threadIdxInt < MAX_WARPS) {
            OffsetT warpSumValue = sharedMemory[threadIdxInt];
            OffsetT warpSumPrefix = WarpPrefixSum(warpSumValue);
            sharedMemory[threadIdxInt] = warpSumPrefix;
        }
        AscendC::Simt::ThreadBarrier();

        OffsetT blockOffset = 0;
        if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MAX_WARPS) {
            blockOffset = sharedMemory[warpId - 1];
        }
        OffsetT finalOffset = blockOffset + warpPrefixSum - threadSum;

#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t globalIdx = threadElementBase + i;
                if (globalIdx >= 0 && globalIdx < totalLength) {
                    prefix[globalIdx] = finalOffset + prefixSums[i];
                }
            }
        }

        if (threadIdxInt == 0) {
            OffsetT blockSum =
                (activeWarpCount > 0 && activeWarpCount - 1 < MAX_WARPS) ? sharedMemory[activeWarpCount - 1] : 0;
            blockSums[globalBlockIdx * stride] = blockSum;
        }
        AscendC::Simt::ThreadBarrier();
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void FlagPrefixSumLargeUpdate(
    __gm__ OffsetT* prefix, __gm__ OffsetT* blockSums, const int32_t totalLength, int32_t blockStartIdx,
    int32_t curBlocksCount, int32_t stride)
{
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadIdxInt = static_cast<int32_t>(tid);
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = threadNumPerCore * MAX_ELEMENTS_PER_THREAD;
    int32_t threadElementOffset = threadIdxInt * MAX_ELEMENTS_PER_THREAD;

    OffsetT blockPrefix = 0;
    for (int32_t i = 0; i < blockStartIdx; ++i) {
        blockPrefix += blockSums[i * stride];
    }

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= totalLength) {
            break;
        }

        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadElementOffset;
        int32_t elementsForThread = elementsThisBlock - threadElementOffset;
        if (elementsForThread < 0) {
            elementsForThread = 0;
        }
        if (elementsForThread > MAX_ELEMENTS_PER_THREAD) {
            elementsForThread = MAX_ELEMENTS_PER_THREAD;
        }
        if (elementsForThread <= 0) {
            continue;
        }

#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t globalIdx = threadElementBase + i;
                if (globalIdx >= 0 && globalIdx < totalLength) {
                    prefix[globalIdx] += blockPrefix;
                }
            }
        }

        blockPrefix += blockSums[globalBlockIdx * stride];
        AscendC::Simt::ThreadBarrier();
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void WriteNumSelectedVF(__gm__ OffsetT* prefix, __gm__ bool* flags,
                                                                      __gm__ OffsetT* numSelected,
                                                                      const int32_t totalLength)
{
    if (AscendC::Simt::GetThreadIdx<0>() != 0) {
        return;
    }
    if (totalLength > 0) {
        int32_t last = totalLength - 1;
        numSelected[0] = prefix[last] + (flags[last] ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0));
    } else {
        numSelected[0] = static_cast<OffsetT>(0);
    }
}

template <typename KeyT, typename OffsetT, bool SelectIndex>
__simt_callee__ inline void ScatterElementAt(__gm__ bool* flags, __gm__ KeyT* inputs, __gm__ KeyT* outputs,
                                             __gm__ OffsetT* prefix, int32_t idx)
{
    if (flags[idx]) {
        OffsetT outPos = prefix[idx];
        if constexpr (SelectIndex) {
            outputs[outPos] = static_cast<KeyT>(idx);
        } else {
            outputs[outPos] = inputs[idx];
        }
    }
}

template <typename KeyT, typename OffsetT, bool SelectIndex>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SelectScatterSmall(
    __gm__ bool* flags, __gm__ KeyT* inputs, __gm__ KeyT* outputs, __gm__ OffsetT* prefix, const int32_t totalLength)
{
    int32_t coreId = AscendC::Simt::GetBlockIdx();
    int32_t tid = AscendC::Simt::GetThreadIdx<0>();
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t globalTid = coreId * threadNumPerCore + tid;

    if (globalTid < totalLength) {
        ScatterElementAt<KeyT, OffsetT, SelectIndex>(flags, inputs, outputs, prefix, globalTid);
    }
}

template <typename KeyT, typename OffsetT, bool SelectIndex>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SelectScatterLarge(
    __gm__ bool* flags, __gm__ KeyT* inputs, __gm__ KeyT* outputs, __gm__ OffsetT* prefix, const int32_t totalLength,
    int32_t blockStartIdx, int32_t curBlocksCount)
{
    int32_t threadIdxInt = static_cast<int32_t>(AscendC::Simt::GetThreadIdx<0>());
    int32_t threadNumPerCore = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = threadNumPerCore * MAX_ELEMENTS_PER_THREAD;
    int32_t threadElementOffset = threadIdxInt * MAX_ELEMENTS_PER_THREAD;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= totalLength) {
            break;
        }

        int32_t elementsRemaining = totalLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }

        int32_t threadElementBase = blockBase + threadElementOffset;
        int32_t elementsForThread = elementsThisBlock - threadElementOffset;
        if (elementsForThread < 0) {
            elementsForThread = 0;
        }
        if (elementsForThread > MAX_ELEMENTS_PER_THREAD) {
            elementsForThread = MAX_ELEMENTS_PER_THREAD;
        }
        if (elementsForThread <= 0) {
            continue;
        }

#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i < elementsForThread) {
                int32_t idx = threadElementBase + i;
                if (idx >= 0 && idx < totalLength) {
                    ScatterElementAt<KeyT, OffsetT, SelectIndex>(flags, inputs, outputs, prefix, idx);
                }
            }
        }
    }
}

}  // namespace DynamicEmbeddingSelectOPSimt

#endif  // SELECT_OP_KERNEL_H
