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

#ifndef MASKSELECT_SIMT_H
#define MASKSELECT_SIMT_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "maskselect_common.h"

using namespace AscendC;

namespace MaskSelectSimt {

template <typename T>
__simt_callee__ inline T WarpPrefixSum(T val)
{
    int32_t laneId = threadIdx.x % MASKSELECT_WARP_SIZE;
#pragma unroll
    for (int32_t offset = 1; offset < MASKSELECT_WARP_SIZE; offset <<= 1) {
        T temp = asc_shfl_up(val, offset);
        if (laneId >= offset) {
            val += temp;
        }
    }
    return val;
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MASKSELECT_MAX_THREADS) inline void SmallFlagPrefixCompute(
    __ubuf__ uint8_t* flags, __ubuf__ OffsetT* prefix, __gm__ OffsetT* blockSums, __ubuf__ OffsetT* sharedUb,
    int32_t elementsThisBlock, int32_t blockIdx)
{
    const int32_t warpId = threadIdx.x / MASKSELECT_WARP_SIZE;
    const int32_t laneId = threadIdx.x % MASKSELECT_WARP_SIZE;
    int32_t activeWarpCount = (elementsThisBlock + MASKSELECT_WARP_SIZE - 1) / MASKSELECT_WARP_SIZE;

    OffsetT currentVal = (threadIdx.x < elementsThisBlock && flags[threadIdx.x] != 0) ? static_cast<OffsetT>(1)
                                                                                      : static_cast<OffsetT>(0);
    OffsetT warpPrefixSum = WarpPrefixSum(currentVal);

    int32_t elementsInThisWarp =
        (warpId < activeWarpCount - 1) ? MASKSELECT_WARP_SIZE : (elementsThisBlock - warpId * MASKSELECT_WARP_SIZE);
    elementsInThisWarp = (warpId >= activeWarpCount) ? 0 : elementsInThisWarp;
    if (laneId == elementsInThisWarp - 1 && warpId < activeWarpCount && warpId < MASKSELECT_MAX_WARPS) {
        sharedUb[warpId] = warpPrefixSum;
    }
    asc_syncthreads();

    if (threadIdx.x < activeWarpCount && threadIdx.x < MASKSELECT_MAX_WARPS) {
        OffsetT warpSumValue = sharedUb[threadIdx.x];
        OffsetT warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedUb[threadIdx.x] = warpSumPrefix;
    }
    asc_syncthreads();

    OffsetT warpExclusive = static_cast<OffsetT>(0);
    if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MASKSELECT_MAX_WARPS) {
        warpExclusive = sharedUb[warpId - 1];
    }
    OffsetT exclusivePrefix = warpExclusive + warpPrefixSum - currentVal;

    if (threadIdx.x < elementsThisBlock) {
        prefix[threadIdx.x] = exclusivePrefix;
    }

    if (threadIdx.x == 0) {
        OffsetT blockSum =
            (activeWarpCount > 0 && activeWarpCount - 1 < MASKSELECT_MAX_WARPS) ? sharedUb[activeWarpCount - 1] : 0;
        blockSums[blockIdx] = blockSum;
    }
}

template <typename OffsetT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MASKSELECT_MAX_THREADS) inline void LargeFlagPrefixCompute(
    __ubuf__ uint8_t* flags, __ubuf__ OffsetT* prefix, __gm__ OffsetT* blockSums, __ubuf__ OffsetT* sharedUb,
    int32_t elementsThisBlock, int32_t blockIdx)
{
    const int32_t warpId = threadIdx.x / MASKSELECT_WARP_SIZE;
    const int32_t laneId = threadIdx.x % MASKSELECT_WARP_SIZE;
    int32_t threadElementBase = threadIdx.x * MASKSELECT_MAX_ELEMENTS_PER_THREAD;
    const bool isFullBatch = (threadElementBase + MASKSELECT_MAX_ELEMENTS_PER_THREAD - 1) < elementsThisBlock;

    OffsetT threadSum = 0;
    OffsetT prefixSums[MASKSELECT_MAX_ELEMENTS_PER_THREAD] = {0};
    if (isFullBatch) {
        {
            int32_t threadIdx = threadElementBase;
            OffsetT value = (flags[threadIdx] != 0) ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
            prefixSums[0] = threadSum;
            threadSum += value;
        }
        {
            int32_t threadIdx = threadElementBase + 1;
            OffsetT value = (flags[threadIdx] != 0) ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
            prefixSums[1] = threadSum;
            threadSum += value;
        }
        {
            int32_t threadIdx = threadElementBase + 2;
            OffsetT value = (flags[threadIdx] != 0) ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
            prefixSums[2] = threadSum;
            threadSum += value;
        }
        {
            int32_t threadIdx = threadElementBase + 3;
            OffsetT value = (flags[threadIdx] != 0) ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
            prefixSums[3] = threadSum;
            threadSum += value;
        }
    } else {
#pragma unroll
        for (int32_t i = 0; i < MASKSELECT_MAX_ELEMENTS_PER_THREAD; ++i) {
            int32_t threadIdx = threadElementBase + i;
            if (threadIdx >= elementsThisBlock) {
                break;
            }
            OffsetT value = (flags[threadIdx] != 0) ? static_cast<OffsetT>(1) : static_cast<OffsetT>(0);
            prefixSums[i] = threadSum;
            threadSum += value;
        }
    }

    int32_t activeThreads =
        (elementsThisBlock + MASKSELECT_MAX_ELEMENTS_PER_THREAD - 1) / MASKSELECT_MAX_ELEMENTS_PER_THREAD;
    int32_t activeWarpCount = (activeThreads + MASKSELECT_WARP_SIZE - 1) / MASKSELECT_WARP_SIZE;
    int32_t threadsInWarp = Std::max(0, Std::min(activeThreads - warpId * MASKSELECT_WARP_SIZE, MASKSELECT_WARP_SIZE));

    OffsetT warpPrefixSum = WarpPrefixSum(threadSum);

    if (threadsInWarp > 0 && warpId < activeWarpCount && warpId < MASKSELECT_MAX_WARPS && laneId == threadsInWarp - 1) {
        sharedUb[warpId] = warpPrefixSum;
    }
    asc_syncthreads();

    if (threadIdx.x < activeWarpCount && threadIdx.x < MASKSELECT_MAX_WARPS) {
        OffsetT warpSumValue = sharedUb[threadIdx.x];
        OffsetT warpSumPrefix = WarpPrefixSum(warpSumValue);
        sharedUb[threadIdx.x] = warpSumPrefix;
    }
    asc_syncthreads();

    OffsetT warpExclusive = static_cast<OffsetT>(0);
    if (warpId > 0 && warpId < activeWarpCount && (warpId - 1) < MASKSELECT_MAX_WARPS) {
        warpExclusive = sharedUb[warpId - 1];
    }
    OffsetT blockOffset = warpExclusive + warpPrefixSum - threadSum;

    if (isFullBatch) {
        prefix[threadElementBase] = blockOffset + prefixSums[0];
        prefix[threadElementBase + 1] = blockOffset + prefixSums[1];
        prefix[threadElementBase + 2] = blockOffset + prefixSums[2];
        prefix[threadElementBase + 3] = blockOffset + prefixSums[3];
    } else {
#pragma unroll
        for (int32_t i = 0; i < MASKSELECT_MAX_ELEMENTS_PER_THREAD; ++i) {
            int32_t threadIdx = threadElementBase + i;
            if (threadIdx >= elementsThisBlock) {
                break;
            }
            prefix[threadIdx] = blockOffset + prefixSums[i];
        }
    }

    asc_syncthreads();
    if (threadIdx.x == 0) {
        OffsetT blockSum =
            (activeWarpCount > 0 && activeWarpCount - 1 < MASKSELECT_MAX_WARPS) ? sharedUb[activeWarpCount - 1] : 0;
        blockSums[blockIdx] = blockSum;
    }
}

template <typename KeyT, typename OffsetT, bool SelectIndex>
__simt_vf__ __aicore__ LAUNCH_BOUND(MASKSELECT_MAX_THREADS) inline void SmallScatterCompute(
    __ubuf__ uint8_t* flags, __ubuf__ KeyT* keys, __ubuf__ OffsetT* prefix, __gm__ KeyT* outputs,
    int32_t elementsThisBlock, int32_t blockBase)
{
    if (threadIdx.x < elementsThisBlock && flags[threadIdx.x] != 0) {
        OffsetT outPos = prefix[threadIdx.x];
        if constexpr (SelectIndex) {
            outputs[outPos] = static_cast<KeyT>(blockBase + threadIdx.x);
        } else {
            outputs[outPos] = keys[threadIdx.x];
        }
    }
}

template <typename KeyT, typename OffsetT, bool SelectIndex>
__simt_vf__ __aicore__ LAUNCH_BOUND(MASKSELECT_MAX_THREADS) inline void LargeScatterCompute(
    __ubuf__ uint8_t* flags, __ubuf__ KeyT* keys, __ubuf__ OffsetT* prefix, __gm__ KeyT* outputs,
    int32_t elementsThisBlock, int32_t blockBase)
{
    int32_t threadElementBase = threadIdx.x * MASKSELECT_MAX_ELEMENTS_PER_THREAD;
    const bool isFullBatch = (threadElementBase + MASKSELECT_MAX_ELEMENTS_PER_THREAD - 1) < elementsThisBlock;

    if (isFullBatch) {
        {
            int32_t threadIdx = threadElementBase;
            if (flags[threadIdx] != 0) {
                OffsetT outPos = prefix[threadIdx];
                if constexpr (SelectIndex) {
                    outputs[outPos] = static_cast<KeyT>(blockBase + threadIdx);
                } else {
                    outputs[outPos] = keys[threadIdx];
                }
            }
        }
        {
            int32_t threadIdx = threadElementBase + 1;
            if (flags[threadIdx] != 0) {
                OffsetT outPos = prefix[threadIdx];
                if constexpr (SelectIndex) {
                    outputs[outPos] = static_cast<KeyT>(blockBase + threadIdx);
                } else {
                    outputs[outPos] = keys[threadIdx];
                }
            }
        }
        {
            int32_t threadIdx = threadElementBase + 2;
            if (flags[threadIdx] != 0) {
                OffsetT outPos = prefix[threadIdx];
                if constexpr (SelectIndex) {
                    outputs[outPos] = static_cast<KeyT>(blockBase + threadIdx);
                } else {
                    outputs[outPos] = keys[threadIdx];
                }
            }
        }
        {
            int32_t threadIdx = threadElementBase + 3;
            if (flags[threadIdx] != 0) {
                OffsetT outPos = prefix[threadIdx];
                if constexpr (SelectIndex) {
                    outputs[outPos] = static_cast<KeyT>(blockBase + threadIdx);
                } else {
                    outputs[outPos] = keys[threadIdx];
                }
            }
        }
    } else {
#pragma unroll
        for (int32_t i = 0; i < MASKSELECT_MAX_ELEMENTS_PER_THREAD; ++i) {
            int32_t threadIdx = threadElementBase + i;
            if (threadIdx >= elementsThisBlock) {
                break;
            }
            if (flags[threadIdx] != 0) {
                OffsetT outPos = prefix[threadIdx];
                if constexpr (SelectIndex) {
                    outputs[outPos] = static_cast<KeyT>(blockBase + threadIdx);
                } else {
                    outputs[outPos] = keys[threadIdx];
                }
            }
        }
    }
}

}  // namespace MaskSelectSimt

#endif  // MASKSELECT_SIMT_H
