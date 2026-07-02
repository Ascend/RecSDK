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

#include <simt_api/common_functions.h>
#include "kernel_operator.h"

using namespace AscendC;

namespace LookupBackwardV2Simt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int32_t UNROLL_FACTOR = 4;
constexpr float SUM_POOLING_FACTOR = 1.0f;

template <typename ValueT>
__simt_callee__ inline void AtomicAddGrad(__gm__ ValueT* addr, float contrib)
{
    AscendC::Simt::AtomicAdd<ValueT>(addr, static_cast<ValueT>(contrib));
}

__simt_callee__ inline void AtomicAddGradFloat2(__gm__ float2* addr, float2 contrib)
{
    AscendC::Simt::AtomicAdd<float>(reinterpret_cast<__gm__ float*>(addr), contrib.x);
    AscendC::Simt::AtomicAdd<float>(reinterpret_cast<__gm__ float*>(addr) + 1, contrib.y);
}

template <typename ValueT>
__simt_callee__ inline void FlushScalarPending(__gm__ ValueT* uniqueBuffer, int32_t launchDim, int32_t indicesDim,
                                               int64_t& lastDst, float& pending)
{
    if (lastDst >= 0) {
        AtomicAddGrad<ValueT>(&uniqueBuffer[lastDst * launchDim + indicesDim], pending);
        lastDst = -1;
        pending = 0.0f;
    }
}

__simt_callee__ inline void FlushFloat2Pending(__gm__ float2* uniqueBuffer, int32_t launchDim, int32_t indicesDim,
                                               int64_t& lastDst, float2& pending)
{
    if (lastDst >= 0) {
        AtomicAddGradFloat2(&uniqueBuffer[lastDst * launchDim + indicesDim], pending);
        lastDst = -1;
        pending = {0.0f, 0.0f};
    }
}

template <typename ValueT>
__simt_callee__ inline void AccumulateScalarPending(int64_t dst, float contrib, __gm__ ValueT* uniqueBuffer,
                                                    int32_t launchDim, int32_t indicesDim, int64_t& lastDst,
                                                    float& pending)
{
    if (lastDst == dst) {
        pending += contrib;
        return;
    }
    FlushScalarPending(uniqueBuffer, launchDim, indicesDim, lastDst, pending);
    lastDst = dst;
    pending = contrib;
}

__simt_callee__ inline void AccumulateFloat2Pending(int64_t dst, float2 contrib, __gm__ float2* uniqueBuffer,
                                                    int32_t launchDim, int32_t indicesDim, int64_t& lastDst,
                                                    float2& pending)
{
    if (lastDst == dst) {
        pending.x += contrib.x;
        pending.y += contrib.y;
        return;
    }
    FlushFloat2Pending(uniqueBuffer, launchDim, indicesDim, lastDst, pending);
    lastDst = dst;
    pending = contrib;
}

template <typename T, bool IsMean>
__simt_callee__ inline void FillMeanPoolingFactor(int64_t src_index, __gm__ T* biasedOffsets, float& pooling_factor)
{
    if constexpr (IsMean) {
        int64_t vec_num =
            static_cast<int64_t>(biasedOffsets[src_index + 1]) - static_cast<int64_t>(biasedOffsets[src_index]);
        pooling_factor = static_cast<float>(vec_num);
    } else {
        pooling_factor = SUM_POOLING_FACTOR;
    }
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2>
__simt_callee__ inline void ScatterSlotDim(__gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices,
                                           __gm__ T* biasedOffsets, int32_t launchDim, int32_t slot, int32_t indicesDim)
{
    int64_t keyStart = static_cast<int64_t>(biasedOffsets[slot]);
    int64_t keyEnd = static_cast<int64_t>(biasedOffsets[slot + 1]);
    if (keyEnd <= keyStart) {
        return;
    }
    float poolingFactor = SUM_POOLING_FACTOR;
    FillMeanPoolingFactor<T, IsMean>(slot, biasedOffsets, poolingFactor);
    if (poolingFactor <= 0.0f) {
        return;
    }

    if constexpr (IsFloat2) {
        float2 gradVal = grad[slot * launchDim + indicesDim];
        float inv = 1.0f / poolingFactor;
        float2 scaledGrad = {gradVal.x * inv, gradVal.y * inv};
        int64_t lastDst = -1;
        float2 pending = {0.0f, 0.0f};
        auto* uniqueBufferFloat2 = reinterpret_cast<__gm__ float2*>(uniqueBuffer);

        int64_t k = keyStart;
        for (; k + UNROLL_FACTOR <= keyEnd; k += UNROLL_FACTOR) {
#pragma unroll
            for (int32_t u = 0; u < UNROLL_FACTOR; ++u) {
                int64_t dst = static_cast<int64_t>(inverseIndices[k + u]);
                AccumulateFloat2Pending(dst, scaledGrad, uniqueBufferFloat2, launchDim, indicesDim, lastDst, pending);
            }
        }
        for (; k < keyEnd; ++k) {
            int64_t dst = static_cast<int64_t>(inverseIndices[k]);
            AccumulateFloat2Pending(dst, scaledGrad, uniqueBufferFloat2, launchDim, indicesDim, lastDst, pending);
        }
        FlushFloat2Pending(uniqueBufferFloat2, launchDim, indicesDim, lastDst, pending);
    } else {
        const float scaledGrad = static_cast<float>(grad[slot * launchDim + indicesDim]) / poolingFactor;
        int64_t lastDst = -1;
        float pending = 0.0f;

        int64_t k = keyStart;
        for (; k + UNROLL_FACTOR <= keyEnd; k += UNROLL_FACTOR) {
#pragma unroll
            for (int32_t u = 0; u < UNROLL_FACTOR; ++u) {
                int64_t dst = static_cast<int64_t>(inverseIndices[k + u]);
                AccumulateScalarPending(dst, scaledGrad, uniqueBuffer, launchDim, indicesDim, lastDst, pending);
            }
        }
        for (; k < keyEnd; ++k) {
            int64_t dst = static_cast<int64_t>(inverseIndices[k]);
            AccumulateScalarPending(dst, scaledGrad, uniqueBuffer, launchDim, indicesDim, lastDst, pending);
        }
        FlushScalarPending(uniqueBuffer, launchDim, indicesDim, lastDst, pending);
    }
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices, __gm__ T* biasedOffsets,
    int32_t launchDim, int32_t numSlots)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t blockBase = blockIdx * blockElementCapacity;
    int32_t inputLength = numSlots * launchDim;
    if (blockBase >= inputLength) {
        return;
    }

    int32_t elementsRemaining = inputLength - blockBase;
    int32_t elementsThisBlock = (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
    int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;
    if (elementsThisBlock <= 0 || threadElementBase >= inputLength) {
        return;
    }

    int32_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
    int32_t elementsForThread =
        (elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD : elementsThisBlockRemaining;

#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
        if (i >= elementsForThread) {
            return;
        }
        int32_t globalIdx = threadElementBase + i;
        if (globalIdx >= inputLength) {
            return;
        }

        int32_t slotIdx = globalIdx / launchDim;
        int32_t indicesDim = globalIdx % launchDim;
        ScatterSlotDim<T, ValueT, IsMean, IsFloat2>(grad, uniqueBuffer, inverseIndices, biasedOffsets, launchDim,
                                                    slotIdx, indicesDim);
    }
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices, __gm__ T* biasedOffsets,
    int32_t launchDim, int32_t numSlots, int32_t totalBlocks, int32_t blockStartIdx, int32_t curBlocksCount)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t inputLength = numSlots * launchDim;
    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (globalBlockIdx >= totalBlocks || blockBase >= inputLength) {
            break;
        }
        int32_t elementsRemaining = inputLength - blockBase;
        int32_t elementsThisBlock =
            (elementsRemaining < blockElementCapacity) ? elementsRemaining : blockElementCapacity;
        if (elementsThisBlock <= 0) {
            continue;
        }
        int32_t threadElementBase = blockBase + threadIdx * MAX_ELEMENTS_PER_THREAD;
        if (threadElementBase >= inputLength) {
            break;
        }
        int32_t elementsThisBlockRemaining = elementsThisBlock - threadIdx * MAX_ELEMENTS_PER_THREAD;
        int32_t elementsForThread = (elementsThisBlockRemaining > MAX_ELEMENTS_PER_THREAD) ? MAX_ELEMENTS_PER_THREAD
                                                                                           : elementsThisBlockRemaining;
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i >= elementsForThread) {
                break;
            }
            int32_t globalIdx = threadElementBase + i;
            if (globalIdx >= inputLength) {
                break;
            }

            int32_t slotIdx = globalIdx / launchDim;
            int32_t indicesDim = globalIdx % launchDim;
            ScatterSlotDim<T, ValueT, IsMean, IsFloat2>(grad, uniqueBuffer, inverseIndices, biasedOffsets, launchDim,
                                                        slotIdx, indicesDim);
        }
    }
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2 = false>
__aicore__ inline void LaunchBackwardCompute(__gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices,
                                             __gm__ T* biasedOffsets, int32_t launchDim, int32_t numSlots,
                                             int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks,
                                             bool isSmall, int32_t coreId)
{
    if (isSmall) {
        asc_vf_call<SimtSmallDataCompute<T, ValueT, IsMean, IsFloat2>>(
            dim3{MAX_THREADS_PER_BLOCK, 1, 1}, grad, uniqueBuffer, inverseIndices, biasedOffsets, launchDim, numSlots);
        return;
    }
    int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
    int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);
    asc_vf_call<SimtLargeDataCompute<T, ValueT, IsMean, IsFloat2>>(
        dim3{MAX_THREADS_PER_BLOCK, 1, 1}, grad, uniqueBuffer, inverseIndices, biasedOffsets, launchDim, numSlots,
        totalBlocks, blockStartIdx, curBlocksCount);
}
}  // namespace LookupBackwardV2Simt
