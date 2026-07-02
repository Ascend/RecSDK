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
namespace LookupBackwardSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr float SUM_POOLING_FACTOR = 1.0f;

template <typename T>
__simt_callee__ inline int64_t FindIdxBinarySearch(__gm__ const T* const arr, int64_t num, int64_t target)
{
    if (num <= 0) {
        return -1;
    }
    int64_t start = 0;
    int64_t end = num;
    while (start < end) {
        int64_t mid = start + (end - start) / 2;
        T value = arr[mid];
        if (value <= target) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    return (start == num && arr[num - 1] != target) ? num : (start - 1);
}

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

template <typename ValueT, bool IsFloat2>
__simt_callee__ inline void ScatterGradContribution(__gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, int64_t src_index,
                                                    int64_t dst_index, int32_t dim, int32_t indices_dim,
                                                    float pooling_factor)
{
    if (pooling_factor <= 0.0f) {
        return;
    }
    if constexpr (IsFloat2) {
        float2 gradVal = grad[src_index * dim + indices_dim];
        float inv = 1.0f / pooling_factor;
        float2 value = {gradVal.x * inv, gradVal.y * inv};
        AtomicAddGradFloat2(&uniqueBuffer[dst_index * dim + indices_dim], value);
    } else {
        float value = static_cast<float>(grad[src_index * dim + indices_dim]) / pooling_factor;
        AtomicAddGrad<ValueT>(&uniqueBuffer[dst_index * dim + indices_dim], value);
    }
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

template <typename T, bool IsMean>
__simt_callee__ inline void ResolveGradSource(int32_t indices_index, __gm__ T* biasedOffsets, int32_t numSamples,
                                              int64_t& src_index, float& pooling_factor)
{
    const int64_t num_offsets = static_cast<int64_t>(numSamples) + 1;
    src_index = FindIdxBinarySearch(biasedOffsets, num_offsets, static_cast<int64_t>(indices_index));
    if (src_index < 0 || src_index >= static_cast<int64_t>(numSamples)) {
        src_index = -1;
        pooling_factor = 0.0f;
        return;
    }
    FillMeanPoolingFactor<T, IsMean>(src_index, biasedOffsets, pooling_factor);
}

template <typename T, bool IsMean>
__simt_callee__ inline void ResolveGradSourceCached(int32_t indices_index, int32_t& cached_indices_index,
                                                    int64_t& cached_src_index, float& cached_pooling_factor,
                                                    __gm__ T* biasedOffsets, int32_t numSamples, int64_t& src_index,
                                                    float& pooling_factor)
{
    if (indices_index != cached_indices_index) {
        ResolveGradSource<T, IsMean>(indices_index, biasedOffsets, numSamples, cached_src_index, cached_pooling_factor);
        cached_indices_index = indices_index;
    }
    src_index = cached_src_index;
    pooling_factor = cached_pooling_factor;
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices, __gm__ T* biasedOffsets, int32_t dim,
    int32_t numKey, int32_t numSamples)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t blockBase = blockIdx * blockElementCapacity;
    int32_t inputLength = numKey * dim;
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

    int32_t cached_indices_index = -1;
    int64_t cached_src_index = 0;
    float cached_pooling_factor = SUM_POOLING_FACTOR;

#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
        if (i >= elementsForThread) {
            return;
        }
        int32_t globalIdx = threadElementBase + i;
        if (globalIdx >= inputLength) {
            return;
        }

        int32_t indices_index = globalIdx / dim;
        int32_t indices_dim = globalIdx % dim;
        int64_t src_index = 0;
        float pooling_factor = SUM_POOLING_FACTOR;
        ResolveGradSourceCached<T, IsMean>(indices_index, cached_indices_index, cached_src_index, cached_pooling_factor,
                                           biasedOffsets, numSamples, src_index, pooling_factor);
        if (src_index < 0) {
            return;
        }
        int64_t dst_index = static_cast<int64_t>(inverseIndices[indices_index]);
        ScatterGradContribution<ValueT, IsFloat2>(grad, uniqueBuffer, src_index, dst_index, dim, indices_dim,
                                                  pooling_factor);
    }
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices, __gm__ T* biasedOffsets, int32_t dim,
    int32_t numKey, int32_t numSamples, int32_t totalBlocks, int32_t blockStartIdx, int32_t curBlocksCount)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t inputLength = numKey * dim;
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
        int32_t cached_indices_index = -1;
        int64_t cached_src_index = 0;
        float cached_pooling_factor = SUM_POOLING_FACTOR;
#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; ++i) {
            if (i >= elementsForThread) {
                break;
            }
            int32_t globalIdx = threadElementBase + i;
            if (globalIdx >= inputLength) {
                break;
            }

            int32_t indices_index = globalIdx / dim;
            int32_t indices_dim = globalIdx % dim;
            int64_t src_index = 0;
            float pooling_factor = SUM_POOLING_FACTOR;
            ResolveGradSourceCached<T, IsMean>(indices_index, cached_indices_index, cached_src_index,
                                               cached_pooling_factor, biasedOffsets, numSamples, src_index,
                                               pooling_factor);
            if (src_index < 0) {
                return;
            }
            int64_t dst_index = static_cast<int64_t>(inverseIndices[indices_index]);
            ScatterGradContribution<ValueT, IsFloat2>(grad, uniqueBuffer, src_index, dst_index, dim, indices_dim,
                                                      pooling_factor);
        }
    }
}

template <typename T, typename ValueT, bool IsMean, bool IsFloat2 = false>
__aicore__ inline void LaunchBackwardCompute(__gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* inverseIndices,
                                             __gm__ T* biasedOffsets, int32_t dim, int32_t numKey, int32_t numSamples,
                                             int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks,
                                             bool isSmall, int32_t coreId)
{
    if (isSmall) {
        AscendC::Simt::VF_CALL<SimtSmallDataCompute<T, ValueT, IsMean, IsFloat2>>(
            AscendC::Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, grad, uniqueBuffer, inverseIndices, biasedOffsets, dim,
            numKey, numSamples);
        return;
    }
    int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
    int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);
    AscendC::Simt::VF_CALL<SimtLargeDataCompute<T, ValueT, IsMean, IsFloat2>>(
        AscendC::Simt::Dim3{MAX_THREADS_PER_BLOCK, 1, 1}, grad, uniqueBuffer, inverseIndices, biasedOffsets, dim,
        numKey, numSamples, totalBlocks, blockStartIdx, curBlocksCount);
}
}  // namespace LookupBackwardSimt
