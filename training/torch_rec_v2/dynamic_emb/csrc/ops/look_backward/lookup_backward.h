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
__simt_callee__ inline int64_t findIdxBinarySearch(__gm__ const T* const arr, int64_t num, int64_t target)
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

template <typename T, typename ValueT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* uniqueIndices, __gm__ T* inverseIndices,
    __gm__ T* biasedOffsets, int32_t dim, int32_t tableNum, int32_t batchSize, int32_t featureNum, int32_t numKey,
    int32_t combiner, __gm__ bool* kernelStatus)
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
        int64_t src_index = findIdxBinarySearch(biasedOffsets, static_cast<int64_t>(batchSize * featureNum + 1),
                                                static_cast<int64_t>(indices_index));
        if (src_index == -1) {
            *kernelStatus = false;
            return;
        }
        int64_t vec_num =
            static_cast<int64_t>(biasedOffsets[src_index + 1]) - static_cast<int64_t>(biasedOffsets[src_index]);
        float pooling_factor = (combiner == 1) ? static_cast<float>(vec_num) : SUM_POOLING_FACTOR;
        int64_t dst_index = static_cast<int64_t>(inverseIndices[indices_index]);
        float value = static_cast<float>(grad[src_index * dim + indices_dim]);
        AtomicAddGrad<ValueT>(&uniqueBuffer[dst_index * dim + indices_dim], value / pooling_factor);
    }
}

template <typename T, typename ValueT>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer, __gm__ T* uniqueIndices, __gm__ T* inverseIndices,
    __gm__ T* biasedOffsets, int32_t dim, int32_t tableNum, int32_t batchSize, int32_t featureNum, int32_t numKey,
    int32_t combiner, int32_t totalBlocks, int32_t blockStartIdx, int32_t curBlocksCount, __gm__ bool* kernelStatus)
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
            int64_t src_index = findIdxBinarySearch(biasedOffsets, static_cast<int64_t>(batchSize * featureNum + 1),
                                                    static_cast<int64_t>(indices_index));
            if (src_index == -1) {
                *kernelStatus = false;
                return;
            }
            int64_t vec_num =
                static_cast<int64_t>(biasedOffsets[src_index + 1]) - static_cast<int64_t>(biasedOffsets[src_index]);
            float pooling_factor = (combiner == 1) ? static_cast<float>(vec_num) : SUM_POOLING_FACTOR;
            int64_t dst_index = static_cast<int64_t>(inverseIndices[indices_index]);
            float value = static_cast<float>(grad[src_index * dim + indices_dim]);
            AtomicAddGrad<ValueT>(&uniqueBuffer[dst_index * dim + indices_dim], value / pooling_factor);
        }
    }
}
}  // namespace LookupBackwardSimt
