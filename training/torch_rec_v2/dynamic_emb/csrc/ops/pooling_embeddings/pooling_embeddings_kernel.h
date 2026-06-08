/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

namespace PoolingEmbeddingsSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t UNROLL_FACTOR = 4;

// SIMT VF函数 - 小数据模式
template <typename T1, typename T2, typename T3, bool IsFloat2 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallDataCompute(
    __gm__ T1* src, __gm__ T2* dst, __gm__ T3* offset, __gm__ T3* inverse, int32_t combiner, int32_t totalDims,
    int32_t accumDims, int32_t evSize, int32_t evSizeVec, int32_t numVec, int32_t batchSize, int32_t outLen)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();

    int32_t blockBase = blockIdx * blockThreadNum;

    if (blockBase >= outLen) {
        return;
    }

    int32_t elementsRemaining = outLen - blockBase;
    // block计算(blockBase, blockBase + elementsThisBlock)
    int32_t elementsThisBlock = (elementsRemaining < blockThreadNum) ? elementsRemaining : blockThreadNum;
    // 该运行线程
    int32_t threadElementBase = blockBase + threadIdx;

    if (threadElementBase >= outLen) {
        return;
    }

    int32_t indicesIndex = threadElementBase / evSizeVec;
    int32_t indicesDimVec = threadElementBase % evSizeVec;
    int32_t start = offset[indicesIndex] - offset[0];
    int32_t vectorNum = offset[indicesIndex + 1] - offset[indicesIndex];
    int32_t dstRowIndex = indicesIndex % batchSize;
    int32_t dstColIndex = indicesIndex / batchSize;

    if constexpr (IsFloat2) {
        float2 accum = {0.0f, 0.0f};
        int32_t j = 0;
        for (; j + UNROLL_FACTOR <= vectorNum; j += UNROLL_FACTOR) {
            accum += src[inverse[j + start] * evSizeVec + indicesDimVec];
            accum += src[inverse[j + start + 1] * evSizeVec + indicesDimVec];
            accum += src[inverse[j + start + 2] * evSizeVec + indicesDimVec];
            accum += src[inverse[j + start + 3] * evSizeVec + indicesDimVec];
        }
        for (; j < vectorNum; ++j) {
            int32_t srcIndex = inverse[j + start];
            accum += src[srcIndex * evSizeVec + indicesDimVec];
        }
        if (combiner > 0) {
            accum.x /= vectorNum;
            accum.y /= vectorNum;
        }
        const int32_t dstFloat2Base = (dstRowIndex * totalDims + accumDims + dstColIndex * evSize) >> 1;
        dst[dstFloat2Base + indicesDimVec] = accum;
    } else {
        float accum{0.0f};
        int32_t j = 0;
        for (; j + UNROLL_FACTOR <= vectorNum; j += UNROLL_FACTOR) {
            accum += src[inverse[j + start] * evSizeVec + indicesDimVec];
            accum += src[inverse[j + start + 1] * evSizeVec + indicesDimVec];
            accum += src[inverse[j + start + 2] * evSizeVec + indicesDimVec];
            accum += src[inverse[j + start + 3] * evSizeVec + indicesDimVec];
        }
        for (; j < vectorNum; ++j) {
            int32_t srcIndex = inverse[j + start];
            accum += src[srcIndex * evSizeVec + indicesDimVec];
        }
        if (combiner > 0) {
            accum /= vectorNum;
        }
        dst[dstRowIndex * totalDims + accumDims + dstColIndex * evSize + indicesDimVec] = accum;
    }
}

// SIMT VF函数 - 大数据模式
template <typename T1, typename T2, typename T3, bool IsFloat2 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ T1* src, __gm__ T2* dst, __gm__ T3* offset, __gm__ T3* inverse, int32_t combiner, int32_t totalDims,
    int32_t accumDims, int32_t evSize, int32_t evSizeVec, int32_t numVec, int32_t batchSize, int32_t totalBlocks,
    int32_t blockStartIdx, int32_t curBlocksCount, int32_t outLen)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        // 1.位置计算
        int32_t globalBlockIdx = blockStartIdx + iter;

        int32_t blockBase = globalBlockIdx * blockThreadNum;
        if (blockBase >= outLen) {
            break;
        }

        int32_t elementsRemaining = outLen - blockBase;
        int32_t elementsThisBlock = (elementsRemaining < blockThreadNum) ? elementsRemaining : blockThreadNum;

        int32_t threadElementBase = blockBase + threadIdx;

        if (threadElementBase >= outLen) {
            break;
        }

        int32_t indicesIndex = threadElementBase / evSizeVec;
        int32_t indicesDimVec = threadElementBase % evSizeVec;
        int32_t start = offset[indicesIndex] - offset[0];
        int32_t vectorNum = offset[indicesIndex + 1] - offset[indicesIndex];
        int32_t dstRowIndex = indicesIndex % batchSize;
        int32_t dstColIndex = indicesIndex / batchSize;

        if constexpr (IsFloat2) {
            float2 accum = {0.0f, 0.0f};
            int32_t j = 0;
            for (; j + UNROLL_FACTOR <= vectorNum; j += UNROLL_FACTOR) {
                accum += src[inverse[j + start] * evSizeVec + indicesDimVec];
                accum += src[inverse[j + start + 1] * evSizeVec + indicesDimVec];
                accum += src[inverse[j + start + 2] * evSizeVec + indicesDimVec];
                accum += src[inverse[j + start + 3] * evSizeVec + indicesDimVec];
            }
            for (; j < vectorNum; ++j) {
                int32_t srcIndex = inverse[j + start];
                accum += src[srcIndex * evSizeVec + indicesDimVec];
            }
            if (combiner > 0) {
                accum.x /= vectorNum;
                accum.y /= vectorNum;
            }
            const int32_t dstFloat2Base = (dstRowIndex * totalDims + accumDims + dstColIndex * evSize) >> 1;
            dst[dstFloat2Base + indicesDimVec] = accum;
        } else {
            float accum{0.0f};
            int32_t j = 0;
            for (; j + UNROLL_FACTOR <= vectorNum; j += UNROLL_FACTOR) {
                accum += src[inverse[j + start] * evSizeVec + indicesDimVec];
                accum += src[inverse[j + start + 1] * evSizeVec + indicesDimVec];
                accum += src[inverse[j + start + 2] * evSizeVec + indicesDimVec];
                accum += src[inverse[j + start + 3] * evSizeVec + indicesDimVec];
            }
            for (; j < vectorNum; ++j) {
                int32_t srcIndex = inverse[j + start];
                accum += src[srcIndex * evSizeVec + indicesDimVec];
            }
            if (combiner > 0) {
                accum /= vectorNum;
            }
            dst[dstRowIndex * totalDims + accumDims + dstColIndex * evSize + indicesDimVec] = accum;
        }
    }
}
}  // namespace PoolingEmbeddingsSimt
