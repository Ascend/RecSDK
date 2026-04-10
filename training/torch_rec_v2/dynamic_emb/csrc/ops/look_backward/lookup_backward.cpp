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
#include <type_traits>

#include "lookup_backward.h"
#include "kernel_operator.h"


#define DTYPE_DISPATCH(isInt32, DTYPE, ...) \
do {                                         \
        if (isInt32) {                       \
            using DTYPE = int32_t;           \
            __VA_ARGS__;                            \
        } else {                               \
            using DTYPE = int64_t;           \
            __VA_ARGS__;                  }         \
} while (0)
constexpr int32_t BLOCK_THREADS = LookupBackwardSimt::MAX_THREADS_PER_BLOCK;

extern "C" __global__ __aicore__ void lookup_backward(
                 GM_ADDR gradData, GM_ADDR uniqueBufferData,
                 GM_ADDR uniqueIndicesData,
                 GM_ADDR inverseIndicesData,
                 GM_ADDR biasedOffsetsData,  int32_t dim,
                 int32_t tableNum, int32_t batchSize, int32_t featureNum,
                 int32_t numKey, int32_t combiner,
                 int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks,
                 bool isInt32, bool isSmall, GM_ADDR kernelStatus)
{
    int32_t coreId = AscendC::GetBlockIdx();

    DTYPE_DISPATCH(isInt32, DTYPE_X, {
        __gm__ float* grad = reinterpret_cast<__gm__ float*>(gradData);
        __gm__ float* uniqueBuffer = reinterpret_cast<__gm__ float*>(uniqueBufferData);
        __gm__ DTYPE_X* uniqueIndices = reinterpret_cast<__gm__ DTYPE_X*>(uniqueIndicesData);
        __gm__ DTYPE_X* inverseIndices = reinterpret_cast<__gm__ DTYPE_X*>(inverseIndicesData);
        __gm__ DTYPE_X* biasedOffsets = reinterpret_cast<__gm__ DTYPE_X*>(biasedOffsetsData);
        __gm__ bool* kernelStatusPtr = reinterpret_cast<__gm__ bool*>(kernelStatus);

        if (isSmall) {
            AscendC::Simt::VF_CALL<LookupBackwardSimt::SimtSmallDataCompute<DTYPE_X>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1},grad, uniqueBuffer, uniqueIndices,inverseIndices,
                biasedOffsets, dim, tableNum, batchSize, featureNum, numKey, combiner, kernelStatusPtr);
        } else {
            int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
            int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

            AscendC::Simt::VF_CALL<LookupBackwardSimt::SimtLargeDataCompute<DTYPE_X>>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grad, uniqueBuffer, uniqueIndices,inverseIndices,
            biasedOffsets, dim, tableNum, batchSize, featureNum, numKey, combiner,
             totalBlocks,blockStartIdx,curBlocksCount, kernelStatusPtr);
        }
    });
}