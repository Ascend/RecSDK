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

#include "ops_utils.h"
#include "lookup_backward.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_THREADS = LookupBackwardSimt::MAX_THREADS_PER_BLOCK;

extern "C" __global__ __aicore__ void lookup_backward(GM_ADDR gradData, GM_ADDR uniqueBufferData,
                                                      GM_ADDR uniqueIndicesData, GM_ADDR inverseIndicesData,
                                                      GM_ADDR biasedOffsetsData, int32_t dim, int32_t tableNum,
                                                      int32_t batchSize, int32_t featureNum, int32_t numKey,
                                                      int32_t combiner, int32_t totalBlocks, int32_t blocksPerCore,
                                                      int32_t remainderBlocks, uint32_t indexTypeNum, bool isSmall,
                                                      uint32_t valueTypeNum, GM_ADDR kernelStatus)
{
    int32_t coreId = AscendC::GetBlockIdx();
    dyn_emb::DataType valueType = static_cast<dyn_emb::DataType>(valueTypeNum);
    dyn_emb::DataType indexType = static_cast<dyn_emb::DataType>(indexTypeNum);

    INDEX_DTYPE_DISPATCH(indexType, DTYPE_X, {
        FLOAT_TYPE_DISPATCH(valueType, value_t, {
            __gm__ value_t* grad = reinterpret_cast<__gm__ value_t*>(gradData);
            __gm__ value_t* uniqueBuffer = reinterpret_cast<__gm__ value_t*>(uniqueBufferData);
            __gm__ DTYPE_X* uniqueIndices = reinterpret_cast<__gm__ DTYPE_X*>(uniqueIndicesData);
            __gm__ DTYPE_X* inverseIndices = reinterpret_cast<__gm__ DTYPE_X*>(inverseIndicesData);
            __gm__ DTYPE_X* biasedOffsets = reinterpret_cast<__gm__ DTYPE_X*>(biasedOffsetsData);
            __gm__ bool* kernelStatusPtr = reinterpret_cast<__gm__ bool*>(kernelStatus);

            if (isSmall) {
                AscendC::Simt::VF_CALL<LookupBackwardSimt::SimtSmallDataCompute<DTYPE_X, value_t>>(
                    AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grad, uniqueBuffer, uniqueIndices, inverseIndices,
                    biasedOffsets, dim, tableNum, batchSize, featureNum, numKey, combiner, kernelStatusPtr);
            } else {
                int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
                int32_t blockStartIdx =
                    coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

                AscendC::Simt::VF_CALL<LookupBackwardSimt::SimtLargeDataCompute<DTYPE_X, value_t>>(
                    AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grad, uniqueBuffer, uniqueIndices, inverseIndices,
                    biasedOffsets, dim, tableNum, batchSize, featureNum, numKey, combiner, totalBlocks, blockStartIdx,
                    curBlocksCount, kernelStatusPtr);
            }
        });
    });
}
