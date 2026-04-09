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
#include "pooling_embeddings_kernel.h"
#include "kernel_operator.h"
#include "../ops_utils.h"

extern "C" __global__ __aicore__ void pooling_embeddings(GM_ADDR srcData, GM_ADDR dstData, GM_ADDR offsetData,
                                                    GM_ADDR inverseData, int32_t combiner, int32_t totalDims,
                                                    int32_t accumDims, int32_t evSize, int32_t numVec,
                                                    int32_t batchSize, int32_t totalBlocks, int32_t blocksPerCore,
                                                    int32_t remainderBlocks, bool isSmall, uint32_t srcTypeNum,
                                                    uint32_t dstTypeNum, uint32_t offsetTypeNum, uint32_t threads,
                                                    int32_t outLen, bool isFloat2, int32_t evSizeVec)
{
    int32_t coreId = AscendC::GetBlockIdx();

    dyn_emb::DataType offsetType = static_cast<dyn_emb::DataType>(offsetTypeNum);
    dyn_emb::DataType srcType = static_cast<dyn_emb::DataType>(srcTypeNum);
    dyn_emb::DataType dstType = static_cast<dyn_emb::DataType>(dstTypeNum);

    if (isFloat2) {
        INT_TYPE_DISPATCH(offsetType, offset_t, {
            __gm__ offset_t* offset = reinterpret_cast<__gm__ offset_t*>(offsetData);
            __gm__ offset_t* inverse = reinterpret_cast<__gm__ offset_t*>(inverseData);
            __gm__ float2* src = reinterpret_cast<__gm__ float2*>(srcData);
            __gm__ float2* dst = reinterpret_cast<__gm__ float2*>(dstData);

            if (isSmall) {
                AscendC::Simt::VF_CALL<PoolingEmbeddingsSimt::SimtSmallDataCompute<float2, float2, offset_t, true>>(
                    AscendC::Simt::Dim3{threads, 1, 1}, src, dst, offset, inverse, combiner,
                    totalDims, accumDims, evSize, evSizeVec, numVec, batchSize, outLen);
            } else {
                int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
                int32_t blockStartIdx = coreId * blocksPerCore +
                    ((coreId < remainderBlocks) ? coreId : remainderBlocks);

                AscendC::Simt::VF_CALL<PoolingEmbeddingsSimt::SimtLargeDataCompute<float2, float2, offset_t, true>>(
                    AscendC::Simt::Dim3{threads, 1, 1}, src, dst, offset, inverse, combiner,
                    totalDims, accumDims, evSize, evSizeVec, numVec, batchSize, totalBlocks,
                    blockStartIdx, curBlocksCount, outLen);
            }
        });
    } else {
        INT_TYPE_DISPATCH(offsetType, offset_t, {
            FLOAT_TYPE_DISPATCH(srcType, src_t, {
                FLOAT_TYPE_DISPATCH(dstType, dst_t, {
                        __gm__ offset_t* offset = reinterpret_cast<__gm__ offset_t*>(offsetData);
                        __gm__ offset_t* inverse = reinterpret_cast<__gm__ offset_t*>(inverseData);
                        __gm__ src_t* src = reinterpret_cast<__gm__ src_t*>(srcData);
                        __gm__ dst_t* dst = reinterpret_cast<__gm__ dst_t*>(dstData);

                        if (isSmall) {
                            AscendC::Simt::VF_CALL<PoolingEmbeddingsSimt::SimtSmallDataCompute<src_t, dst_t, offset_t, false>>(
                                AscendC::Simt::Dim3{threads, 1, 1}, src, dst, offset, inverse, combiner,
                                totalDims, accumDims, evSize, evSizeVec, numVec, batchSize, outLen);
                        } else {
                            int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
                            int32_t blockStartIdx = coreId * blocksPerCore +
                                ((coreId < remainderBlocks) ? coreId : remainderBlocks);

                            AscendC::Simt::VF_CALL<PoolingEmbeddingsSimt::SimtLargeDataCompute<src_t, dst_t, offset_t, false>>(
                                AscendC::Simt::Dim3{threads, 1, 1}, src, dst, offset, inverse, combiner,
                                totalDims, accumDims, evSize, evSizeVec, numVec, batchSize, totalBlocks,
                                blockStartIdx, curBlocksCount, outLen);
                        }
                });
            });
        });
    }
}
