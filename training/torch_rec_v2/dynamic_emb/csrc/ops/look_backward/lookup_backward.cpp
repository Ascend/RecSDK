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

#define LOOKUP_BACKWARD_LAUNCH(IS_MEAN, IS_FLOAT2)                                                              \
    LookupBackwardSimt::LaunchBackwardCompute<DTYPE_X, value_t, IS_MEAN, IS_FLOAT2>(                            \
        grad, uniqueBuffer, inverseIndices, biasedOffsets, dim, numKey, numSamples, totalBlocks, blocksPerCore, \
        remainderBlocks, isSmall, coreId)

#define LOOKUP_BACKWARD_SCATTER(IS_MEAN, IS_FLOAT2)   \
    do {                                              \
        if (isMean) {                                 \
            LOOKUP_BACKWARD_LAUNCH(true, IS_FLOAT2);  \
        } else {                                      \
            LOOKUP_BACKWARD_LAUNCH(false, IS_FLOAT2); \
        }                                             \
    } while (0)

extern "C" __global__ __aicore__ void lookup_backward(GM_ADDR gradData, GM_ADDR uniqueBufferData,
                                                      GM_ADDR inverseIndicesData, GM_ADDR biasedOffsetsData,
                                                      int32_t dim, int32_t numKey, int32_t numSamples, int32_t combiner,
                                                      int32_t totalBlocks, int32_t blocksPerCore,
                                                      int32_t remainderBlocks, uint32_t indexTypeNum, bool isSmall,
                                                      bool isFloat2, uint32_t valueTypeNum)
{
    int32_t coreId = AscendC::GetBlockIdx();
    dyn_emb::DataType valueType = static_cast<dyn_emb::DataType>(valueTypeNum);
    dyn_emb::DataType indexType = static_cast<dyn_emb::DataType>(indexTypeNum);
    const bool isMean = (combiner == 1);

    if (isFloat2) {
        INDEX_DTYPE_DISPATCH(indexType, DTYPE_X, {
            __gm__ float2* grad = reinterpret_cast<__gm__ float2*>(gradData);
            __gm__ float2* uniqueBuffer = reinterpret_cast<__gm__ float2*>(uniqueBufferData);
            __gm__ DTYPE_X* inverseIndices = reinterpret_cast<__gm__ DTYPE_X*>(inverseIndicesData);
            __gm__ DTYPE_X* biasedOffsets = reinterpret_cast<__gm__ DTYPE_X*>(biasedOffsetsData);
            using value_t = float2;
            LOOKUP_BACKWARD_SCATTER(isMean, true);
        });
        return;
    }

    INDEX_DTYPE_DISPATCH(indexType, DTYPE_X, {
        FLOAT_TYPE_DISPATCH(valueType, value_t, {
            __gm__ value_t* grad = reinterpret_cast<__gm__ value_t*>(gradData);
            __gm__ value_t* uniqueBuffer = reinterpret_cast<__gm__ value_t*>(uniqueBufferData);
            __gm__ DTYPE_X* inverseIndices = reinterpret_cast<__gm__ DTYPE_X*>(inverseIndicesData);
            __gm__ DTYPE_X* biasedOffsets = reinterpret_cast<__gm__ DTYPE_X*>(biasedOffsetsData);
            LOOKUP_BACKWARD_SCATTER(isMean, false);
        });
    });
}
