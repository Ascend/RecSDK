/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "kernel_operator.h"
#include "attention_fusion_kernel.h"
using namespace AscendC;

namespace Attention_Kernel {
// call of kernel function
extern "C" __global__ __aicore__ void attention_fusion(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attnMask,
                                            GM_ADDR attenScore, GM_ADDR softmaxOut, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    const TCubeTiling *qkMatmulTiling = &tiling_data.qkMatmulTiling;
    const TCubeTiling *kvMatmulTiling = &tiling_data.kvMatmulTiling;
    const SoftMaxTiling *softMaxTilingData = &tiling_data.softMaxTilingData;

    AttentionFusionArgs args {
        query, key, value, attnMask, attenScore, softmaxOut, tiling_data.normalizeAttr, tiling_data.queryDim1,
        tiling_data.queryDim2, tiling_data.keyDim1, tiling_data.valueDim2, tiling_data.batchNum,
        tiling_data.normalizeLoop, tiling_data.normalizeRow, tiling_data.normalizeColumn, tiling_data.maskIsOn,
        tiling_data.normalizeSqrt, tiling_data.maxSharedTmpBuf, qkMatmulTiling, kvMatmulTiling,
        softMaxTilingData, &tiling_data.confusionTransposeTilingData, &tiling_data.confusionTransposeTilingData1,
        &tiling_data.confusionTransposeTilingData2, &tiling_data.confusionTransposeTilingData3
    };

    AttentionFusionKernel<float, float, float> kernel;
    kernel.Compute(args);
}
}