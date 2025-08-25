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

#include "args.h"
#include "attention_fusion_grad_kernel.h"
#include "kernel_operator.h"
#include "utils.h"

using namespace AscendC;
using namespace AscendFusionGrad;

extern "C" __global__ __aicore__ void attention_fusion_grad(GM_ADDR dout, GM_ADDR softmaxOut, GM_ADDR query,
                                                            GM_ADDR key, GM_ADDR value, GM_ADDR gradQuery,
                                                            GM_ADDR gradKey, GM_ADDR gradValue, GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // calculate batch offset
    int batchOffset;
    int batchLen;

    int blockLenPerCoreBase = tilingData.batchNum / (GetBlockNum() * 2);
    int remain = tilingData.batchNum % (GetBlockNum() * 2);
    if (GetBlockIdx() < remain) {
        batchLen = blockLenPerCoreBase + 1;
        batchOffset = GetBlockIdx() * batchLen;
    } else {
        batchLen = blockLenPerCoreBase;
        batchOffset = GetBlockIdx() * blockLenPerCoreBase + remain;
    }

    InputArgs inputArgs{dout, softmaxOut, query, key, value, workspace, tilingData.attenDimSqrt};

    OutputArgs outputArgs{gradQuery, gradKey, gradValue};

    ShapeArgs shapeArgs{tilingData.batchNum, tilingData.queryDim1, tilingData.queryDim2, tilingData.keyDim1,
                        tilingData.keyDim2,  tilingData.valueDim1, tilingData.valueDim2};

    ShapeTilingArgs shapeTilingArgs{tilingData.paddingKeyDim1,
                                    tilingData.keyDim1Align,
                                    tilingData.transposeAlignDim,
                                    tilingData.numRowOfNormalizeOne,
                                    batchOffset,
                                    batchLen};

    TilingArgs tilingArgs{&tilingData.gardVMatmulTiling,        &tilingData.gardSMatmulTiling,
                          &tilingData.gardQMatmulTiling,        &tilingData.gardKMatmulTiling,
                          &tilingData.unAlign2AlignStep1Tiling, &tilingData.unAlign2AlignStep2Tiling,
                          &tilingData.Align2UnAlignStep1Tiling, &tilingData.Align2UnAlignStep2Tiling,
                          &tilingData.softMaxGradTiling};

    AttentionFusionGradArgs attentionFusionGradAgs{inputArgs, outputArgs, shapeArgs, shapeTilingArgs, tilingArgs};

    AttentionFusionGradKernel<float> attentionGradKernel;
    attentionGradKernel.Compute(attentionFusionGradAgs);
}