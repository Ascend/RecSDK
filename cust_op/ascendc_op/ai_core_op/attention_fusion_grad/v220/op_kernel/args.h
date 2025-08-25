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

#ifndef ATTENTION_FUSION_GRAD_ARGS_H
#define ATTENTION_FUSION_GRAD_ARGS_H
#include "kernel_operator.h"

using namespace AscendC;

namespace AscendFusionGrad {

constexpr int FLOAT_ALIGNMENT = 8;
constexpr int TRANSPOSE_ALIGNMENT = 16;
constexpr int KEY_DIM1_COPY_ALIGN_MODE = 1;
constexpr int KEY_DIM1_COPY_TRANSPOSE_ALIGN_MODE = 2;
constexpr int KEY_DIM1_COPY_PAD_MODE = 3;

struct InputArgs {
    GM_ADDR dout;
    GM_ADDR softmaxOut;
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR workspace;

    float attenDimSqrt;
};

struct OutputArgs {
    GM_ADDR gradQuery;
    GM_ADDR gradKey;
    GM_ADDR gradValue;
};

struct ShapeArgs {
    int32_t batchNum;
    int32_t queryDim1;
    int32_t queryDim2;
    int32_t keyDim1;
    int32_t keyDim2;
    int32_t valueDim1;
    int32_t valueDim2;
    
    int32_t paddingKeyDim1;
    int32_t keyDim1Align;
    int32_t ubCanUsed;
    int32_t numRowOfNormalizeOne;
    
    int32_t batchOffset;
    int32_t batchLen;
};

struct ShapeTilingArgs {
    int32_t paddingKeyDim1;
    int32_t keyDim1Align;
    int32_t transposeAlignDim;
    int32_t numRowOfNormalizeOne;
    
    int32_t batchOffset;
    int32_t batchLen;
};

struct TilingArgs {
    const TCubeTiling* gardVMatmulTiling;
    const TCubeTiling* gardSMatmulTiling;
    const TCubeTiling* gardQMatmulTiling;
    const TCubeTiling* gardKMatmulTiling;

    const ConfusionTransposeTiling* unAlign2AlignStep1Tiling;
    const ConfusionTransposeTiling* unAlign2AlignStep2Tiling;
    const ConfusionTransposeTiling* Align2UnAlignStep1Tiling;
    const ConfusionTransposeTiling* Align2UnAlignStep2Tiling;

    const SoftMaxTiling* softmaxtiling;
};

struct AttentionFusionGradArgs {
    InputArgs inputArgs;
    OutputArgs outputArgs;
    ShapeArgs shapeArgs;
    ShapeTilingArgs shapeTilingArgs;
    TilingArgs tilingArgs;
};

struct NormGradPipeArgs {
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1>* vecInQueue;
    TQue<QuePosition::VECIN, 1>* vecInGradQueue;
    TQue<QuePosition::VECOUT, 1>* vecOutQueue;
    TBuf<TPosition::VECCALC>* tmpBuff;
};
}

#endif