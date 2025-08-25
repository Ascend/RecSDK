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

#ifndef ATTENTION_FUSION_GRAD_TILING_H
#define ATTENTION_FUSION_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AttentionFusionGradTilingData)
        TILING_DATA_FIELD_DEF(float, attnDim);
        TILING_DATA_FIELD_DEF(int32_t, queryDim1);
        TILING_DATA_FIELD_DEF(int32_t, queryDim2);
        TILING_DATA_FIELD_DEF(int32_t, keyDim1);
        TILING_DATA_FIELD_DEF(int32_t, keyDim2);
        TILING_DATA_FIELD_DEF(int32_t, valueDim1);
        TILING_DATA_FIELD_DEF(int32_t, valueDim2);
        TILING_DATA_FIELD_DEF(int32_t, batchNum);
        TILING_DATA_FIELD_DEF(int32_t, numRowOfNormalizeOne);
        TILING_DATA_FIELD_DEF(int32_t, paddingKeyDim1);
        TILING_DATA_FIELD_DEF(float, attenDimSqrt);
        TILING_DATA_FIELD_DEF(int32_t, keyDim1Align);
        TILING_DATA_FIELD_DEF(int32_t, transposeAlignDim);

        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardVMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardSMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardQMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gardKMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softMaxGradTiling);

        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, unAlign2AlignStep1Tiling);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, unAlign2AlignStep2Tiling);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, Align2UnAlignStep1Tiling);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, Align2UnAlignStep2Tiling);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionFusionGrad, AttentionFusionGradTilingData)
}

#endif