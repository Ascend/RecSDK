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

#ifndef ATTENTION_FUSION_TILING_H
#define ATTENTION_FUSION_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
    BEGIN_TILING_DATA_DEF(AttentionFusionTilingData)
        TILING_DATA_FIELD_DEF(uint8_t, normalizeAttr);
        TILING_DATA_FIELD_DEF(float, attnDim);
        TILING_DATA_FIELD_DEF(int32_t, queryDim1);
        TILING_DATA_FIELD_DEF(int32_t, queryDim2);
        TILING_DATA_FIELD_DEF(int32_t, keyDim1);
        TILING_DATA_FIELD_DEF(int32_t, valueDim2);
        TILING_DATA_FIELD_DEF(int32_t, batchNum);
        TILING_DATA_FIELD_DEF(int32_t, normalizeLoop);
        TILING_DATA_FIELD_DEF(int32_t, normalizeRow);
        TILING_DATA_FIELD_DEF(int32_t, normalizeColumn);
        TILING_DATA_FIELD_DEF(int32_t, maskIsOn);
        TILING_DATA_FIELD_DEF(float, normalizeSqrt);
        TILING_DATA_FIELD_DEF(uint64_t, maxSharedTmpBuf);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, kvMatmulTiling);
        TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softMaxTilingData);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData1);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData2);
        TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData3);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(AttentionFusion, AttentionFusionTilingData)
}
#endif