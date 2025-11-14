/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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


#ifndef HSTU_DENSE_BACKWARD_TILING_H
#define HSTU_DENSE_BACKWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

#include "hstu_dense_backward_tiling_common.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HstuDenseBackwardTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qkMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, qGradMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, kGradMatmul);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, vGradMatmul);

TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, headNum);
TILING_DATA_FIELD_DEF(int64_t, headDim);

TILING_DATA_FIELD_DEF(int64_t, blockHeight);
TILING_DATA_FIELD_DEF(int64_t, dataTypeLength);

TILING_DATA_FIELD_DEF(int32_t, maskType);
TILING_DATA_FIELD_DEF(int32_t, enableBias);
TILING_DATA_FIELD_DEF(int32_t, enableContextMask);
TILING_DATA_FIELD_DEF(int32_t, enableTargetMask);
TILING_DATA_FIELD_DEF(int32_t, maxSeqLen);
TILING_DATA_FIELD_DEF(int32_t, biasGradSeqLen);
TILING_DATA_FIELD_DEF(float, siluScale);
TILING_DATA_FIELD_DEF(int32_t, targetGroupSize);
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(int32_t, isNormal);
TILING_DATA_FIELD_DEF(uint32_t, aivNum);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HstuDenseBackward, HstuDenseBackwardTilingData)
} // namespace optiling
#endif