/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef IN_LINEAR_SILU_BACKWARD_TILING_H
#define IN_LINEAR_SILU_BACKWARD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InLinearSiluBackwardTilingData)
TILING_DATA_FIELD_DEF(uint32_t, embedDim);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, uDim);
TILING_DATA_FIELD_DEF(uint32_t, vDim);
TILING_DATA_FIELD_DEF(uint32_t, qDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, blockK);
TILING_DATA_FIELD_DEF(uint32_t, blockM);
TILING_DATA_FIELD_DEF(bool, enableBias);
TILING_DATA_FIELD_DEF(bool, isTrans);
TILING_DATA_FIELD_DEF(uint32_t, aivNum);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, LW_MM);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, LX_MM);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InLinearSiluBackward, InLinearSiluBackwardTilingData)
}  // namespace optiling
#endif
