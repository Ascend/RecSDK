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

#ifndef IN_LINEAR_SILU_TILING_H
#define IN_LINEAR_SILU_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InLinearSiluTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, mShape);
TILING_DATA_FIELD_DEF(uint32_t, kShape);
TILING_DATA_FIELD_DEF(uint32_t, nShape);
TILING_DATA_FIELD_DEF(uint32_t, uLength);
TILING_DATA_FIELD_DEF(uint32_t, vLength);
TILING_DATA_FIELD_DEF(uint32_t, qLength);
TILING_DATA_FIELD_DEF(uint32_t, kLength);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(uint32_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, formerSingleLen);
TILING_DATA_FIELD_DEF(uint32_t, tailSingleLen);
TILING_DATA_FIELD_DEF(uint32_t, formerLoop);
TILING_DATA_FIELD_DEF(uint32_t, tailLoop);
TILING_DATA_FIELD_DEF(uint32_t, formerPerLoopLen);
TILING_DATA_FIELD_DEF(uint32_t, tailPerLoopLen);
TILING_DATA_FIELD_DEF(uint32_t, formerRemain);
TILING_DATA_FIELD_DEF(uint32_t, tailRemain);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, bufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InLinearSilu, InLinearSiluTilingData)
}  // namespace optiling
#endif