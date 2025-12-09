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

#ifndef TOKEN_MIXING_TILING_H
#define TOKEN_MIXING_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TokenMixingTilingData)
TILING_DATA_FIELD_DEF(uint64_t, xDim0);
TILING_DATA_FIELD_DEF(uint64_t, xDim1);
TILING_DATA_FIELD_DEF(uint64_t, xDim2);
TILING_DATA_FIELD_DEF(uint64_t, xDim2WithPadding);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint64_t, perCoreComputeRows);
TILING_DATA_FIELD_DEF(uint64_t, formerCoreRows);
TILING_DATA_FIELD_DEF(uint64_t, formerLoopCount);
TILING_DATA_FIELD_DEF(uint64_t, formerRemainRows);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreRows);
TILING_DATA_FIELD_DEF(uint64_t, tailLoopCount);
TILING_DATA_FIELD_DEF(uint64_t, tailRemainRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TokenMixing, TokenMixingTilingData)
}  // namespace optiling
#endif