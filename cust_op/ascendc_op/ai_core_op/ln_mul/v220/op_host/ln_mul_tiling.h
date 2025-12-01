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

#ifndef MXREC_LAYERNORM_MUL_TILING_H
#define MXREC_LAYERNORM_MUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LnMulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, aLength);
TILING_DATA_FIELD_DEF(uint32_t, rLength);
TILING_DATA_FIELD_DEF(uint32_t, rLengthWithPadding);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);             // 实际使用核数
TILING_DATA_FIELD_DEF(uint32_t, perCoreComputeRows);  // 每核每次处理的行数
TILING_DATA_FIELD_DEF(uint32_t, formerCoreRows);      // 前核分配的行数
TILING_DATA_FIELD_DEF(uint32_t, loopCountFormer);     // 前核完整迭代次数
TILING_DATA_FIELD_DEF(uint32_t, formerRowLeft);       // 前核剩余行数
TILING_DATA_FIELD_DEF(uint32_t, loopCountTail);       // 尾核完整迭代次数
TILING_DATA_FIELD_DEF(uint32_t, tailRowLeft);         // 尾核剩余行数
TILING_DATA_FIELD_DEF(uint32_t, baseCoreRows);        // 尾核分配的行数
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNums);      // 分配到formerCoreRows数据的核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LnMul, LnMulTilingData)
}  // namespace optiling

#endif  // MXREC_LAYERNORM_MUL_TILING_H
