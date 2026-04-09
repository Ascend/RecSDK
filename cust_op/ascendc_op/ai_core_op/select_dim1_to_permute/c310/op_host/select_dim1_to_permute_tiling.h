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

#ifndef SELECT_DIM1_TO_PERMUTE_TILING
#define SELECT_DIM1_TO_PERMUTE_TILING
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectDim1ToPermuteTilingData)
TILING_DATA_FIELD_DEF(int64_t, lengthsSize);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, indicesLength);
TILING_DATA_FIELD_DEF(int64_t, batchNum);
TILING_DATA_FIELD_DEF(int64_t, splitBaseLen);
TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);
TILING_DATA_FIELD_DEF(int64_t, blockLen);
TILING_DATA_FIELD_DEF(int64_t, batchSizeWithPadding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectDim1ToPermute, SelectDim1ToPermuteTilingData)
}  // namespace optiling

#endif
