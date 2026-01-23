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

#ifndef EXPAND_INTO_JAGGED_PERMUTE_TILING
#define EXPAND_INTO_JAGGED_PERMUTE_TILING
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(ExpandIntoJaggedPermuteTilingData)
    TILING_DATA_FIELD_DEF(int64_t, outputSize);
    TILING_DATA_FIELD_DEF(int64_t, permuteLen);
    TILING_DATA_FIELD_DEF(uint64_t, ubCanUsed);
    TILING_DATA_FIELD_DEF(int64_t, splitBaseLen);
    TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(ExpandIntoJaggedPermute, ExpandIntoJaggedPermuteTilingData)

} // namespce optiling
#endif
