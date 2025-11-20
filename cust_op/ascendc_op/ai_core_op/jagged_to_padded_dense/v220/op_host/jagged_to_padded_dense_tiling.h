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

#ifndef INDEX_SELECT_RANK1_BACKWARD_TILING
#define INDEX_SELECT_RANK1_BACKWARD_TILING

#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(JaggedToPaddedDenseTilingData)

    TILING_DATA_FIELD_DEF(int64_t, totalBatch);
    TILING_DATA_FIELD_DEF(int64_t, baseBatchLen);
    TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);

    TILING_DATA_FIELD_DEF(int64_t, valuesDim0);
    TILING_DATA_FIELD_DEF(int64_t, valuesDim1);
    TILING_DATA_FIELD_DEF(int64_t, offsetDim0);
    TILING_DATA_FIELD_DEF(int64_t, outDim1);

    TILING_DATA_FIELD_DEF(int64_t, ubCanUsed);

    TILING_DATA_FIELD_DEF(float, paddingValueFp32);
    TILING_DATA_FIELD_DEF(int64_t, paddingValueInt64);

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(JaggedToPaddedDense, JaggedToPaddedDenseTilingData)
}  // namespace optiling
#endif