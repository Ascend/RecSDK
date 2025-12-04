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

#ifndef PERMUTE_2D_SPARSE_DATA_TILING
#define PERMUTE_2D_SPARSE_DATA_TILING

#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(Permute2dSparseDataTilingData)

    TILING_DATA_FIELD_DEF(size_t, coreNum);

    TILING_DATA_FIELD_DEF(int64_t, permuteDim0);
    TILING_DATA_FIELD_DEF(int64_t, lengthsT);
    TILING_DATA_FIELD_DEF(int64_t, lengthsB);
    TILING_DATA_FIELD_DEF(int64_t, valuesDim);
    TILING_DATA_FIELD_DEF(int64_t, valuesOutDim);
    TILING_DATA_FIELD_DEF(bool, enableWeights);

    // 判断是否传入totalOffset, 传入totalOffset时，采用行内分核方案
    TILING_DATA_FIELD_DEF(bool, enableTotalOffset);

    TILING_DATA_FIELD_DEF(int64_t, totalBatch);
    TILING_DATA_FIELD_DEF(int64_t, baseBatchLen);
    TILING_DATA_FIELD_DEF(int64_t, tailSplitIndex);

    TILING_DATA_FIELD_DEF(int64_t, ubCanUsed);

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(Permute2dSparseData, Permute2dSparseDataTilingData)
}  // namespace optiling
#endif
