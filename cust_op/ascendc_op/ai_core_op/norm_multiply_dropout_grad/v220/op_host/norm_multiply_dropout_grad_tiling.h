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
#ifndef NORM_MULTIPLY_DROPOUT_GRAD_TILING
#define NORM_MULTIPLY_DROPOUT_GRAD_TILING

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NormMultiplyDropoutGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, littleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, avgCoreCalcRows);
TILING_DATA_FIELD_DEF(uint32_t, xRowCount);
TILING_DATA_FIELD_DEF(uint32_t, xColCount);
TILING_DATA_FIELD_DEF(uint32_t, singleBlockRows);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, dropoutRatio);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NormMultiplyDropoutGrad, NormMultiplyDropoutGradTilingData)
}  // namespace optiling
#endif