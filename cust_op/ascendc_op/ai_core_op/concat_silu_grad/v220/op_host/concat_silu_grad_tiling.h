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
#ifndef CONCAT_SILU_GRAD_TILING
#define CONCAT_SILU_GRAD_TILING
#include "register/tilingdata_base.h"

namespace optiling {
constexpr size_t MAX_SPLIT_NUM = 4;

BEGIN_TILING_DATA_DEF(ConcatSiluGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_SPLIT_NUM, splitList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConcatSiluGrad, ConcatSiluGradTilingData)
}  // namespace optiling
#endif