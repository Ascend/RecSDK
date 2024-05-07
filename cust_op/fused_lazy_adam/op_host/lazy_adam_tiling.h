/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef LAZY_ADAM_TILING_H
#define LAZY_ADAM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LazyAdamTilingData)
TILING_DATA_FIELD_DEF(float, beta1);
TILING_DATA_FIELD_DEF(float, beta2);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int32_t, dim0);
TILING_DATA_FIELD_DEF(int32_t, dim1);
TILING_DATA_FIELD_DEF(int32_t, dim2);
TILING_DATA_FIELD_DEF(int32_t, row);
TILING_DATA_FIELD_DEF(int32_t, indicesAllocSize);
TILING_DATA_FIELD_DEF(int32_t, otherAllocSize);
TILING_DATA_FIELD_DEF(int32_t, batch);
TILING_DATA_FIELD_DEF(int32_t, loopCount);
TILING_DATA_FIELD_DEF(int32_t, rowLeft);
TILING_DATA_FIELD_DEF(int32_t, loopCountTail);
TILING_DATA_FIELD_DEF(int32_t, rowLeftTail);
TILING_DATA_FIELD_DEF(int32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LazyAdam, LazyAdamTilingData)
}  // namespace optiling
#endif  // LAZY_ADAM_TILING_H