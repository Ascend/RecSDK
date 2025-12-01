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
#ifndef Invert_Permute_H
#define Invert_Permute_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InvertPermuteTilingData)
    TILING_DATA_FIELD_DEF(int32_t, xDim0);
    TILING_DATA_FIELD_DEF(uint32_t, actualThreadsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InvertPermute, InvertPermuteTilingData)
}
#endif // Invert_Permute_H