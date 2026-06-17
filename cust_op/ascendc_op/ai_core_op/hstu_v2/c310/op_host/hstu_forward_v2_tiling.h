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

#ifndef HSTU_FORWARD_V2_TILING_H
#define HSTU_FORWARD_V2_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HstuForwardV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, dimQK);
TILING_DATA_FIELD_DEF(uint32_t, dimV);
TILING_DATA_FIELD_DEF(uint32_t, totalSeqLenQ);
TILING_DATA_FIELD_DEF(uint32_t, totalSeqLenK);
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLenQ);
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLenK);
TILING_DATA_FIELD_DEF(int32_t, targetGroupSize);
TILING_DATA_FIELD_DEF(float, alpha);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HstuForwardV2, HstuForwardV2TilingData)
}  // namespace optiling

#endif
