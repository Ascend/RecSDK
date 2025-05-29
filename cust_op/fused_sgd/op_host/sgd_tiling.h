/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef SGD_TILING_H
#define SGD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SgdTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, tableSize);
TILING_DATA_FIELD_DEF(uint32_t, dimSize);
TILING_DATA_FIELD_DEF(uint32_t, actualCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, ubFreeSize);
TILING_DATA_FIELD_DEF(uint32_t, splitNextCoreProcBs);
TILING_DATA_FIELD_DEF(uint32_t, splitPrevCoreProcBs);
TILING_DATA_FIELD_DEF(uint32_t, splitCoreIndex);
TILING_DATA_FIELD_DEF(float, weightDecay);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Sgd, SgdTilingData)
}

#endif