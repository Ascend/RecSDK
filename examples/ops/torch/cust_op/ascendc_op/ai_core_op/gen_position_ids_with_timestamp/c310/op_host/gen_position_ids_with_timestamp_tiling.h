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

#ifndef GEN_POSITION_IDS_WITH_TIMESTAMP_TILING_H
#define GEN_POSITION_IDS_WITH_TIMESTAMP_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(GenPositionIdsWithTimestampTilingData)
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(float, invTimeScale);
    TILING_DATA_FIELD_DEF(int64_t, blocksPerCore);
    TILING_DATA_FIELD_DEF(int32_t, remainderBlocks);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(GenPositionIdsWithTimestamp, GenPositionIdsWithTimestampTilingData)
}  // namespace optiling

#endif  // GEN_POSITION_IDS_WITH_TIMESTAMP_TILING_H