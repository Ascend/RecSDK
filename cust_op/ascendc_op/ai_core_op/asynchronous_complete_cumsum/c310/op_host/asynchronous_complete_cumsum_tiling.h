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

#ifndef ASYNCHRONOUS_COMPLETE_CUMSUM_H
#define ASYNCHRONOUS_COMPLETE_CUMSUM_H
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(AsynchronousCompleteCumsumTilingData)
    TILING_DATA_FIELD_DEF(int32_t, totalLength);
    TILING_DATA_FIELD_DEF(int32_t, totalBlocks);
    TILING_DATA_FIELD_DEF(int32_t, blocksPerCore);
    TILING_DATA_FIELD_DEF(int32_t, remainderBlocks);
    TILING_DATA_FIELD_DEF(bool, isInt32);
    TILING_DATA_FIELD_DEF(bool, isSmall);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(AsynchronousCompleteCumsum, AsynchronousCompleteCumsumTilingData)
}
#endif // ASYNCHRONOUS_COMPLETE_CUMSUM_H
