/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LCCL_ALL_USS_TILING_H
#define LCCL_ALL_USS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(LcclAllUssTilingData)
    TILING_DATA_FIELD_DEF(int64_t, rank);
    TILING_DATA_FIELD_DEF(int64_t, rankSize);
    TILING_DATA_FIELD_DEF(int64_t, magic);
    TILING_DATA_FIELD_DEF(int64_t, deterministic);
    TILING_DATA_FIELD_DEF(int64_t, dim);
    TILING_DATA_FIELD_DEF(int64_t, outShape);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(LcclAllUss, LcclAllUssTilingData)
}

#endif