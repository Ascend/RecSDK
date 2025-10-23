/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef EMBEDDING_UPDATE_BY_ADDRESS_TILING_H
#define EMBEDDING_UPDATE_BY_ADDRESS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(TilingData2)
    TILING_DATA_FIELD_DEF(int32_t, update_dim);
    TILING_DATA_FIELD_DEF(int32_t, addr_nums);
    TILING_DATA_FIELD_DEF(int32_t, embedding_type);
    TILING_DATA_FIELD_DEF(int32_t, update_type);
    TILING_DATA_FIELD_DEF(int32_t, addr_per_loop);
    TILING_DATA_FIELD_DEF(int32_t, ping_pong_num);
    TILING_DATA_FIELD_DEF(int32_t, type_size);
    TILING_DATA_FIELD_DEF(int32_t, input_dim_aligned);

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(EmbeddingUpdateByAddress, TilingData2)
}
#endif // EMBEDDING_UPDATE_BY_ADDRESS_TILING_H