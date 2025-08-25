/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef LCCL_RMA_SWAP_MULTI_TABLES_H
#define LCCL_RMA_SWAP_MULTI_TABLES_H


#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(RmaSwapMultiTablesTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, tableNum);
    TILING_DATA_FIELD_DEF(uint64_t, tableLength);
    TILING_DATA_FIELD_DEF(uint64_t, shmSwapIn);
    TILING_DATA_FIELD_DEF(uint64_t, shmSwapOut);
    TILING_DATA_FIELD_DEF(uint64_t, updateLen);
    TILING_DATA_FIELD_DEF(uint32_t, dimNum);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, dimValue);   // swap_out_len, emb_dim（只有val，不包含slot）
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(RmaSwapMultiTables, RmaSwapMultiTablesTilingData)
}


#endif // LCCL_RMA_SWAP_MULTI_TABLES_H

