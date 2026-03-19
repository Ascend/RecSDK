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

#ifndef INIT_ADDRESS_LOOKUP_TILING_H
#define INIT_ADDRESS_LOOKUP_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InitAddressLookupTilingData)
TILING_DATA_FIELD_DEF(int64_t, num_tables);  // 嵌入表的数量
TILING_DATA_FIELD_DEF(int64_t, total_rows);  // 总行数（buffer_offsets的最后一个元素）
TILING_DATA_FIELD_DEF(int64_t, core_num);    // 实际参与计算的核数
TILING_DATA_FIELD_DEF(int64_t, lut_size);    // 向量查找表大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InitAddressLookup, InitAddressLookupTilingData)
}  // namespace optiling
#endif  // INIT_ADDRESS_LOOKUP_TILING_H
