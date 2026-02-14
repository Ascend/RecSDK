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

#ifndef MULTISLICE_CONCAT_TILING_H
#define MULTISLICE_CONCAT_TILING_H

#include "register/tilingdata_base.h"

constexpr uint16_t MAX_CONCAT_TENSOR_NUM = 256;
constexpr uint16_t MAX_SLICE_NUM = 3600;

namespace optiling {
BEGIN_TILING_DATA_DEF(MultisliceConcatTilingData)
TILING_DATA_FIELD_DEF(int64_t, colSize);
TILING_DATA_FIELD_DEF(int64_t, formerCore);
TILING_DATA_FIELD_DEF(int64_t, tailCore);
TILING_DATA_FIELD_DEF(uint16_t, batchNumInFormer);
TILING_DATA_FIELD_DEF(uint16_t, batchNumInTail);
TILING_DATA_FIELD_DEF(uint16_t, maxProColumnNum);
TILING_DATA_FIELD_DEF(uint16_t, concatNum);

TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CONCAT_TENSOR_NUM, concatSize);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_SLICE_NUM, sliceBegin);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_SLICE_NUM, sliceLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultisliceConcat, MultisliceConcatTilingData)
}  // namespace optiling
#endif
