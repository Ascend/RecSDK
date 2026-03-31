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
#ifndef CONCAT_JAGGED_TENSOR_TILING_H
#define CONCAT_JAGGED_TENSOR_TILING_H

#include "register/tilingdata_base.h"

constexpr uint32_t MAX_SLICE_SIZE = 2 * 1024;
namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatJaggedTensorTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, jtNum);
    TILING_DATA_FIELD_DEF(uint32_t, inputColSize);
    TILING_DATA_FIELD_DEF(uint32_t, ubMaxLength);
    TILING_DATA_FIELD_DEF(uint32_t, formerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailCore);
    TILING_DATA_FIELD_DEF(uint32_t, batchNumInTail);
    TILING_DATA_FIELD_DEF(uint32_t, batchNumInFormer);

    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SLICE_SIZE, inputOffsetBegin)
    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SLICE_SIZE, sliceSize)
    TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_SLICE_SIZE, outputOffsetBegin)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConcatJaggedTensor, ConcatJaggedTensorTilingData)
}  // namespace optiling
#endif