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

#ifndef MXREC_COMMON_H
#define MXREC_COMMON_H
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr int USE_QUEUE_NUM = 2;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int DATA_TYPE_INT64 = 1;
constexpr int FLOAT_ALIGNMENT = 8;
constexpr int DATA_TYPE_FLOAT32 = 0;

constexpr int MAX_INDICES_ONE_BLOCK = 1024;
constexpr int SMALL_TABLE_THRESHOLD = 64 * 1024 / sizeof(float);  // 64KB以内小表
constexpr int CACHE_MISS_MARK = -1;

enum class PoolingMode {
    SUM = 0,
    MEAN = 1,
    NONE = 2
};

struct Args {
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR hashIndices;
    GM_ADDR offsetPerKey;
    GM_ADDR rowsPerTable;
    GM_ADDR out;
    GM_ADDR tiling;
    GM_ADDR workspace;
};

#endif  // MXREC_COMMON_H
