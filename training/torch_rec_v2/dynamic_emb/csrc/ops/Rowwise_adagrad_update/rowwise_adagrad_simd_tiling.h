/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct RowwiseAdagradSimdTilingData {
    uint32_t gradDim{};
    uint32_t valDim{};
    int32_t numRows{};
    float lr{};
    float eps{};
    int32_t needCoreNum{};
    uint32_t rowsPerGroup{};
    uint32_t gradType{};
    uint32_t weightType{};
};
#pragma pack(pop)
static_assert(sizeof(RowwiseAdagradSimdTilingData) == 36U, "RowwiseAdagradSimdTilingData packed layout");
