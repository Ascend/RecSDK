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
#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct AdamWSimdTilingData {
    uint32_t gradDim{};
    uint32_t valDim{};
    int32_t numRows{};
    float beta1{};
    float beta2{};
    float oneMinusBeta1{};
    float oneMinusBeta2{};
    float stepSize{};
    float invVHatDenom{};
    float decayFactor{};
    float eps{};
    int32_t needCoreNum{};
    uint32_t rowsPerGroup{};
    uint32_t gradType{};
    uint32_t weightType{};
};
#pragma pack(pop)
static_assert(sizeof(AdamWSimdTilingData) == 60U, "AdamWSimdTilingData packed layout");
