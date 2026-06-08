/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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
struct PoolingEmbeddingsSimdTilingData {
    int32_t combiner{};
    int32_t totalDims{};
    int32_t accumDims{};
    int32_t evSize{};
    int32_t numVec{};
    int32_t batchSize{};
    int32_t srcNumRows{};
    int32_t inverseLen{};
    int32_t needCoreNum{};
    int64_t offsetBase{};
    uint32_t offsetType{};
    uint32_t srcType{};
    uint32_t dstType{};
};
#pragma pack(pop)
static_assert(sizeof(PoolingEmbeddingsSimdTilingData) == 56U, "PoolingEmbeddingsSimdTilingData packed layout");
