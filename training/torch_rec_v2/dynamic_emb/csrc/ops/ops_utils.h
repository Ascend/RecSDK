/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "../utils.h"

namespace ops_utils {

#define CASE_TYPE_USING_HINT(enum_type, type, HINT, ...)                       \
    case (enum_type): {                                                          \
        using HINT = type;                                                         \
        __VA_ARGS__;                                                             \
        break;                                                                     \
    }

#define INT_TYPE_DISPATCH(enum_type, HINT, ...)                                \
    do {                                                                        \
        switch (enum_type) {                                                    \
            CASE_TYPE_USING_HINT(dyn_emb::DataType::UInt64, uint64_t, HINT, __VA_ARGS__); \
            default:                                                                 \
                CASE_TYPE_USING_HINT(dyn_emb::DataType::Int64, int64_t, HINT, __VA_ARGS__); \
        }                                                                       \
    } while (0)

#define FLOAT_TYPE_DISPATCH(enum_type, HINT, ...)                              \
    do {                                                                        \
        switch (enum_type) {                                                    \
            CASE_TYPE_USING_HINT(dyn_emb::DataType::Float16, half, HINT, __VA_ARGS__); \
            CASE_TYPE_USING_HINT(dyn_emb::DataType::BFloat16, bfloat16_t,   HINT, __VA_ARGS__); \
            default:                                                                    \
                CASE_TYPE_USING_HINT(dyn_emb::DataType::Float32, float, HINT, __VA_ARGS__); \
        }                                                                       \
    } while (0)
}