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

#include "../ops_utils.h"

namespace dyn_emb_adamw_simd {

__aicore__ inline bool IsSupportedSimdDtype(dyn_emb::DataType dtype)
{
    return dtype == dyn_emb::DataType::Float32 || dtype == dyn_emb::DataType::Float16 ||
           dtype == dyn_emb::DataType::BFloat16;
}

#define ADAMW_SIMD_FLOAT_TYPE_DISPATCH(enum_type, HINT, ...)                                  \
    do {                                                                                      \
        switch (enum_type) {                                                                  \
            CASE_TYPE_USING_HINT(dyn_emb::DataType::Float16, half, HINT, __VA_ARGS__);        \
            CASE_TYPE_USING_HINT(dyn_emb::DataType::BFloat16, bfloat16_t, HINT, __VA_ARGS__); \
            CASE_TYPE_USING_HINT(dyn_emb::DataType::Float32, float, HINT, __VA_ARGS__);       \
            default:                                                                          \
                break;                                                                        \
        }                                                                                     \
    } while (0)

}  // namespace dyn_emb_adamw_simd
