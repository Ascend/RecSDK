/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#pragma once

#include <cstdint>

// Primary template is intentionally declared without a definition: instantiating
// it for an unsupported element type fails at link time rather than silently
// compiling. Only bfloat16 is supported today.
template <typename T, size_t vec_size>
struct vec_t {
    static __aicore__ inline void memcpy(__gm__ T* dst, const __gm__ T* src);
};

// Callers derive vec_size as max(16 / sizeof(DType), head_dim / 32); with
// bfloat16 and head_dim in {64, 128, 256, 512} that yields 8 or 16 only, so the
// explicit specialization below plus the generic one cover every instantiation.

// bfloat16 × 8
template <>
struct vec_t<bfloat16_t, 8> {
    static __aicore__ inline void memcpy(__gm__ bfloat16_t* dst, const __gm__ bfloat16_t* src)
    {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        dst[4] = src[4];
        dst[5] = src[5];
        dst[6] = src[6];
        dst[7] = src[7];
    }
};

// bfloat16 × N where N % 8 == 0
template <size_t vec_size>
struct vec_t<bfloat16_t, vec_size> {
    static_assert(vec_size % 8 == 0, "Invalid vector size");
    static __aicore__ inline void memcpy(__gm__ bfloat16_t* dst, const __gm__ bfloat16_t* src)
    {
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            dst[i] = src[i];
        }
    }
};
