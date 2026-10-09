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

namespace kvcache {

// Shared by the append/gather/scatter kernel launch sites and by
// gather_scatter.cpp's host-side launch-geometry check, so the two can never
// derive a different vec_size/thread-count for the same (DType, head_dim).
constexpr uint32_t kVecBytes = 16;         // bytes moved per SIMT lane per memcpy
constexpr uint32_t kThreadGroupSize = 32;  // threads per lane-group; each head gets >= 1 full group

template <typename DType, uint32_t HEAD_DIM>
constexpr uint32_t VecSize()
{
    // std::max is unavailable in aicore context; use a constexpr ternary instead.
    constexpr uint32_t by_bytes = kVecBytes / sizeof(DType);
    constexpr uint32_t by_group = HEAD_DIM / kThreadGroupSize;
    return by_bytes > by_group ? by_bytes : by_group;
}

template <uint32_t HEAD_DIM, uint32_t VEC_SIZE>
constexpr uint32_t BlockDimX(uint32_t num_heads)
{
    return (HEAD_DIM / VEC_SIZE) * num_heads;
}

}  // namespace kvcache
