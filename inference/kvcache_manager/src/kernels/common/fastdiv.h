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

// Device-side fast division using precomputed magic numbers.
// Used in both scatter and gather kernels to compute page_id_idx and entry_idx.
__aicore__ inline void divmod(uint32_t n, uint32_t d, uint32_t m, uint32_t s, uint32_t a, uint32_t& q, uint32_t& r)
{
    if (d == 1) {
        q = n;
    } else {
        q = static_cast<uint32_t>((static_cast<uint64_t>(m) * static_cast<uint64_t>(n)) >> 32);
        q += a * n;
        q >>= s;
    }
    r = n - q * d;
}
