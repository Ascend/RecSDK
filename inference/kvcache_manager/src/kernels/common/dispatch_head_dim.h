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

// Mirrors GPU DISPATCH_HEAD_DIM from gather_scatter_kernels.cu:38-64.
// Usage: DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, { /* use HEAD_DIM here */ })
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...) \
    do {                                           \
        switch (head_dim) {                        \
            case 64: {                             \
                constexpr size_t HEAD_DIM = 64;    \
                __VA_ARGS__;                       \
                break;                             \
            }                                      \
            case 128: {                            \
                constexpr size_t HEAD_DIM = 128;   \
                __VA_ARGS__;                       \
                break;                             \
            }                                      \
            case 256: {                            \
                constexpr size_t HEAD_DIM = 256;   \
                __VA_ARGS__;                       \
                break;                             \
            }                                      \
            case 512: {                            \
                constexpr size_t HEAD_DIM = 512;   \
                __VA_ARGS__;                       \
                break;                             \
            }                                      \
            default: { /* unsupported head_dim */  \
                break;                             \
            }                                      \
        }                                          \
    } while (0)
