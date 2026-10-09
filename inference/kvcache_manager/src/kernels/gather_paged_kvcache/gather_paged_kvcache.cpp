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

#include <cmath>

#include "kernel_operator.h"
#include "gather_paged_kvcache_kernel.h"
#include "../common/dispatch_head_dim.h"
#include "../common/fastdiv.h"
#include "../common/launch_geometry.h"

extern "C" __global__ __aicore__ void gather_paged_kvcache(GM_ADDR kv_cache, GM_ADDR page_ids, GM_ADDR continuous_kv,
                                                           uint32_t nnz, uint32_t page_size, uint32_t stride_page,
                                                           uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h,
                                                           uint32_t head_dim, uint32_t num_heads, uint32_t fastdiv_m,
                                                           uint32_t fastdiv_s, uint32_t fastdiv_a)
{
    using DType = bfloat16_t;
    using IdType = int32_t;

    const __gm__ DType* kvc = reinterpret_cast<const __gm__ DType*>(kv_cache);
    const __gm__ IdType* pids = reinterpret_cast<const __gm__ IdType*>(page_ids);
    __gm__ DType* ckv = reinterpret_cast<__gm__ DType*>(continuous_kv);

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        constexpr uint32_t vec_size = kvcache::VecSize<DType, HEAD_DIM>();
        AscendC::Simt::VF_CALL<GatherPagedKV<HEAD_DIM, vec_size, DType, IdType>>(
            AscendC::Simt::Dim3{kvcache::BlockDimX<HEAD_DIM, vec_size>(num_heads), 1, 1}, kvc, pids, page_size,
            stride_page, stride_k2v, stride_n, stride_h, nnz, ckv, num_heads, fastdiv_m, fastdiv_s, fastdiv_a);
    });
}
