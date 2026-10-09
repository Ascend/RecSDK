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
#include "append_paged_kvcache_kernel.h"
#include "../common/dispatch_head_dim.h"
#include "../common/fastdiv.h"
#include "../common/launch_geometry.h"

extern "C" __global__ __aicore__ void append_paged_kvcache(
    GM_ADDR k_data, GM_ADDR v_data, GM_ADDR indices, GM_ADDR indptr, uint32_t page_size, uint32_t stride_page,
    uint32_t stride_n, uint32_t stride_h, uint32_t head_dim, uint32_t num_heads, GM_ADDR append_key,
    GM_ADDR append_value, GM_ADDR batch_indices, GM_ADDR positions, GM_ADDR offsets, GM_ADDR nnz_dev,
    uint64_t append_k_stride_n, uint64_t append_k_stride_h, uint64_t append_v_stride_n, uint64_t append_v_stride_h,
    uint32_t fastdiv_m, uint32_t fastdiv_s, uint32_t fastdiv_a)
{
    using DType = bfloat16_t;
    using IdType = int32_t;

    __gm__ DType* kd = reinterpret_cast<__gm__ DType*>(k_data);
    __gm__ DType* vd = reinterpret_cast<__gm__ DType*>(v_data);
    const __gm__ IdType* idx = reinterpret_cast<const __gm__ IdType*>(indices);
    const __gm__ IdType* ip = reinterpret_cast<const __gm__ IdType*>(indptr);
    const __gm__ DType* ak = reinterpret_cast<const __gm__ DType*>(append_key);
    const __gm__ DType* av = reinterpret_cast<const __gm__ DType*>(append_value);
    const __gm__ IdType* bi = reinterpret_cast<const __gm__ IdType*>(batch_indices);
    const __gm__ IdType* pos = reinterpret_cast<const __gm__ IdType*>(positions);
    const __gm__ IdType* off = reinterpret_cast<const __gm__ IdType*>(offsets);
    const __gm__ IdType* nnz = reinterpret_cast<const __gm__ IdType*>(nnz_dev);

    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        constexpr uint32_t vec_size = kvcache::VecSize<DType, HEAD_DIM>();
        AscendC::Simt::VF_CALL<AppendPagedKVCache<HEAD_DIM, vec_size, DType, IdType>>(
            AscendC::Simt::Dim3{kvcache::BlockDimX<HEAD_DIM, vec_size>(num_heads), 1, 1}, kd, vd, idx, ip, page_size,
            stride_page, stride_n, stride_h, ak, av, bi, pos, off, nnz, static_cast<size_t>(append_k_stride_n),
            static_cast<size_t>(append_k_stride_h), static_cast<size_t>(append_v_stride_n),
            static_cast<size_t>(append_v_stride_h), fastdiv_m, fastdiv_s, fastdiv_a);
    });
}
