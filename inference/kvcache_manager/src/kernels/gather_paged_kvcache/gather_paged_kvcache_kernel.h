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
#include "kernel_operator.h"
#include "../common/vec_dtypes.h"
#include "../common/fastdiv.h"

constexpr uint32_t MAX_THREADS_PER_BLOCK = 1024;

// VF wrapper for gather, inverse of scatter operation.
// Reads from paged KV cache (indexed by page_ids) and writes to continuous buffer.
// Thread mapping identical to scatter:
//   tid = GetThreadIdx<0>()  covers head_dim lanes * heads, BDX = HEAD_DIM/VEC_SIZE
//   tx = tid % BDX           head_dim lane index
//   head_idx = tid / BDX     head index
// Block id (GetBlockIdx) grid-strides over nnz entries.
template <uint32_t HEAD_DIM, uint32_t VEC_SIZE, typename DType, typename IdType>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void GatherPagedKV(
    const __gm__ DType* __restrict__ kv_cache, const __gm__ IdType* __restrict__ page_ids, uint32_t page_size,
    uint32_t stride_page, uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h, uint32_t nnz,
    __gm__ DType* continuous_kv, uint32_t num_heads, uint32_t fastdiv_m, uint32_t fastdiv_s, uint32_t fastdiv_a)
{
    constexpr uint32_t block_dim_x = HEAD_DIM / VEC_SIZE;
    uint32_t tid = AscendC::Simt::GetThreadIdx();
    uint32_t tx = tid % block_dim_x;
    uint32_t head_idx = tid / block_dim_x;
    uint32_t cta_id = AscendC::Simt::GetBlockIdx();
    uint32_t num_ctas = AscendC::Simt::GetBlockNum();

    const __gm__ DType* __restrict__ k_cache = kv_cache;
    const __gm__ DType* __restrict__ v_cache = kv_cache + stride_k2v;
    __gm__ DType* continuous_k = continuous_kv;
    __gm__ DType* continuous_v = continuous_kv + stride_k2v;

#pragma unroll 4
    for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
        uint32_t page_id_idx, entry_idx;
        divmod(i, page_size, fastdiv_m, fastdiv_s, fastdiv_a, page_id_idx, entry_idx);

        uint32_t inner_page_offset = head_idx * stride_h + entry_idx * stride_n + tx * VEC_SIZE;
        uint32_t src_offset = page_ids[page_id_idx] * stride_page + inner_page_offset;
        uint32_t dst_offset = page_id_idx * stride_page + inner_page_offset;

        vec_t<DType, VEC_SIZE>::memcpy(continuous_k + dst_offset, k_cache + src_offset);
        vec_t<DType, VEC_SIZE>::memcpy(continuous_v + dst_offset, v_cache + src_offset);
    }
}
