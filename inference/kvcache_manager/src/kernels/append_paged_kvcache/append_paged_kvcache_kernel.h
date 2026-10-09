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

// Mirrors GPU AppendPagedKVCacheKernel (paged_kvcache_ops_kernel.cu:106-145).
// Thread mapping: flattened 1D (same as scatter/gather).
//   tid = GetThreadIdx()
//   tx = tid % block_dim_x   (head_dim lane)
//   head_idx = tid / block_dim_x  (head index)
// Grid-strides over nnz entries (number of new tokens to append).
//
// For each new token i:
//   divmod(positions[i], page_size) -> page_iter, entry_idx
//   elem_offset = indices[indptr[batch_indices[i]] + page_iter] * stride_page
//                 + head_idx * stride_h + entry_idx * stride_n + tx * VEC_SIZE
//   k_data[elem_offset] = append_key[(i + offsets[batch_indices[i]]) * append_k_stride_n
//                                     + head_idx * append_k_stride_h + tx * VEC_SIZE]
//   v_data[elem_offset] = append_value[similar]
template <uint32_t HEAD_DIM, uint32_t VEC_SIZE, typename DType, typename IdType>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void AppendPagedKVCache(
    __gm__ DType* k_data, __gm__ DType* v_data, const __gm__ IdType* __restrict__ indices,
    const __gm__ IdType* __restrict__ indptr, uint32_t page_size, uint32_t stride_page, uint32_t stride_n,
    uint32_t stride_h, const __gm__ DType* __restrict__ append_key, const __gm__ DType* __restrict__ append_value,
    const __gm__ IdType* __restrict__ batch_indices, const __gm__ IdType* __restrict__ positions,
    const __gm__ IdType* __restrict__ offsets, const __gm__ IdType* __restrict__ nnz_dev, size_t append_k_stride_n,
    size_t append_k_stride_h, size_t append_v_stride_n, size_t append_v_stride_h, uint32_t fastdiv_m,
    uint32_t fastdiv_s, uint32_t fastdiv_a)
{
    constexpr uint32_t block_dim_x = HEAD_DIM / VEC_SIZE;
    uint32_t tid = AscendC::Simt::GetThreadIdx();
    uint32_t tx = tid % block_dim_x;
    uint32_t head_idx = tid / block_dim_x;
    uint32_t cta_id = AscendC::Simt::GetBlockIdx();
    uint32_t num_ctas = AscendC::Simt::GetBlockNum();

    uint32_t nnz = static_cast<uint32_t>(nnz_dev[0]);

    for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
        uint32_t page_iter, entry_idx;
        divmod(static_cast<uint32_t>(positions[i]), page_size, fastdiv_m, fastdiv_s, fastdiv_a, page_iter, entry_idx);

        size_t elem_offset = static_cast<size_t>(indices[indptr[batch_indices[i]] + page_iter]) * stride_page +
                             head_idx * stride_h + entry_idx * stride_n + tx * VEC_SIZE;

        size_t k_src_offset = static_cast<size_t>(i + offsets[batch_indices[i]]) * append_k_stride_n +
                              head_idx * append_k_stride_h + tx * VEC_SIZE;
        size_t v_src_offset = static_cast<size_t>(i + offsets[batch_indices[i]]) * append_v_stride_n +
                              head_idx * append_v_stride_h + tx * VEC_SIZE;

        vec_t<DType, VEC_SIZE>::memcpy(k_data + elem_offset, append_key + k_src_offset);
        vec_t<DType, VEC_SIZE>::memcpy(v_data + elem_offset, append_value + v_src_offset);
    }
}
