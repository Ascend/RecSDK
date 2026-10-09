/* ==============================================================================
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"

namespace kvcache {

// Host-side launch wrappers for the paged KVCache AscendC kernels.
//
// Shared contract for all four entries:
//   - kv_cache / continuous_kv point to bfloat16 data; the uint16_t* type is a
//     storage-only view and does not imply integer semantics.
//   - page_ids is an int32 buffer of num_pages entries; every id must be a valid
//     page index of the registered cache table. Ids are not bounds-checked on the
//     device side, so callers are responsible for their validity.
//   - head_dim must be one of {64, 128, 256, 512}; other values are rejected by
//     the host-side check in each wrapper.
//   - stream must outlive the launched kernel; these calls are asynchronous.
//   - num_heads (Scatter/Gather) and batch_size (GetPagedBatchIndicesPositions)
//     are kept for signature parity with !3206's hand-written forward
//     declarations and call sites; the kernel bodies don't read them. Dropping
//     them needs a synchronized signature change across both PRs, tracked as
//     later cleanup once the series is merged.

// Scatter a contiguous KV buffer into the paged cache table.
void ScatterPagedKVCache(uint16_t* continuous_kv,  // input
                         int* page_ids, uint32_t num_heads, uint32_t head_dim, uint32_t page_size, uint32_t stride_page,
                         uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h, uint32_t num_pages,
                         uint16_t* kv_cache,  // output
                         uint32_t num_ctas, aclrtStream stream);

// Gather pages from the paged cache table into a contiguous KV buffer.
void GatherPagedKVCache(uint16_t* kv_cache,  // input
                        int* page_ids, uint32_t num_heads, uint32_t head_dim, uint32_t page_size, uint32_t stride_page,
                        uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h, uint32_t num_pages,
                        uint16_t* continuous_kv,  // output
                        uint32_t num_ctas, aclrtStream stream);

// Derive per-token batch indices and in-sequence positions from append_indptr.
// batch_size is not bounds-checked against append_indptr/seq_lens_ptr inside
// the kernel; callers must size those buffers to at least batch_size entries.
void GetPagedBatchIndicesPositions(int32_t batch_size, int32_t* append_indptr, int32_t* seq_lens_ptr,
                                   int32_t* batch_indices_ptr, int32_t* positions_ptr, aclrtStream stream);

// Append freshly computed K/V into the paged cache table.
void AppendPagedKVCache(uint16_t* k_data, uint16_t* v_data, int32_t* indices, int32_t* indptr, uint32_t num_heads,
                        uint32_t head_dim, uint32_t page_size, uint32_t stride_page, uint32_t stride_n,
                        uint32_t stride_h, uint16_t* append_key, uint16_t* append_value, int32_t* batch_indices,
                        int32_t* positions, int32_t* offsets, int32_t* nnz_dev, size_t append_k_stride_n,
                        size_t append_k_stride_h, size_t append_v_stride_n, size_t append_v_stride_h, uint32_t num_ctas,
                        aclrtStream stream);

}  // namespace kvcache
