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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "acl/acl_rt.h"

#include "gather_scatter.h"
#include "kernels/common/launch_geometry.h"

#include "aclrtlaunch_scatter_paged_kvcache.h"
#include "aclrtlaunch_gather_paged_kvcache.h"
#include "aclrtlaunch_get_paged_batch_indices_positions.h"
#include "aclrtlaunch_append_paged_kvcache.h"

namespace kvcache {

namespace {

// Upper bound declared by LAUNCH_BOUND on the kernel side.
constexpr uint32_t kMaxThreadsPerBlock = 1024;
// Element width of the KV data the kernels operate on (bfloat16).
constexpr uint32_t kElemBytes = 2;

// Rejects head_dim values that DISPATCH_HEAD_DIM would otherwise drop into its
// default branch, where the kernel silently computes nothing. Also checks the
// launch geometry against LAUNCH_BOUND so an over-sized grid fails here rather
// than on the device. Uses the same kVecBytes/kThreadGroupSize the kernels
// derive vec_size from (launch_geometry.h), so this check can't silently
// drift from the geometry actually launched.
void check_launch_geometry(uint32_t head_dim, uint32_t num_heads, const char* entry)
{
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        throw std::runtime_error(std::string(entry) + ": unsupported head_dim " + std::to_string(head_dim) +
                                 ", expected one of {64, 128, 256, 512}");
    }
    const uint32_t vec_size = std::max(kVecBytes / kElemBytes, head_dim / kThreadGroupSize);
    const uint32_t threads = (head_dim / vec_size) * num_heads;
    if (threads > kMaxThreadsPerBlock) {
        throw std::runtime_error(std::string(entry) + ": launch geometry " + std::to_string(threads) +
                                 " exceeds LAUNCH_BOUND " + std::to_string(kMaxThreadsPerBlock) + " (head_dim=" +
                                 std::to_string(head_dim) + ", num_heads=" + std::to_string(num_heads) + ")");
    }
}

}  // namespace

// Host-side precomputation of fastdiv parameters for d == page_size.
// Mirrors GPU get_uint_fastdiv_msa from gather_scatter_kernels.cu:66-97.
static inline void get_uint_fastdiv_msa(uint32_t d, uint32_t& m, uint32_t& s, uint32_t& a)
{
    uint32_t p, nc, delta, q1, r1, q2, r2;
    a = 0;
    nc = unsigned(-1) - unsigned(-d) % d;
    p = 31;
    q1 = 0x80000000u / nc;
    r1 = 0x80000000u - q1 * nc;
    q2 = 0x7FFFFFFFu / d;
    r2 = 0x7FFFFFFFu - q2 * d;
    do {
        p++;
        if (r1 >= nc - r1) {
            q1 = 2 * q1 + 1;
            r1 = 2 * r1 - nc;
        } else {
            q1 = 2 * q1;
            r1 = 2 * r1;
        }
        if (r2 + 1 >= d - r2) {
            if (q2 >= 0x7FFFFFFFu)
                a = 1;
            q2 = 2 * q2 + 1;
            r2 = 2 * r2 + 1 - d;
        } else {
            if (q2 >= 0x80000000u)
                a = 1;
            q2 = 2 * q2;
            r2 = 2 * r2 + 1;
        }
        delta = d - 1 - r2;
    } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
    m = q2 + 1;
    s = p - 32;
}

void ScatterPagedKVCache(uint16_t* continuous_kv,  // input
                         int* page_ids, uint32_t num_heads, uint32_t head_dim, uint32_t page_size, uint32_t stride_page,
                         uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h, uint32_t num_pages,
                         uint16_t* kv_cache,  // output
                         uint32_t num_ctas, aclrtStream stream)
{
    check_launch_geometry(head_dim, num_heads, "ScatterPagedKVCache");

    // Precompute fastdiv parameters for page_size
    uint32_t fastdiv_m, fastdiv_s, fastdiv_a;
    get_uint_fastdiv_msa(page_size, fastdiv_m, fastdiv_s, fastdiv_a);

    ACLRT_LAUNCH_KERNEL(scatter_paged_kvcache)
    (num_ctas, stream, continuous_kv, page_ids, kv_cache, num_pages * page_size, page_size, stride_page, stride_k2v,
     stride_n, stride_h, head_dim, num_heads, fastdiv_m, fastdiv_s, fastdiv_a);
}

void GatherPagedKVCache(uint16_t* kv_cache,  // input
                        int* page_ids, uint32_t num_heads, uint32_t head_dim, uint32_t page_size, uint32_t stride_page,
                        uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h, uint32_t num_pages,
                        uint16_t* continuous_kv,  // output
                        uint32_t num_ctas, aclrtStream stream)
{
    check_launch_geometry(head_dim, num_heads, "GatherPagedKVCache");

    // Precompute fastdiv parameters for page_size
    uint32_t fastdiv_m, fastdiv_s, fastdiv_a;
    get_uint_fastdiv_msa(page_size, fastdiv_m, fastdiv_s, fastdiv_a);

    ACLRT_LAUNCH_KERNEL(gather_paged_kvcache)
    (num_ctas, stream, kv_cache, page_ids, continuous_kv, num_pages * page_size, page_size, stride_page, stride_k2v,
     stride_n, stride_h, head_dim, num_heads, fastdiv_m, fastdiv_s, fastdiv_a);
}
void GetPagedBatchIndicesPositions(int32_t batch_size, int32_t* append_indptr, int32_t* seq_lens_ptr,
                                   int32_t* batch_indices_ptr, int32_t* positions_ptr, aclrtStream stream)
{
    ACLRT_LAUNCH_KERNEL(get_paged_batch_indices_positions)
    (batch_size, stream, batch_size, append_indptr, seq_lens_ptr, batch_indices_ptr, positions_ptr);
}

void AppendPagedKVCache(uint16_t* k_data, uint16_t* v_data, int32_t* indices, int32_t* indptr, uint32_t num_heads,
                        uint32_t head_dim, uint32_t page_size, uint32_t stride_page, uint32_t stride_n,
                        uint32_t stride_h, uint16_t* append_key, uint16_t* append_value, int32_t* batch_indices,
                        int32_t* positions, int32_t* offsets, int32_t* nnz_dev, size_t append_k_stride_n,
                        size_t append_k_stride_h, size_t append_v_stride_n, size_t append_v_stride_h, uint32_t num_ctas,
                        aclrtStream stream)
{
    check_launch_geometry(head_dim, num_heads, "AppendPagedKVCache");

    uint32_t fastdiv_m, fastdiv_s, fastdiv_a;
    get_uint_fastdiv_msa(page_size, fastdiv_m, fastdiv_s, fastdiv_a);

    ACLRT_LAUNCH_KERNEL(append_paged_kvcache)
    (num_ctas, stream, k_data, v_data, indices, indptr, page_size, stride_page, stride_n, stride_h, head_dim, num_heads,
     append_key, append_value, batch_indices, positions, offsets, nnz_dev, static_cast<uint64_t>(append_k_stride_n),
     static_cast<uint64_t>(append_k_stride_h), static_cast<uint64_t>(append_v_stride_n),
     static_cast<uint64_t>(append_v_stride_h), fastdiv_m, fastdiv_s, fastdiv_a);
}

}  // namespace kvcache
