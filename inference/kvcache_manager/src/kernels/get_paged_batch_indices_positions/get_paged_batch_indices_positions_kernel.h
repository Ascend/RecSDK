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

constexpr int32_t BATCH_INDICES_BLOCK_SIZE = 1024;

__simt_vf__ __aicore__ LAUNCH_BOUND(BATCH_INDICES_BLOCK_SIZE) inline void GetPagedBatchIndicesPositions(
    int32_t batch_size, const __gm__ int32_t* __restrict__ append_indptr,
    const __gm__ int32_t* __restrict__ seq_lens_ptr, __gm__ int32_t* __restrict__ batch_indices_ptr,
    __gm__ int32_t* __restrict__ positions_ptr)
{
    int32_t tx = static_cast<int32_t>(AscendC::Simt::GetThreadIdx());
    int32_t seq_idx = static_cast<int32_t>(AscendC::Simt::GetBlockIdx());
    // Host launcher always sizes the grid to batch_size (gather_scatter.cpp),
    // so this never triggers today; kept as a guard against a future caller
    // launching a larger grid.
    if (seq_idx >= batch_size) {
        return;
    }

    int32_t seq_start = append_indptr[seq_idx];
    int32_t total_seq_len = seq_lens_ptr[seq_idx];
    int32_t append_per_seq = append_indptr[seq_idx + 1] - seq_start;

    int32_t pos_start = total_seq_len - append_per_seq;
#pragma unroll 4
    for (int32_t i = tx; i < append_per_seq; i += blockDim.x) {
        batch_indices_ptr[seq_start + i] = seq_idx;
        positions_ptr[seq_start + i] = pos_start + i;
    }
}
