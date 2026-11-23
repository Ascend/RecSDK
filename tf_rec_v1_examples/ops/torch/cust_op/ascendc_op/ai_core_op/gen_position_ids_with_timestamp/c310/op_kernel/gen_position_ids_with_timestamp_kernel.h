/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

/**
 * A5（Ascend 950）SIMT：AscendC::Simt::VF_CALL + __simt_vf__。
 * 变量命名与 test_gen_position_ids_with_timestamp.py 中 compute_position_ids_golden_v1 一致。
 *
 * SIMT 子核内的 time_diff / log_pos 等为线程私有的标量，由编译器放入寄存器（或必要时溢出到栈），
 * 不是 AscendC TPipe 的 UB 缓冲；无需 InitBuffer，与 CUDA __global__ 里局部 float 同理。
 */
#ifndef MXREC_GEN_POSITION_IDS_WITH_TIMESTAMP_KERNEL_H
#define MXREC_GEN_POSITION_IDS_WITH_TIMESTAMP_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace GenPositionIdsWithTimestamp {

// 与 golden：inv_log_base = 10.4920586873，max_position_id = 1024
static constexpr float inv_log_base = 10.4920586873f;
static constexpr int32_t simt_block_dim = 256;
static constexpr int32_t max_position_id = 1024;

struct Args {
    GM_ADDR seqlen;
    GM_ADDR seqlenOffsets;
    GM_ADDR timestamps;
    GM_ADDR positionIds;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

// golden: time_diff = (t_end - timestamp) / time_scale；此处 inv_time_scale = 1/time_scale
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void GenPositionIdsWithTimestampSimt(
    __gm__ int32_t* __restrict__ timestamps_gm, __gm__ int32_t* __restrict__ position_ids_gm, int32_t start_pos,
    int32_t seq_len, int32_t t_end, float inv_time_scale)
{
    int32_t thread_idx = AscendC::Simt::GetThreadIdx<0>();
    for (int32_t offset = thread_idx; offset < seq_len; offset += simt_block_dim) {
        int32_t timestamp_idx = start_pos + offset;
        int32_t timestamp = timestamps_gm[timestamp_idx];
        float time_diff = (static_cast<float>(t_end - timestamp)) * inv_time_scale;
        float log_pos = logf(1.0f + time_diff) * inv_log_base;
        int32_t position_id = static_cast<int32_t>(floorf(log_pos));
        if (position_id < 0) {
            position_id = 0;
        }
        if (position_id > max_position_id) {
            position_id = max_position_id;
        }
        position_ids_gm[timestamp_idx] = position_id;
    }
}

class GenPositionIdsWithTimestampKernel {
public:
    __aicore__ inline GenPositionIdsWithTimestampKernel(Args& args)
    {
        GET_TILING_DATA(td, args.tiling);
        batch_size = td.batchSize;
        inv_time_scale = td.invTimeScale;
        blocks_per_core = td.blocksPerCore;
        remainder_blocks = td.remainderBlocks;

        seqlen_gt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.seqlen), batch_size);
        seqlen_offsets_gt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.seqlenOffsets), batch_size + 1);
        timestamps_gm = reinterpret_cast<__gm__ int32_t*>(args.timestamps);
        position_ids_gm = reinterpret_cast<__gm__ int32_t*>(args.positionIds);
    }

    __aicore__ inline void Compute()
    {
        if (batch_size <= 0) {
            return;
        }

        // 与 Host / lengths_index：前 remainder_blocks 个核多 1 个 logical_block；logical_block == batch_idx
        int64_t core_idx = static_cast<int64_t>(GetBlockIdx());
        int64_t block_count;
        int64_t block_start;
        if (core_idx < remainder_blocks) {
            block_count = blocks_per_core + 1;
            block_start = core_idx * block_count;
        } else {
            block_count = blocks_per_core;
            block_start = remainder_blocks * (blocks_per_core + 1) + (core_idx - remainder_blocks) * blocks_per_core;
        }
        if (block_count <= 0) {
            return;
        }

        for (int64_t logical_block = block_start; logical_block < block_start + block_count; ++logical_block) {
            int64_t batch_idx = logical_block;
            int32_t seq_len = seqlen_gt.GetValue(batch_idx);
            int32_t start_pos = seqlen_offsets_gt.GetValue(batch_idx);
            int32_t end_pos = seqlen_offsets_gt.GetValue(batch_idx + 1);
            int32_t actual_len = end_pos - start_pos;
            if (actual_len != seq_len) {
                continue;
            }
            if (seq_len <= 0) {
                continue;
            }
            int32_t t_end = timestamps_gm[end_pos - 1];
            AscendC::Simt::VF_CALL<GenPositionIdsWithTimestampSimt>(
                AscendC::Simt::Dim3{static_cast<uint32_t>(simt_block_dim), 1, 1}, timestamps_gm, position_ids_gm,
                start_pos, seq_len, t_end, inv_time_scale);
        }
    }

private:
    GlobalTensor<int32_t> seqlen_gt;
    GlobalTensor<int32_t> seqlen_offsets_gt;
    __gm__ int32_t* timestamps_gm{nullptr};
    __gm__ int32_t* position_ids_gm{nullptr};

    int64_t batch_size{0};
    int64_t blocks_per_core{0};
    int32_t remainder_blocks{0};
    float inv_time_scale{0.0f};
};

}  // namespace GenPositionIdsWithTimestamp

#endif  // MXREC_GEN_POSITION_IDS_WITH_TIMESTAMP_KERNEL_H