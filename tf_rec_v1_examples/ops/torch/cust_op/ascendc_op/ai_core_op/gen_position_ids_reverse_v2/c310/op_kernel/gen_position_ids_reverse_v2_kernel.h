/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

/**
 * A5（Ascend 950 / c310）SIMT：与 CUDA `gen_position_ids_reverse_v2_kernel` 语义一致。
 * 每个 batch 一条 VF：pre 段为 pre_len, pre_len-1, …, 1；post 段为 0（interleaved 未支持）。
 */
#ifndef GEN_POSITION_IDS_REVERSE_V2_KERNEL_H
#define GEN_POSITION_IDS_REVERSE_V2_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

namespace GenPositionIdsReverseV2 {

    static constexpr int32_t SIMT_BLOCK_DIM = 256;

    struct Args {
        GM_ADDR seqlen;
        GM_ADDR seqlenOffsets;
        GM_ADDR rspos;
        GM_ADDR positionIds;
        GM_ADDR workspace;
        GM_ADDR tiling;
    };

    __simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void GenPositionIdsReverseV2Simt(
            __gm__ int32_t* __restrict__ seqlen_gm, __gm__ int32_t* __restrict__ seqlen_offsets_gm,
    __gm__ int32_t* __restrict__ rspos_gm, __gm__ int32_t* __restrict__ position_ids_gm,
    int32_t batch_idx)
{
    int32_t thread_idx = AscendC::Simt::GetThreadIdx<0>();

    int32_t seq_len = seqlen_gm[batch_idx];
    int32_t rs_pos = rspos_gm[batch_idx];

    int32_t pre_len = rs_pos;
    int32_t post_len = seq_len - rs_pos;
    int32_t total_len = pre_len + post_len;

    int32_t output_start = seqlen_offsets_gm[batch_idx];

    for (int32_t offset = thread_idx; offset < total_len; offset += SIMT_BLOCK_DIM) {
    int32_t position_value = 0;
    if (offset < pre_len) {
    position_value = pre_len - offset;
}
position_ids_gm[output_start + offset] = position_value;
}
}

class GenPositionIdsReverseV2Kernel {
public:
    __aicore__ inline GenPositionIdsReverseV2Kernel(Args& args)
    {
        GET_TILING_DATA(td, args.tiling);
        batch_size = td.batchSize;
        blocks_per_core = td.blocksPerCore;
        remainder_blocks = td.remainderBlocks;

        seqlen_gm = reinterpret_cast<__gm__ int32_t*>(args.seqlen);
        seqlen_offsets_gm = reinterpret_cast<__gm__ int32_t*>(args.seqlenOffsets);
        rspos_gm = reinterpret_cast<__gm__ int32_t*>(args.rspos);
        position_ids_gm = reinterpret_cast<__gm__ int32_t*>(args.positionIds);
    }

    __aicore__ inline void Compute()
    {
        if (batch_size <= 0) {
            return;
        }

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
            if (batch_idx >= batch_size) {
                continue;
            }
            AscendC::Simt::VF_CALL<GenPositionIdsReverseV2Simt>(
                    AscendC::Simt::Dim3{static_cast<uint32_t>(SIMT_BLOCK_DIM), 1, 1},
                    seqlen_gm, seqlen_offsets_gm, rspos_gm, position_ids_gm, static_cast<int32_t>(batch_idx));
        }
    }

private:
    __gm__ int32_t* seqlen_gm{nullptr};
    __gm__ int32_t* seqlen_offsets_gm{nullptr};
    __gm__ int32_t* rspos_gm{nullptr};
    __gm__ int32_t* position_ids_gm{nullptr};

    int64_t batch_size{0};
    int64_t blocks_per_core{0};
    int32_t remainder_blocks{0};
};

}  // namespace GenPositionIdsReverseV2

#endif  // GEN_POSITION_IDS_REVERSE_V2_KERNEL_H