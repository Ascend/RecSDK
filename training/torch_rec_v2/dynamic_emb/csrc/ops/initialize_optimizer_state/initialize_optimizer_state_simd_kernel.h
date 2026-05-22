/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#pragma once

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "device_utils.h"

using namespace AscendC;

namespace dyn_emb {

template <typename V>
__global__ __vector__ void initialize_optimizer_state_simd_kernel(uint32_t former_num, uint64_t former_core_move_num,
                                                                  uint64_t tail_core_move_num, uint32_t tile_size,
                                                                  uint32_t num_tiles, uint64_t n, int emb_dim,
                                                                  __gm__ V* __gm__* value_ptrs, __gm__ bool* founds,
                                                                  int optstate_dim, float init_value)
{
    int64_t cur_block_idx = GetBlockIdx();
    uint64_t core_start_idx = 0;
    uint64_t core_proc_count = 0;
    if (cur_block_idx < former_num) {
        core_start_idx = cur_block_idx * former_core_move_num;
        core_proc_count = former_core_move_num;
    } else {
        core_start_idx = former_num * former_core_move_num + (cur_block_idx - former_num) * tail_core_move_num;
        core_proc_count = tail_core_move_num;
    }

    TPipe pipe;
    // 仅有搬出流程，1片内存就行，无需double_buffer
    TQue<TPosition::VECOUT, 1> que;
    pipe.InitBuffer(que, 1, tile_size);

    auto local_tensor = que.AllocTensor<V>();
    V scalar = SimdTypeConvertFunc<V, float>::convert(init_value);
    Duplicate(local_tensor, scalar, tile_size);
    que.EnQue(local_tensor);
    local_tensor = que.DeQue<V>();

    GlobalTensor<V> dst_gm;
    for (uint64_t i = core_start_idx; i < core_start_idx + core_proc_count; i++) {
        bool found = founds[i];
        __gm__ V* dst_ptr = value_ptrs[i];
        if (found || dst_ptr == nullptr) {
            continue;
        }

        __gm__ V* dst = dst_ptr + emb_dim;
        for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            uint32_t current_tile_size =
                (tile_idx == num_tiles - 1) ? (optstate_dim - tile_idx * tile_size) : tile_size;
            DataCopyExtParams copy_params{1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};

            dst_gm.SetGlobalBuffer(dst + tile_idx * tile_size);
            AscendC::DataCopyPad(dst_gm, local_tensor, copy_params);
        }
    }
    que.FreeTensor(local_tensor);
}

}  // namespace dyn_emb