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
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include <simt_api/common_functions.h>
#include "../../../cust_op/hkv/include/types.h"
#include "acl/acl.h"

using namespace AscendC;

template <class V>
__global__ __vector__ void load_from_pointer_kernel_hybrid(uint32_t former_num,
                                             uint64_t former_core_move_num,
                                             uint64_t tail_core_move_num,
                                             uint32_t tile_size,
                                             uint32_t num_tiles, uint32_t dim,
                                             __gm__ V* values, uint64_t n,
                                             __gm__ V* __gm__* d_src_values) {
  uint64_t cur_block_idx = GetBlockIdx();
  uint64_t core_start_idx = 0;
  uint64_t core_move_count = 0;
  constexpr uint32_t BUFFER_NUM = 2;
  if (cur_block_idx < former_num) {
    core_start_idx = cur_block_idx * former_core_move_num;
    core_move_count = former_core_move_num;
  } else {
    core_start_idx = former_num * former_core_move_num +
                     (cur_block_idx - former_num) * tail_core_move_num;
    core_move_count = tail_core_move_num;
  }

  TPipe pipe;
  TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> move_queue;
  pipe.InitBuffer(move_queue, BUFFER_NUM, tile_size * sizeof(V));

  GlobalTensor<V> src_values_gm;
  GlobalTensor<V> dst_values_gm;
  LocalTensor<V> move_local;
  DataCopyPadExtParams<V> pad_params{true, 0, 0, 0};

  dst_values_gm.SetGlobalBuffer(values);

  for (uint64_t i = core_start_idx; i < core_start_idx + core_move_count; i++) {
    __gm__ V* src_value = d_src_values[i];
    if (src_value == nullptr) {
      continue;
    }

    uint64_t dst_offset = i * dim;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      uint32_t current_tile_size = (tile_idx == num_tiles - 1)
                                       ? (dim - tile_idx * tile_size)
                                       : tile_size;
      DataCopyExtParams copy_params{
          1, static_cast<uint32_t>(current_tile_size * sizeof(V)), 0, 0, 0};

      move_queue.AllocTensor<V>(move_local);

      src_values_gm.SetGlobalBuffer(src_value + tile_idx * tile_size);
      DataCopyPad(move_local, src_values_gm, copy_params, pad_params);
      move_queue.EnQue<V>(move_local);
      move_queue.DeQue<V>(move_local);

      DataCopyPad(dst_values_gm[dst_offset + tile_idx * tile_size], move_local,
                  copy_params);

      move_queue.FreeTensor(move_local);
    }
  }
}
