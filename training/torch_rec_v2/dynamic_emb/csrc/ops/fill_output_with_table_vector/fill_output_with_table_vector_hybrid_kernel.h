/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_FILL_OUTPUT_WITH_TABLE_VECTOR_HYBRID_KERNEL_H_
#define ASCENDC_FILL_OUTPUT_WITH_TABLE_VECTOR_HYBRID_KERNEL_H_

#include "kernel_operator.h"
#include "device_utils.h"
#include "table_vector.h"

using namespace AscendC;

namespace dyn_emb {

template <typename T, typename EmbeddingGenerator>
__global__ __vector__ void fill_output_with_table_vectors_hybrid_kernel(
    uint32_t former_num, uint64_t former_core_move_num, uint64_t tail_core_move_num, uint32_t tile_size,
    uint32_t num_tiles, uint32_t emb_dim, __gm__ T* outputs, typename TableVectorSimd<T>::Args vector_args,
    typename EmbeddingGenerator::Args generator_args)
{
    uint64_t cur_block_idx = GetBlockIdx();
    uint64_t core_start_idx = 0;
    uint64_t core_move_count = 0;
    constexpr uint32_t BUFFER_NUM = 2;

    if (cur_block_idx < former_num) {
        core_start_idx = cur_block_idx * former_core_move_num;
        core_move_count = former_core_move_num;
    } else {
        core_start_idx = former_num * former_core_move_num + (cur_block_idx - former_num) * tail_core_move_num;
        core_move_count = tail_core_move_num;
    }

    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> move_queue;
    TQue<TPosition::VECOUT, 0> out_queue;
    pipe.InitBuffer(move_queue, BUFFER_NUM, tile_size * sizeof(T));
    pipe.InitBuffer(out_queue, BUFFER_NUM, tile_size * sizeof(T));

    GlobalTensor<T> table_gm;
    GlobalTensor<T> output_gm;
    LocalTensor<T> move_local;
    DataCopyPadExtParams<T> pad_params{true, 0, 0, 0};

    output_gm.SetGlobalBuffer(outputs);
    TableVectorSimd<T> table_vector;
    table_vector.init_simd(vector_args);
    EmbeddingGenerator emb_gen;
    emb_gen.init_simd(generator_args);

    for (uint64_t emb_id = core_start_idx; emb_id < core_start_idx + core_move_count; emb_id++) {
        __gm__ T* table_ptr = table_vector.data_ptr(emb_id);
        bool initialized = table_vector.isInitialized(emb_id);
        bool valid = table_ptr != nullptr;
        uint64_t output_offset = emb_id * emb_dim;
        if (initialized) {
            for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                uint32_t tile_offset = tile_idx * tile_size;
                uint32_t current_tile_size = (tile_idx == num_tiles - 1) ? (emb_dim - tile_offset) : tile_size;
                DataCopyExtParams copy_params{1, static_cast<uint32_t>(current_tile_size * sizeof(T)), 0, 0, 0};

                move_queue.AllocTensor<T>(move_local);
                table_gm.SetGlobalBuffer(table_ptr + tile_offset);
                DataCopyPad(move_local, table_gm, copy_params, pad_params);
                move_queue.EnQue<T>(move_local);
                move_queue.DeQue<T>(move_local);
                DataCopyPad(output_gm[output_offset + tile_offset], move_local, copy_params);
                move_queue.FreeTensor(move_local);
            }
        } else {
            for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
                uint32_t tile_offset = tile_idx * tile_size;
                uint32_t current_tile_size = (tile_idx == num_tiles - 1) ? (emb_dim - tile_offset) : tile_size;
                DataCopyExtParams copy_params{1, static_cast<uint32_t>(current_tile_size * sizeof(T)), 0, 0, 0};

                out_queue.AllocTensor<T>(move_local);
                emb_gen.template generate_simd_tensor<T>(move_local, emb_id, current_tile_size, tile_size, !valid);
                out_queue.EnQue<T>(move_local);
                out_queue.DeQue<T>(move_local);

                if (valid) {
                    table_gm.SetGlobalBuffer(table_ptr + tile_offset);
                    DataCopyPad(table_gm, move_local, copy_params);
                }
                DataCopyPad(output_gm[output_offset + tile_offset], move_local, copy_params);
                out_queue.FreeTensor(move_local);
            }
        }
    }

    emb_gen.destroy_simd();
}

}  // namespace dyn_emb

#endif  // ASCENDC_FILL_OUTPUT_WITH_TABLE_VECTOR_HYBRID_KERNEL_H_
