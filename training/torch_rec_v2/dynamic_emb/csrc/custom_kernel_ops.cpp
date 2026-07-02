/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "custom_kernel_ops.h"
#include "./ops/load_from_pointer/load_from_pointer_kernel_hybrid.h"
#include "./ops/look_backward/lookup_backward_v2_kernel.h"
#include "./ops/ops_utils.h"
#include "utils.h"

namespace dyn_emb {
void load_from_pointer_hybrid_ops(void* pointers, void* dst, uint32_t dim, uint32_t num, aclrtStream stream,
                                  uint32_t coreNum, uint32_t oType, uint64_t totalUbSize)
{
    dyn_emb::DataType outType = static_cast<dyn_emb::DataType>(oType);
    FLOAT_TYPE_DISPATCH(outType, DataType, {
        auto launchTiling = dyn_emb::ComputeSimdValueMoveLaunchTiling(num, coreNum, dim, sizeof(DataType), totalUbSize);
        load_from_pointer_kernel_hybrid<DataType><<<launchTiling.block_dim, launchTiling.valid_ub_size, stream>>>(
            launchTiling.former_num, launchTiling.former_core_move_num, launchTiling.tail_core_move_num,
            launchTiling.tile_size, launchTiling.num_tiles, dim, reinterpret_cast<DataType*>(dst), num,
            reinterpret_cast<DataType**>(pointers));
    });
}

#define LOOKUP_BACKWARD_V2_KERNEL_LAUNCH(IS_FLOAT2, VALUE_T)                                                         \
    do {                                                                                                             \
        if (is_mean) {                                                                                               \
            lookup_backward_v2_kernel<DTYPE_X, VALUE_T, true, IS_FLOAT2><<<core_num, 0, stream>>>(                   \
                reinterpret_cast<VALUE_T*>(grad), reinterpret_cast<VALUE_T*>(unique_buffer),                         \
                reinterpret_cast<DTYPE_X*>(inverse_indices), reinterpret_cast<DTYPE_X*>(biased_offsets), launch_dim, \
                num_slots, total_blocks, blocks_per_core, remainder_blocks, is_small);                               \
        } else {                                                                                                     \
            lookup_backward_v2_kernel<DTYPE_X, VALUE_T, false, IS_FLOAT2><<<core_num, 0, stream>>>(                  \
                reinterpret_cast<VALUE_T*>(grad), reinterpret_cast<VALUE_T*>(unique_buffer),                         \
                reinterpret_cast<DTYPE_X*>(inverse_indices), reinterpret_cast<DTYPE_X*>(biased_offsets), launch_dim, \
                num_slots, total_blocks, blocks_per_core, remainder_blocks, is_small);                               \
        }                                                                                                            \
    } while (0)

void lookup_backward_v2_launch(void* grad, void* unique_buffer, void* inverse_indices, void* biased_offsets,
                               int32_t launch_dim, int32_t num_slots, int32_t combiner, int32_t total_blocks,
                               int32_t blocks_per_core, int32_t remainder_blocks, uint32_t index_type, bool is_small,
                               bool is_float2, uint32_t value_type, int32_t core_num, aclrtStream stream)
{
    const bool is_mean = (combiner == 1);
    const dyn_emb::DataType index_enum = static_cast<dyn_emb::DataType>(index_type);
    const dyn_emb::DataType value_enum = static_cast<dyn_emb::DataType>(value_type);

    if (is_float2) {
        INDEX_DTYPE_DISPATCH(index_enum, DTYPE_X, { LOOKUP_BACKWARD_V2_KERNEL_LAUNCH(true, float2); });
        return;
    }

    INDEX_DTYPE_DISPATCH(index_enum, DTYPE_X, {
        FLOAT_TYPE_DISPATCH(value_enum, value_t, { LOOKUP_BACKWARD_V2_KERNEL_LAUNCH(false, value_t); });
    });
}
}  // namespace dyn_emb
