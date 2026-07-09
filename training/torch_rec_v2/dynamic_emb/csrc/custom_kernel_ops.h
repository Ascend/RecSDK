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
#ifndef CUSTOM_KERNEL_OPS_H
#define CUSTOM_KERNEL_OPS_H

#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace dyn_emb {
void load_from_pointer_hybrid_ops(void* pointers, void* dst, uint32_t dim, uint32_t num, aclrtStream stream,
                                  uint32_t coreNum, uint32_t oType, uint64_t totalUbSize);

void maskselect_ops(void* flags, uint8_t* inputs, void* outputs, void* num_selected, void* workspace, int64_t num_total,
                    uint32_t is_uint64, uint32_t select_index, aclrtStream stream, uint32_t max_cores,
                    uint64_t ub_size);

int64_t GetMaskSelectWorkspaceElems(int64_t num_total, uint32_t max_cores, uint64_t ub_size, bool select_index);

void lookup_backward_v2_launch(void* grad, void* unique_buffer, void* inverse_indices, void* biased_offsets,
                               int32_t launch_dim, int32_t num_slots, int32_t combiner, int32_t total_blocks,
                               int32_t blocks_per_core, int32_t remainder_blocks, uint32_t index_type, bool is_small,
                               bool is_float2, uint32_t value_type, int32_t core_num, aclrtStream stream);
}  // namespace dyn_emb

#endif  // CUSTOM_KERNEL_OPS_H
