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
}  // namespace dyn_emb
