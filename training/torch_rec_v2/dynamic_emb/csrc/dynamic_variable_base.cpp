/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "dynamic_variable_base.h"
#include "hkv_variable.h"
namespace dyn_emb {

std::shared_ptr<DynamicVariableBase> VariableFactory::Create(
    DataType key_type, DataType value_type, EvictStrategy evict_type,
    int64_t dim, size_t init_capacity, size_t max_capacity,
    size_t max_hbm_for_vectors, size_t max_bucket_size, float max_load_factor,
    int block_size, int io_block_size, int device_id, bool io_by_cpu,
    bool use_constant_memory, int reserved_key_start_bit,
    size_t num_of_buckets_per_alloc, const InitializerArgs &initializer_args,
    const SafeCheckMode safe_check_mode,
    const OptimizerType optimizer_type)
{
    // value type float
    std::shared_ptr<DynamicVariableBase> table;
    DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, keyT, [&] {
        DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, valueT, [&] {
            DISPATCH_EVICTYPE_FUNCTION(evict_type, evictT, [&] {
                table = std::make_shared<HKVVariable<keyT, valueT, evictT>>(
                    key_type, value_type, dim, init_capacity, max_capacity,
                    max_hbm_for_vectors, max_bucket_size, max_load_factor, block_size,
                    io_block_size, device_id, io_by_cpu, use_constant_memory,
                    reserved_key_start_bit, num_of_buckets_per_alloc, initializer_args,
                    safe_check_mode, optimizer_type);
            });
        });
    });
    return table;
}

} // namespace dyn_emb
