/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#pragma once

#include <cstdint>
#include <acl/acl.h>

#include "dynamic_variable_base.h"
#include "utils.h"

namespace dyn_emb {

void alloc_curand_states(curandState** states, aclrtStream stream);
void free_curand_states(curandState* states);

void launch_index_normal_init(DataType value_type, DataType index_type, int64_t num, int64_t dim, int64_t stride,
                              void* buffer, void* indices, curandState* state, float mean, float std_dev,
                              aclrtStream stream);

void launch_index_truncated_normal_init(DataType value_type, DataType index_type, int64_t num, int64_t dim,
                                        int64_t stride, void* buffer, void* indices, curandState* state, float mean,
                                        float std_dev, float lower, float upper, aclrtStream stream);

void launch_index_uniform_init(DataType value_type, DataType index_type, int64_t num, int64_t dim, int64_t stride,
                               void* buffer, void* indices, curandState* state, float lower, float upper,
                               aclrtStream stream);

void launch_index_const_init(DataType value_type, DataType index_type, int64_t num, int64_t dim, int64_t stride,
                             void* buffer, void* indices, float value, aclrtStream stream);

void launch_index_debug_init(DataType value_type, DataType index_type, DataType key_type, int64_t num, int64_t dim,
                             int64_t stride, void* buffer, void* indices, const void* keys, uint64_t mod,
                             aclrtStream stream);

}  // namespace dyn_emb
