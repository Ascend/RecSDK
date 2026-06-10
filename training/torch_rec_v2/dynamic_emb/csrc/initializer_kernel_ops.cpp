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

#include "initializer_kernel_ops.h"

#include <random>

#include "acl_singleton.h"
#include "initializer_generators.h"
#include "hkv_variable.h"
#include "ops/initialize_with_index/initialize_with_index_kernel.h"

namespace dyn_emb {

__simt_vf__ __aicore__ LAUNCH_BOUND(EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK) inline void setup_kernel_vf(
    uint64_t seed, __gm__ curandState* states, const uint32_t block_index)
{
    uint32_t tid = block_index * blockDim.x + threadIdx.x;
    curand_init(seed + static_cast<uint64_t>(tid), static_cast<uint64_t>(tid), 0, &states[tid]);
}

__global__ __vector__ void setup_kernel(uint64_t seed, __gm__ curandState* states)
{
    asc_vf_call<setup_kernel_vf>(dim3{EMBEDDING_GENERATOR_MAX_THREADS_PER_BLOCK}, seed, states, GetBlockIdx());
}

void alloc_curand_states(curandState** states, aclrtStream stream)
{
    auto& deviceProp = DeviceProp::getDeviceProp();
    NPU_CHECK(aclrtMalloc(reinterpret_cast<void**>(states), sizeof(curandState) * deviceProp.total_threads,
                          ACL_MEM_MALLOC_HUGE_FIRST));

    std::random_device rd;
    auto seed = rd();
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    setup_kernel<<<maxCores, 0, stream>>>(seed, *states);
}

void free_curand_states(curandState* states)
{
    if (states != nullptr) {
        NPU_CHECK(aclrtFree(states));
    }
}

template <typename GeneratorT>
static void launch_initialize_with_generator(DataType value_type, DataType index_type, int64_t num, int64_t dim,
                                             int64_t stride, void* buffer, void* indices,
                                             typename GeneratorT::Args generator_args, aclrtStream stream)
{
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
        DISPATCH_INTEGER_DATATYPE_FUNCTION(index_type, IndexType, [&] {
            initialize_with_index_kernel<ValueType, IndexType, GeneratorT>
                <<<max_cores, 0, stream>>>(num, dim, stride, reinterpret_cast<ValueType*>(buffer),
                                           reinterpret_cast<IndexType*>(indices), generator_args);
        });
    });
}

void launch_index_normal_init(DataType value_type, DataType index_type, int64_t num, int64_t dim, int64_t stride,
                              void* buffer, void* indices, curandState* state, float mean, float std_dev,
                              aclrtStream stream)
{
    NormalEmbeddingGenerator::Args generator_args{state, mean, std_dev};
    launch_initialize_with_generator<NormalEmbeddingGenerator>(value_type, index_type, num, dim, stride, buffer,
                                                               indices, generator_args, stream);
}

void launch_index_truncated_normal_init(DataType value_type, DataType index_type, int64_t num, int64_t dim,
                                        int64_t stride, void* buffer, void* indices, curandState* state, float mean,
                                        float std_dev, float lower, float upper, aclrtStream stream)
{
    TruncatedNormalEmbeddingGenerator::Args generator_args{state, mean, std_dev, lower, upper};
    launch_initialize_with_generator<TruncatedNormalEmbeddingGenerator>(value_type, index_type, num, dim, stride,
                                                                        buffer, indices, generator_args, stream);
}

void launch_index_uniform_init(DataType value_type, DataType index_type, int64_t num, int64_t dim, int64_t stride,
                               void* buffer, void* indices, curandState* state, float lower, float upper,
                               aclrtStream stream)
{
    UniformEmbeddingGenerator::Args generator_args{state, lower, upper};
    launch_initialize_with_generator<UniformEmbeddingGenerator>(value_type, index_type, num, dim, stride, buffer,
                                                                indices, generator_args, stream);
}

void launch_index_const_init(DataType value_type, DataType index_type, int64_t num, int64_t dim, int64_t stride,
                             void* buffer, void* indices, float value, aclrtStream stream)
{
    ConstEmbeddingGenerator::Args generator_args{value};
    launch_initialize_with_generator<ConstEmbeddingGenerator>(value_type, index_type, num, dim, stride, buffer, indices,
                                                              generator_args, stream);
}

void launch_index_debug_init(DataType value_type, DataType index_type, DataType key_type, int64_t num, int64_t dim,
                             int64_t stride, void* buffer, void* indices, const void* keys, uint64_t mod,
                             aclrtStream stream)
{
    DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
        MappingEmbeddingGenerator<KeyType>::Args generator_args{reinterpret_cast<const KeyType*>(keys), mod};
        launch_initialize_with_generator<MappingEmbeddingGenerator<KeyType>>(value_type, index_type, num, dim, stride,
                                                                             buffer, indices, generator_args, stream);
    });
}

}  // namespace dyn_emb
