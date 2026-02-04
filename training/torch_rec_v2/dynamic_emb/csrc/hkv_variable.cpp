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

#include "hkv_variable.h"
#include <stdexcept>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "utils.h"

namespace dyn_emb {

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
HKVVariable<KeyType, ValueType, Strategy>::HKVVariable(
    DataType key_type, DataType value_type, int64_t dim,
    int64_t init_capacity, size_t max_capacity,
    size_t max_hbm_for_vectors, size_t max_bucket_size,
    float max_load_factor, int block_size,
    int io_block_size, int device_id,
    bool io_by_cpu, bool use_constant_memory,
    int reserved_key_start_bit,
    size_t num_of_buckets_per_alloc,
    const InitializerArgs &initializer_args,
    const SafeCheckMode safe_check_mode,
    const OptimizerType optimizer_type)
    : dim_(dim), max_capacity_(max_capacity), initializer_args_(initializer_args),
    key_type_(key_type), value_type_(value_type),
    safe_check_mode_(safe_check_mode), optimizer_type_(optimizer_type)
{
    if (dim <= 0) {
        throw std::invalid_argument("dimension must > 0 but got " +
            std::to_string(dim));
    }

    uint32_t deviceCount;
    NPU_CHECK(aclrtGetDeviceCount(&deviceCount));
    if (device_id < 0 || device_id >= deviceCount) {
        throw std::invalid_argument("Invalid device id, device id is ." +
            std::to_string(device_id));
    }
    NPU_CHECK(aclrtSetDevice(static_cast<int32_t>(device_id)));

    hkv_table_option_.init_capacity = init_capacity;
    hkv_table_option_.max_capacity = max_capacity;
    hkv_table_option_.dim =
        dim + get_optimizer_state_dim<ValueType>(optimizer_type, dim);
    int64_t max_hbm_needed = hkv_table_option_.max_capacity *
                            hkv_table_option_.dim * sizeof(ValueType);
    hkv_table_option_.max_hbm_for_vectors = max_hbm_needed < max_hbm_for_vectors
                                                ? max_hbm_needed
                                                : max_hbm_for_vectors;
    hkv_table_option_.max_bucket_size = max_bucket_size;
    hkv_table_option_.max_load_factor = max_load_factor;
    hkv_table_option_.block_size = block_size;
    hkv_table_option_.io_block_size = io_block_size;
    hkv_table_option_.device_id = device_id;
    hkv_table_option_.io_by_cpu = io_by_cpu;
    hkv_table_option_.use_constant_memory = use_constant_memory;
    hkv_table_option_.reserved_key_start_bit = reserved_key_start_bit;
    hkv_table_option_.num_of_buckets_per_alloc = num_of_buckets_per_alloc;
    hkv_table_option_.api_lock = false;

    hkv_table_->init(hkv_table_option_);
}


template <typename KeyType, typename ValueType, EvictStrategy Strategy>
HKVVariable<KeyType, ValueType, Strategy>::~HKVVariable() {}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int64_t HKVVariable<KeyType, ValueType, Strategy>::get_max_capacity()
{
    return max_capacity_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
DataType HKVVariable<KeyType, ValueType, Strategy>::get_key_type()
{
    return key_type_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
DataType HKVVariable<KeyType, ValueType, Strategy>::get_value_type()
{
    return value_type_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
EvictStrategy HKVVariable<KeyType, ValueType, Strategy>::get_evict_strategy() const
{
    return Strategy;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
const InitializerArgs &HKVVariable<KeyType, ValueType, Strategy>::get_initializer_args() const
{
    return initializer_args_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_pointers(
    const size_t n, const void *keys, void **value_ptrs, bool *founds,
    void *scores, aclrtStream stream) const
{
    if (n == 0) {
        return;
    }

    hkv_table_->find(n, (KeyType *)keys, (ValueType **)value_ptrs, founds,
        (uint64_t *)scores, stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_pointers(
    const size_t n, const void *keys, void **value_ptrs, bool *founds,
    void *scores, aclrtStream stream)
{
    if (n == 0) {
        return;
    }

    hkv_table_->find_and_update(n, (KeyType *)keys, (ValueType **)value_ptrs,
        founds, (uint64_t *)scores, stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int HKVVariable<KeyType, ValueType, Strategy>::optstate_dim() const
{
    return hkv_table_option_.dim - dim_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int HKVVariable<KeyType, ValueType, Strategy>::get_emb_cols() const
{
    return dim_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::export_batch(const size_t n, const size_t offset,
    const torch::Tensor d_counter, const torch::Tensor keys,
    const torch::Tensor values,
    const c10::optional<torch::Tensor>& score) const
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        hkv_table_->export_batch(n, offset, d_counter.data_ptr<size_t>(), (KeyType*)keys.data_ptr(),
                                 (ValueType*)values.data_ptr(), (uint64_t*)score_.data_ptr(), stream);
    } else {
        hkv_table_->export_batch(n, offset, d_counter.data_ptr<size_t>(), (KeyType*)keys.data_ptr(),
                                 (ValueType*)values.data_ptr(), nullptr, stream);
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::export_batch_matched(const uint64_t threshold, const uint64_t n,
    const uint64_t offset, at::Tensor num_matched,
    at::Tensor keys, at::Tensor values) const
{
    // Mock implementation:
    std::cout << "Mock export_batch_matched called with threshold=" << threshold << ", n=" << n << std::endl;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::count_matched(const uint64_t threshold, at::Tensor num_matched) const
{
    // Mock implementation:
    std::cout << "Mock count_matched called with threshold=" << threshold << std::endl;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::load(const size_t n, const torch::Tensor keys,
    const torch::Tensor values,
    const c10::optional<torch::Tensor>& score, bool unique_key,
    bool ignore_evict_strategy)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        hkv_table_->insert_or_assign(n, (KeyType*)keys.data_ptr(), (ValueType*)values.data_ptr(),
            (uint64_t*)score_.data_ptr(), stream, unique_key, ignore_evict_strategy);
    } else {
        hkv_table_->insert_or_assign(n, (KeyType*)keys.data_ptr(), (ValueType*)values.data_ptr(), nullptr, stream,
            unique_key, ignore_evict_strategy);
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::update(const size_t n, const torch::Tensor keys,
    const torch::Tensor values,
    const c10::optional<torch::Tensor>& score, bool unique_key,
    bool ignore_evict_strategy)
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    if (score.has_value()) {
        at::Tensor score_ = score.value();
        hkv_table_->insert_or_assign(n, (KeyType*)keys.data_ptr(), (ValueType*)values.data_ptr(),
            (uint64_t*)score_.data_ptr(), stream, unique_key, ignore_evict_strategy);
    } else {
        hkv_table_->insert_or_assign(n, (KeyType*)keys.data_ptr(), (ValueType*)values.data_ptr(), nullptr, stream,
            unique_key, ignore_evict_strategy);
    }
}

// 2 × 3 × 3 = 18 种组合全部给出实例化
template class HKVVariable<int64_t,  float, EvictStrategy::kCustomized>;
template class HKVVariable<int64_t,  float, EvictStrategy::kLru>;
template class HKVVariable<int64_t,  float, EvictStrategy::kLfu>;

template class HKVVariable<int64_t,  half, EvictStrategy::kCustomized>;
template class HKVVariable<int64_t,  half, EvictStrategy::kLru>;
template class HKVVariable<int64_t,  half, EvictStrategy::kLfu>;

template class HKVVariable<int64_t,  bfloat16_t, EvictStrategy::kCustomized>;
template class HKVVariable<int64_t,  bfloat16_t, EvictStrategy::kLru>;
template class HKVVariable<int64_t,  bfloat16_t, EvictStrategy::kLfu>;

template class HKVVariable<uint64_t, float, EvictStrategy::kCustomized>;
template class HKVVariable<uint64_t, float, EvictStrategy::kLru>;
template class HKVVariable<uint64_t, float, EvictStrategy::kLfu>;

template class HKVVariable<uint64_t,  half, EvictStrategy::kCustomized>;
template class HKVVariable<uint64_t,  half, EvictStrategy::kLru>;
template class HKVVariable<uint64_t,  half, EvictStrategy::kLfu>;

template class HKVVariable<uint64_t,  bfloat16_t, EvictStrategy::kCustomized>;
template class HKVVariable<uint64_t,  bfloat16_t, EvictStrategy::kLru>;
template class HKVVariable<uint64_t,  bfloat16_t, EvictStrategy::kLfu>;

} // namespace dyn_emb
