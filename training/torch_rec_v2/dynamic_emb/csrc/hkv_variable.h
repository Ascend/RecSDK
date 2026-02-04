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

#ifndef MOCK_HKV_VARIABLE_H
#define MOCK_HKV_VARIABLE_H

#include <string>
#include <torch/extension.h>
#include "dynamic_variable_base.h"
#include "hkv_hashtable.h"


namespace dyn_emb {

template <typename KeyType, typename ValueType, EvictStrategy Strategy = EvictStrategy::kLru>
class HKVVariable : public DynamicVariableBase {
public:
    HKVVariable(DataType key_type, DataType value_type, int64_t dim, int64_t init_capacity, size_t max_capacity,
                size_t max_hbm_for_vectors = 0, size_t max_bucket_size = 128, float max_load_factor = 0.5f,
                int block_size = 128, int io_block_size = 1024, int device_id = -1, bool io_by_cpu = false,
                bool use_constant_memory = false, int reserved_key_start_bit = 0, size_t num_of_buckets_per_alloc = 1,
                const InitializerArgs& initializer_args = InitializerArgs(),
                const SafeCheckMode safe_check_mode = SafeCheckMode::IGNORE,
                const OptimizerType optimizer_type = OptimizerType::Null);

    ~HKVVariable() override;

    DataType get_key_type() override;

    DataType get_value_type() override;

    EvictStrategy get_evict_strategy() const override;

    int64_t get_max_capacity() override;

    const InitializerArgs& get_initializer_args() const override;
    
    void find_pointers(const size_t n, const void *keys, // (n)
                     void **values,                      // (n)
                     bool *founds,                       // (n)
                     void *scores = nullptr,             // (n)
                     aclrtStream stream = 0) const override;

    void find_pointers(const size_t n, const void *keys, // (n)
                     void **values,                      // (n)
                     bool *founds,                       // (n)
                     void *scores = nullptr,             // (n)
                     aclrtStream stream = 0) override;

    int optstate_dim() const override;

    int get_emb_cols() const override;

    void export_batch(const size_t n, const size_t offset, const torch::Tensor d_counter, const torch::Tensor keys,
                      const torch::Tensor values,
                      const c10::optional<torch::Tensor>& score = c10::nullopt) const override;

    void export_batch_matched(const uint64_t threshold, const uint64_t n, const uint64_t offset, at::Tensor num_matched,
                              at::Tensor keys, at::Tensor values) const override;

    void count_matched(const uint64_t threshold, at::Tensor num_matched) const override;

    void update(const size_t n, const torch::Tensor keys, const torch::Tensor values,
                const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
                bool ignore_evict_strategy = false) override;

    void load(const size_t n, const torch::Tensor keys, const torch::Tensor values,
              const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
              bool ignore_evict_strategy = false) override;

private:
    using HKVTable =
      npu::hkv::HashTable<KeyType, ValueType, uint64_t, (int)Strategy>;
    std::unique_ptr<HKVTable> hkv_table_ = std::make_unique<HKVTable>();
    npu::hkv::HashTableOptions hkv_table_option_;
    size_t dim_;
    size_t max_capacity_;
    const InitializerArgs initializer_args_;

    DataType key_type_;
    DataType value_type_;
    SafeCheckMode safe_check_mode_;
    OptimizerType optimizer_type_;
};

}  // namespace dyn_emb

#endif  // HKV_VARIABLE_H
