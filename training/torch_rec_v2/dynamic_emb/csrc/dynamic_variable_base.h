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

#ifndef DYNAMIC_VARIABLE_BASE_H
#define DYNAMIC_VARIABLE_BASE_H

#include <string>
#include <memory>
#include <stdexcept>
#include <torch/extension.h>

#include "acl/acl.h"

#include "utils.h"

namespace dyn_emb {

struct InitializerArgs {
    const std::string mode_;
    float mean_;
    float std_dev_;
    float lower_;
    float upper_;
    float value_;
    InitializerArgs(const std::string &mode, float mean, float stdDev,
                    float lower, float upper, float value)
        : mode_(mode), mean_(mean), std_dev_(stdDev), lower_(lower),
          upper_(upper), value_(value) {}
    InitializerArgs()
        : InitializerArgs("uniform", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f) {}
};

template <typename T>
uint32_t get_optimizer_state_dim(OptimizerType opt_type, uint32_t emb_dim)
{
    uint32_t optstate_dim = 0;
    switch (opt_type) {
        case OptimizerType::Null: {
            break;
        }
        case OptimizerType::SGD: {
            break;
        }
        case OptimizerType::Adam: {
            optstate_dim = emb_dim * 2;
            break;
        }
        case OptimizerType::AdamW: {
            optstate_dim = emb_dim * 2;
            break;
        }
        case OptimizerType::AdaGrad: {
            optstate_dim = emb_dim;
            break;
        }
        case OptimizerType::RowWiseAdaGrad: {
            optstate_dim = 16 / sizeof(T); // 16 bytes per row
            break;
        }
        default: {
            throw std::invalid_argument("Unsupported optimizer type.");
        }
    }
    return optstate_dim;
}

class DynamicVariableBase {
public:
    virtual ~DynamicVariableBase() = default;
    virtual int64_t get_max_capacity() = 0;
    virtual DataType get_key_type() = 0;
    virtual DataType get_value_type() = 0;
    virtual EvictStrategy get_evict_strategy() const = 0;
    virtual const InitializerArgs &get_initializer_args() const = 0;
    virtual void find_pointers(const size_t n, const void *keys, // (n)
                              void **values,                    // (n)
                              bool *founds,                     // (n)
                              void *scores = nullptr,           // (n)
                              aclrtStream stream = 0) const = 0;
    virtual void find_pointers(const size_t n, const void *keys, // (n)
                              void **values,                    // (n)
                              bool *founds,                     // (n)
                              void *scores = nullptr,           // (n)
                              aclrtStream stream = 0) = 0;
    virtual int optstate_dim() const = 0;
    virtual int get_emb_cols() const = 0;
    virtual void export_batch(const size_t n, const size_t offset, const torch::Tensor d_counter,
                              const torch::Tensor keys, const torch::Tensor values,
                              const c10::optional<torch::Tensor>& score = c10::nullopt) const = 0;
    virtual void export_batch_matched(const uint64_t threshold, const uint64_t n, const uint64_t offset,
                                      at::Tensor num_matched, at::Tensor keys, at::Tensor values) const = 0;
    virtual void count_matched(const uint64_t threshold, at::Tensor num_matched) const = 0;
    virtual void update(const size_t n, const torch::Tensor keys, const torch::Tensor values,
                        const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
                        bool ignore_evict_strategy = false) = 0;
    virtual void load(const size_t n, const torch::Tensor keys, const torch::Tensor values,
                      const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
                      bool ignore_evict_strategy = false) = 0;
};

class VariableFactory {
public:
    static std::shared_ptr<DynamicVariableBase> Create(
        DataType key_type, DataType value_type, EvictStrategy evict_type,
        int64_t dim, size_t init_capacity, size_t max_capacity,
        size_t max_hbm_for_vectors, size_t max_bucket_size, float max_load_factor,
        int block_size, int io_block_size, int device_id, bool io_by_cpu,
        bool use_constant_memory, int reserved_key_start_bit,
        size_t num_of_buckets_per_alloc, const InitializerArgs &initializer_args,
        const SafeCheckMode safe_check_mode = SafeCheckMode::IGNORE,
        const OptimizerType optimizer_type = OptimizerType::Null);
};

} // namespace dyn_emb
#endif // DYNAMIC_VARIABLE_BASE_H
