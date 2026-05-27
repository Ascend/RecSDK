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
    InitializerArgs(const std::string& mode, float mean, float stdDev, float lower, float upper, float value)
        : mode_(mode),
          mean_(mean),
          std_dev_(stdDev),
          lower_(lower),
          upper_(upper),
          value_(value)
    {
    }
    InitializerArgs() : InitializerArgs("uniform", 0.0f, 1.0f, 0.0f, 1.0f, 0.0f) {}
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
            optstate_dim = 16 / sizeof(T);  // 16 bytes per row
            break;
        }
        default: {
            throw std::invalid_argument("Unsupported optimizer type.");
        }
    }
    return optstate_dim;
}

// 由于cann暂无curandState对应结构体，模拟cuda curandState, 默认为curandStateXORWOW
struct curandState {
    // 5个位移状态寄存器x, y, z, w, v
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
    uint32_t v;
    uint32_t d;          // 加法计数器
    float normal_spare;  // 用于缓存Box-Muller产生的第二个随机数
    int has_spare;       // 标记是否有缓存值
};

#ifdef USE_RTTI
class DynamicVariableBase {
public:
    using RowsFn = std::function<int64_t(aclrtStream)>;
    using ColsFn = std::function<int64_t()>;
    using IsPureHbmModeFn = std::function<bool()>;
    using GetMaxCapacityFn = std::function<int64_t()>;
    using GetKeyTypeFn = std::function<DataType()>;
    using GetValueTypeFn = std::function<DataType()>;
    using GetEvictStrategyFn = std::function<EvictStrategy()>;
    using GetInitializerArgsFn = std::function<const InitializerArgs&()>;
    using InsertAndEvictFn = std::function<void(const size_t, const void*, const void*, const void*, void*, void*,
                                                void*, uint64_t*, aclrtStream, bool, bool)>;
    using FindFn = std::function<void(const size_t, const void*, void*, bool*, void*, aclrtStream)>;
    using EraseFn = std::function<void(const size_t, const void*, aclrtStream)>;
    using ClearFn = std::function<void(aclrtStream)>;
    using ReserveFn = std::function<void(const size_t, aclrtStream)>;
    using AccumOrAssignFn =
        std::function<void(const size_t, const void*, const void*, const bool*, const void*, aclrtStream, bool)>;
    using FindOrInsertFn =
        std::function<void(const size_t, const void*, void**, void*, bool*, void*, aclrtStream, bool, bool)>;
    using FindOrInsertPointersFn =
        std::function<void(const size_t, const void*, void**, bool*, void*, aclrtStream, bool, bool)>;
    using AssignFn = std::function<void(const size_t, const void*, const void*, const void*, aclrtStream, bool)>;
    using LockFn = std::function<void(const size_t, const void*, void**, bool*, void*, aclrtStream)>;
    using UnlockFn = std::function<void(const size_t, void**, const void*, bool*, aclrtStream)>;
    using FindPointersConstFn = std::function<void(const size_t, const void*, void**, bool*, void*, aclrtStream)>;
    using FindPointersFn = std::function<void(const size_t, const void*, void**, bool*, void*, aclrtStream)>;
    using OptStateDimFn = std::function<int()>;
    using SetInitialOptStateFn = std::function<void(const float)>;
    using GetInitialOptStateFn = std::function<const float()>;
    using GetEmbColsFn = std::function<int()>;
    using ExportBatchFn = std::function<void(const size_t, const size_t, const torch::Tensor, const torch::Tensor,
                                             const torch::Tensor, const c10::optional<torch::Tensor>&)>;
    using ExportBatchMatchedFn =
        std::function<void(const uint64_t, const uint64_t, const uint64_t, torch::Tensor, torch::Tensor, torch::Tensor,
                           const c10::optional<torch::Tensor>&, aclrtStream)>;
    using CountMatchedFn = std::function<void(const uint64_t, torch::Tensor, aclrtStream)>;
    using UpdateFn = std::function<void(const size_t, const torch::Tensor, const torch::Tensor,
                                        const c10::optional<torch::Tensor>&, bool, bool)>;
    using LoadFn = std::function<void(const size_t, const torch::Tensor, const torch::Tensor,
                                      const c10::optional<torch::Tensor>&, bool, bool)>;
    using FindAndInitializeFn = std::function<void(const size_t, const void*, void**, void*, bool*,
                                                   const c10::optional<InitializerArgs>&, aclrtStream)>;

    DynamicVariableBase(RowsFn rows_fn, ColsFn cols_fn, IsPureHbmModeFn is_pure_hbm_mode_fn,
                        GetMaxCapacityFn get_max_capacity_fn, GetKeyTypeFn get_key_type_fn,
                        GetValueTypeFn get_value_type_fn, GetEvictStrategyFn get_evict_strategy_fn,
                        GetInitializerArgsFn get_initializer_args_fn, InsertAndEvictFn insert_and_evict_fn,
                        FindFn find_fn, EraseFn erase_fn, ClearFn clear_fn, ReserveFn reserve_fn,
                        AccumOrAssignFn accum_or_assign_fn, FindOrInsertFn find_or_insert_fn,
                        FindOrInsertPointersFn find_or_insert_pointers_fn, AssignFn assign_fn, LockFn lock_fn,
                        UnlockFn unlock_fn, FindPointersConstFn find_pointers_const_fn, FindPointersFn find_pointers_fn,
                        OptStateDimFn optstate_dim_fn, SetInitialOptStateFn set_initial_optstate_fn,
                        GetInitialOptStateFn get_initial_optstate_fn, GetEmbColsFn get_emb_cols_fn,
                        ExportBatchFn export_batch_fn, ExportBatchMatchedFn export_batch_matched_fn,
                        CountMatchedFn count_matched_fn, UpdateFn update_fn, LoadFn load_fn,
                        FindAndInitializeFn find_and_initialize_fn);

    ~DynamicVariableBase() = default;

    int64_t rows(aclrtStream stream = 0);
    int64_t cols();
    bool is_pure_hbm_mode() const;
    int64_t get_max_capacity();
    DataType get_key_type();
    DataType get_value_type();
    EvictStrategy get_evict_strategy() const;
    const InitializerArgs& get_initializer_args() const;
    EvictStrategy evict_strategy() const;
    void insert_and_evict(const size_t n, const void* keys, const void* values, const void* scores, void* evicted_keys,
                          void* evicted_values, void* evicted_scores, uint64_t* d_evicted_counter,
                          aclrtStream stream = 0, bool unique_key = true, bool ignore_evict_strategy = false);
    void find(const size_t n, const void* keys, void* values, bool* founds, void* scores = nullptr,
              aclrtStream stream = 0) const;
    void erase(const size_t n, const void* keys, aclrtStream stream = 0);
    void clear(aclrtStream stream = 0);
    void reserve(const size_t new_capacity, aclrtStream stream = 0);
    void accum_or_assign(const size_t n, const void* keys, const void* value_or_deltas, const bool* accum_or_assigns,
                         const void* scores = nullptr, aclrtStream stream = 0, bool ignore_evict_strategy = false);
    void find_or_insert(const size_t n, const void* keys, void** value_ptrs, void* values, bool* founds,
                        void* scores = nullptr, aclrtStream stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false);
    void find_or_insert_pointers(const size_t n, const void* keys, void** value_ptrs, bool* d_found,
                                 void* scores = nullptr, aclrtStream stream = 0, bool unique_key = true,
                                 bool ignore_evict_strategy = false);
    void set_initial_optstate(const float value);
    const float get_initial_optstate() const;
    void assign(const size_t n, const void* keys, const void* values, const void* scores = nullptr,
                aclrtStream stream = 0, bool unique_key = true);
    void lock(const size_t n, const void* keys, void** locked_keys_ptr, bool* flags = nullptr, void* scores = nullptr,
              aclrtStream stream = 0);
    void unlock(const size_t n, void** locked_keys_ptr, const void* keys, bool* flags = nullptr,
                aclrtStream stream = 0);
    void find_pointers(const size_t n, const void* keys, void** values, bool* founds, void* scores = nullptr,
                       aclrtStream stream = 0) const;
    void find_pointers(const size_t n, const void* keys, void** values, bool* founds, void* scores = nullptr,
                       aclrtStream stream = 0);
    int optstate_dim() const;
    int get_emb_cols() const;
    void export_batch(const size_t n, const size_t offset, const torch::Tensor d_counter, const torch::Tensor keys,
                      const torch::Tensor values, const c10::optional<torch::Tensor>& score = c10::nullopt) const;
    void export_batch_matched(const uint64_t threshold, const uint64_t n, const uint64_t offset,
                              torch::Tensor num_matched, torch::Tensor keys, torch::Tensor values,
                              const c10::optional<torch::Tensor>& scores = c10::nullopt, aclrtStream stream = 0) const;
    void count_matched(const uint64_t threshold, torch::Tensor num_matched, aclrtStream stream = 0) const;
    void update(const size_t n, const torch::Tensor keys, const torch::Tensor values,
                const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
                bool ignore_evict_strategy = false);
    void load(const size_t n, const torch::Tensor keys, const torch::Tensor values,
              const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
              bool ignore_evict_strategy = false);

    void find_and_initialize(const size_t n, const void* keys, void** value_ptrs, void* values, bool* d_found,
                             const c10::optional<InitializerArgs>& initializer_args = c10::nullopt,
                             aclrtStream stream = 0);

private:
    RowsFn rows_fn_;
    ColsFn cols_fn_;
    IsPureHbmModeFn is_pure_hbm_mode_fn_;
    GetMaxCapacityFn get_max_capacity_fn_;
    GetKeyTypeFn get_key_type_fn_;
    GetValueTypeFn get_value_type_fn_;
    GetEvictStrategyFn get_evict_strategy_fn_;
    GetInitializerArgsFn get_initializer_args_fn_;
    InsertAndEvictFn insert_and_evict_fn_;
    FindFn find_fn_;
    EraseFn erase_fn_;
    ClearFn clear_fn_;
    ReserveFn reserve_fn_;
    AccumOrAssignFn accum_or_assign_fn_;
    FindOrInsertFn find_or_insert_fn_;
    FindOrInsertPointersFn find_or_insert_pointers_fn_;
    AssignFn assign_fn_;
    LockFn lock_fn_;
    UnlockFn unlock_fn_;
    FindPointersConstFn find_pointers_const_fn_;
    FindPointersFn find_pointers_fn_;
    OptStateDimFn optstate_dim_fn_;
    SetInitialOptStateFn set_initial_optstate_fn_;
    GetInitialOptStateFn get_initial_optstate_fn_;
    GetEmbColsFn get_emb_cols_fn_;
    ExportBatchFn export_batch_fn_;
    ExportBatchMatchedFn export_batch_matched_fn_;
    CountMatchedFn count_matched_fn_;
    UpdateFn update_fn_;
    LoadFn load_fn_;
    FindAndInitializeFn find_and_initialize_fn_;
};
#else
class DynamicVariableBase {
public:
    virtual ~DynamicVariableBase() = default;
    virtual int64_t rows(aclrtStream stream = 0) = 0;
    virtual int64_t cols() = 0;
    virtual bool is_pure_hbm_mode() const = 0;
    virtual int64_t get_max_capacity() = 0;
    virtual DataType get_key_type() = 0;
    virtual DataType get_value_type() = 0;
    virtual EvictStrategy get_evict_strategy() const = 0;
    virtual const InitializerArgs& get_initializer_args() const = 0;
    virtual EvictStrategy evict_strategy() const = 0;
    virtual void insert_and_evict(const size_t n,
                                  const void* keys,             // (n)
                                  const void* values,           // (n, DIM)
                                  const void* scores,           // (n)
                                  void* evicted_keys,           // (n)
                                  void* evicted_values,         // (n, DIM)
                                  void* evicted_scores,         // (n)
                                  uint64_t* d_evicted_counter,  // (1)
                                  aclrtStream stream = 0, bool unique_key = true,
                                  bool ignore_evict_strategy = false) = 0;
    virtual void find(const size_t n, const void* keys,  // (n)
                      void* values,                      // (n, DIM)
                      bool* founds,                      // (n)
                      void* scores = nullptr,            // (n)
                      aclrtStream stream = 0) const = 0;
    virtual void erase(const size_t n, const void* keys, aclrtStream stream = 0) = 0;
    virtual void clear(aclrtStream stream = 0) = 0;
    virtual void reserve(const size_t new_capacity, aclrtStream stream = 0) = 0;
    virtual void accum_or_assign(const size_t n,
                                 const void* keys,              // (n)
                                 const void* value_or_deltas,   // (n, DIM)
                                 const bool* accum_or_assigns,  // (n)
                                 const void* scores = nullptr,  // (n)
                                 aclrtStream stream = 0, bool ignore_evict_strategy = false) = 0;
    virtual void find_or_insert(const size_t n, const void* keys, void** value_ptrs, void* values, bool* founds,
                                void* scores = nullptr, aclrtStream stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;
    virtual void find_or_insert_pointers(const size_t n, const void* keys,  // (n)
                                         void** value_ptrs,                 // (n * ptrs)
                                         bool* d_found,                     // (n * 1)
                                         void* scores = nullptr,            // (n)
                                         aclrtStream stream = 0, bool unique_key = true,
                                         bool ignore_evict_strategy = false) = 0;
    virtual void set_initial_optstate(const float value) = 0;
    virtual const float get_initial_optstate() const = 0;
    virtual void assign(const size_t n,
                        const void* keys,              // (n)
                        const void* values,            // (n, DIM)
                        const void* scores = nullptr,  // (n)
                        aclrtStream stream = 0, bool unique_key = true) = 0;
    virtual void lock(const size_t n,
                      const void* keys,        // (n)
                      void** locked_keys_ptr,  // (n)
                      bool* flags = nullptr,   // (n)
                      void* scores = nullptr,  // (n)
                      aclrtStream stream = 0) = 0;
    virtual void unlock(const size_t n,
                        void** locked_keys_ptr,  // (n)
                        const void* keys,        // (n)
                        bool* flags = nullptr,   // (n)
                        aclrtStream stream = 0) = 0;
    virtual void find_pointers(const size_t n, const void* keys,  // (n)
                               void** values,                     // (n)
                               bool* founds,                      // (n)
                               void* scores = nullptr,            // (n)
                               aclrtStream stream = 0) const = 0;
    virtual void find_pointers(const size_t n, const void* keys,  // (n)
                               void** values,                     // (n)
                               bool* founds,                      // (n)
                               void* scores = nullptr,            // (n)
                               aclrtStream stream = 0) = 0;
    virtual int optstate_dim() const = 0;
    virtual int get_emb_cols() const = 0;
    virtual void export_batch(const size_t n, const size_t offset, const torch::Tensor d_counter,
                              const torch::Tensor keys, const torch::Tensor values,
                              const c10::optional<torch::Tensor>& score = c10::nullopt) const = 0;
    virtual void export_batch_matched(const uint64_t threshold, const uint64_t n, const uint64_t offset,
                                      torch::Tensor num_matched, torch::Tensor keys, torch::Tensor values,
                                      const c10::optional<torch::Tensor>& scores = c10::nullopt,
                                      aclrtStream stream = 0) const = 0;
    virtual void count_matched(const uint64_t threshold, torch::Tensor num_matched, aclrtStream stream = 0) const = 0;
    virtual void update(const size_t n, const torch::Tensor keys, const torch::Tensor values,
                        const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
                        bool ignore_evict_strategy = false) = 0;
    virtual void load(const size_t n, const torch::Tensor keys, const torch::Tensor values,
                      const c10::optional<torch::Tensor>& score = c10::nullopt, bool unique_key = true,
                      bool ignore_evict_strategy = false) = 0;
    virtual void find_and_initialize(const size_t n, const void* keys, void** value_ptrs, void* values, bool* d_found,
                                     const c10::optional<InitializerArgs>& initializer_args = c10::nullopt,
                                     aclrtStream stream = 0) = 0;
};
#endif

class VariableFactory {
public:
    static std::shared_ptr<DynamicVariableBase> Create(DataType key_type, DataType value_type, EvictStrategy evict_type,
                                                       int64_t dim, size_t init_capacity, size_t max_capacity,
                                                       size_t max_hbm_for_vectors, size_t max_bucket_size,
                                                       float max_load_factor, int block_size, int io_block_size,
                                                       int device_id, bool io_by_cpu, bool use_constant_memory,
                                                       int reserved_key_start_bit, size_t num_of_buckets_per_alloc,
                                                       const InitializerArgs& initializer_args,
                                                       const SafeCheckMode safe_check_mode = SafeCheckMode::IGNORE,
                                                       const OptimizerType optimizer_type = OptimizerType::Null);
};

}  // namespace dyn_emb
#endif  // DYNAMIC_VARIABLE_BASE_H
