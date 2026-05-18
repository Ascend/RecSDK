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
#ifdef USE_RTTI
DynamicVariableBase::DynamicVariableBase(
    RowsFn rows_fn, ColsFn cols_fn, GetMaxCapacityFn get_max_capacity_fn,
    GetKeyTypeFn get_key_type_fn, GetValueTypeFn get_value_type_fn,
    GetEvictStrategyFn get_evict_strategy_fn,
    GetInitializerArgsFn get_initializer_args_fn,
    InsertAndEvictFn insert_and_evict_fn, FindFn find_fn, EraseFn erase_fn,
    ClearFn clear_fn, ReserveFn reserve_fn, AccumOrAssignFn accum_or_assign_fn,
    FindOrInsertPointersFn find_or_insert_pointers_fn,
    AssignFn assign_fn, LockFn lock_fn, UnlockFn unlock_fn,
    FindPointersConstFn find_pointers_const_fn, FindPointersFn find_pointers_fn,
    OptStateDimFn optstate_dim_fn, GetEmbColsFn get_emb_cols_fn,
    ExportBatchFn export_batch_fn, ExportBatchMatchedFn export_batch_matched_fn,
    CountMatchedFn count_matched_fn, UpdateFn update_fn, LoadFn load_fn)
    : rows_fn_(std::move(rows_fn)),
      cols_fn_(std::move(cols_fn)),
      get_max_capacity_fn_(std::move(get_max_capacity_fn)),
      get_key_type_fn_(std::move(get_key_type_fn)),
      get_value_type_fn_(std::move(get_value_type_fn)),
      get_evict_strategy_fn_(std::move(get_evict_strategy_fn)),
      get_initializer_args_fn_(std::move(get_initializer_args_fn)),
      insert_and_evict_fn_(std::move(insert_and_evict_fn)),
      find_fn_(std::move(find_fn)),
      erase_fn_(std::move(erase_fn)),
      clear_fn_(std::move(clear_fn)),
      reserve_fn_(std::move(reserve_fn)),
      accum_or_assign_fn_(std::move(accum_or_assign_fn)),
      find_or_insert_pointers_fn_(std::move(find_or_insert_pointers_fn)),
      assign_fn_(std::move(assign_fn)),
      lock_fn_(std::move(lock_fn)),
      unlock_fn_(std::move(unlock_fn)),
      find_pointers_const_fn_(std::move(find_pointers_const_fn)),
      find_pointers_fn_(std::move(find_pointers_fn)),
      optstate_dim_fn_(std::move(optstate_dim_fn)),
      get_emb_cols_fn_(std::move(get_emb_cols_fn)),
      export_batch_fn_(std::move(export_batch_fn)),
      export_batch_matched_fn_(std::move(export_batch_matched_fn)),
      count_matched_fn_(std::move(count_matched_fn)),
      update_fn_(std::move(update_fn)),
      load_fn_(std::move(load_fn)) {}

int64_t DynamicVariableBase::rows(aclrtStream stream) {
  return rows_fn_(stream);
}

int64_t DynamicVariableBase::cols() {
    return cols_fn_();
}

int64_t DynamicVariableBase::get_max_capacity() {
  return get_max_capacity_fn_();
}

DataType DynamicVariableBase::get_key_type() { return get_key_type_fn_(); }

DataType DynamicVariableBase::get_value_type() { return get_value_type_fn_(); }

EvictStrategy DynamicVariableBase::get_evict_strategy() const {
  return get_evict_strategy_fn_();
}

const InitializerArgs& DynamicVariableBase::get_initializer_args() const {
  return get_initializer_args_fn_();
}

EvictStrategy DynamicVariableBase::evict_strategy() const {
  return get_evict_strategy_fn_();
}

void DynamicVariableBase::insert_and_evict(
    const size_t n, const void* keys, const void* values, const void* scores,
    void* evicted_keys, void* evicted_values, void* evicted_scores,
    uint64_t* d_evicted_counter, aclrtStream stream, bool unique_key,
    bool ignore_evict_strategy) {
  insert_and_evict_fn_(n, keys, values, scores, evicted_keys, evicted_values,
                       evicted_scores, d_evicted_counter, stream, unique_key,
                       ignore_evict_strategy);
}

void DynamicVariableBase::find(const size_t n, const void* keys, void* values,
                               bool* founds, void* scores,
                               aclrtStream stream) const {
  find_fn_(n, keys, values, founds, scores, stream);
}

void DynamicVariableBase::erase(const size_t n, const void* keys,
                                aclrtStream stream) {
  erase_fn_(n, keys, stream);
}

void DynamicVariableBase::clear(aclrtStream stream) { clear_fn_(stream); }

void DynamicVariableBase::reserve(const size_t new_capacity,
                                  aclrtStream stream) {
  reserve_fn_(new_capacity, stream);
}

void DynamicVariableBase::accum_or_assign(const size_t n, const void* keys,
                                          const void* value_or_deltas,
                                          const bool* accum_or_assigns,
                                          const void* scores,
                                          aclrtStream stream,
                                          bool ignore_evict_strategy) {
  accum_or_assign_fn_(n, keys, value_or_deltas, accum_or_assigns, scores,
                      stream, ignore_evict_strategy);
}

void DynamicVariableBase::find_or_insert_pointers(const size_t n, const void *keys,
    void **value_ptrs,
    bool *d_found,
    void *scores,
    aclrtStream stream,
    bool unique_key,
    bool ignore_evict_strategy) {
    find_or_insert_pointers_fn_(n, keys, value_ptrs, d_found, scores,
                                stream, unique_key, ignore_evict_strategy);
}

void DynamicVariableBase::assign(const size_t n, const void* keys,
                                 const void* values, const void* scores,
                                 aclrtStream stream, bool unique_key) {
  assign_fn_(n, keys, values, scores, stream, unique_key);
}

void DynamicVariableBase::lock(const size_t n, const void* keys,
                               void** locked_keys_ptr, bool* flags,
                               void* scores, aclrtStream stream) {
  lock_fn_(n, keys, locked_keys_ptr, flags, scores, stream);
}

void DynamicVariableBase::unlock(const size_t n, void** locked_keys_ptr,
                                 const void* keys, bool* flags,
                                 aclrtStream stream) {
  unlock_fn_(n, locked_keys_ptr, keys, flags, stream);
}

void DynamicVariableBase::find_pointers(const size_t n, const void* keys,
                                        void** values, bool* founds,
                                        void* scores,
                                        aclrtStream stream) const {
  find_pointers_const_fn_(n, keys, values, founds, scores, stream);
}

void DynamicVariableBase::find_pointers(const size_t n, const void* keys,
                                        void** values, bool* founds,
                                        void* scores, aclrtStream stream) {
  find_pointers_fn_(n, keys, values, founds, scores, stream);
}

int DynamicVariableBase::optstate_dim() const { return optstate_dim_fn_(); }

int DynamicVariableBase::get_emb_cols() const { return get_emb_cols_fn_(); }

void DynamicVariableBase::export_batch(
    const size_t n, const size_t offset, const torch::Tensor d_counter,
    const torch::Tensor keys, const torch::Tensor values,
    const c10::optional<torch::Tensor>& score) const {
  export_batch_fn_(n, offset, d_counter, keys, values, score);
}

void DynamicVariableBase::export_batch_matched(
    const uint64_t threshold, const uint64_t n, const uint64_t offset,
    torch::Tensor num_matched, torch::Tensor keys, torch::Tensor values,
    const c10::optional<torch::Tensor>& scores, aclrtStream stream) const {
  export_batch_matched_fn_(threshold, n, offset, num_matched, keys, values,
                           scores, stream);
}

void DynamicVariableBase::count_matched(const uint64_t threshold,
                                        torch::Tensor num_matched,
                                        aclrtStream stream) const {
  count_matched_fn_(threshold, num_matched, stream);
}

void DynamicVariableBase::update(const size_t n, const torch::Tensor keys,
                                 const torch::Tensor values,
                                 const c10::optional<torch::Tensor>& score,
                                 bool unique_key, bool ignore_evict_strategy) {
  update_fn_(n, keys, values, score, unique_key, ignore_evict_strategy);
}

void DynamicVariableBase::load(const size_t n, const torch::Tensor keys,
                               const torch::Tensor values,
                               const c10::optional<torch::Tensor>& score,
                               bool unique_key, bool ignore_evict_strategy) {
  load_fn_(n, keys, values, score, unique_key, ignore_evict_strategy);
}

std::shared_ptr<DynamicVariableBase> VariableFactory::Create(
    DataType key_type, DataType value_type, EvictStrategy evict_type,
    int64_t dim, size_t init_capacity, size_t max_capacity,
    size_t max_hbm_for_vectors, size_t max_bucket_size, float max_load_factor,
    int block_size, int io_block_size, int device_id, bool io_by_cpu,
    bool use_constant_memory, int reserved_key_start_bit,
    size_t num_of_buckets_per_alloc, const InitializerArgs& initializer_args,
    const SafeCheckMode safe_check_mode, const OptimizerType optimizer_type) {
  std::shared_ptr<DynamicVariableBase> table;
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, keyT, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, valueT, [&] {
      DISPATCH_EVICTYPE_FUNCTION(evict_type, evictT, [&] {
        auto impl = std::make_shared<HKVVariable<keyT, valueT, evictT>>(
            key_type, value_type, dim, init_capacity, max_capacity,
            max_hbm_for_vectors, max_bucket_size, max_load_factor, block_size,
            io_block_size, device_id, io_by_cpu, use_constant_memory,
            reserved_key_start_bit, num_of_buckets_per_alloc, initializer_args,
            safe_check_mode, optimizer_type);

        table = std::make_shared<DynamicVariableBase>(
            [impl](aclrtStream s) { return impl->rows(s); },
            [impl]() { return impl->cols(); },
            [impl]() { return impl->get_max_capacity(); },
            [impl]() { return impl->get_key_type(); },
            [impl]() { return impl->get_value_type(); },
            [impl]() { return impl->get_evict_strategy(); },
            [impl]() -> const InitializerArgs& {
              return impl->get_initializer_args();
            },
            [impl](const size_t n, const void* keys, const void* values,
                   const void* scores, void* evicted_keys, void* evicted_values,
                   void* evicted_scores, uint64_t* d_evicted_counter,
                   aclrtStream stream, bool unique_key,
                   bool ignore_evict_strategy) {
              impl->insert_and_evict(n, keys, values, scores, evicted_keys,
                                     evicted_values, evicted_scores,
                                     d_evicted_counter, stream, unique_key,
                                     ignore_evict_strategy);
            },
            [impl](const size_t n, const void* keys, void* values, bool* founds,
                   void* scores, aclrtStream stream) {
              impl->find(n, keys, values, founds, scores, stream);
            },
            [impl](const size_t n, const void* keys, aclrtStream stream) {
              impl->erase(n, keys, stream);
            },
            [impl](aclrtStream stream) { impl->clear(stream); },
            [impl](const size_t new_capacity, aclrtStream stream) {
              impl->reserve(new_capacity, stream);
            },
            [impl](const size_t n, const void* keys,
                   const void* value_or_deltas, const bool* accum_or_assigns,
                   const void* scores, aclrtStream stream,
                   bool ignore_evict_strategy) {
              impl->accum_or_assign(n, keys, value_or_deltas, accum_or_assigns,
                                    scores, stream, ignore_evict_strategy);
            },
            [impl](const size_t n, const void *keys, void **value_ptrs, bool *d_found, void *scores,
                   aclrtStream stream, bool unique_key, bool ignore_evict_strategy) {
                impl->find_or_insert_pointers(n, keys, value_ptrs, d_found, scores,
                        stream, unique_key, ignore_evict_strategy);
            },
            [impl](const size_t n, const void* keys, const void* values,
                   const void* scores, aclrtStream stream, bool unique_key) {
              impl->assign(n, keys, values, scores, stream, unique_key);
            },
            [impl](const size_t n, const void* keys, void** locked_keys_ptr,
                   bool* flags, void* scores, aclrtStream stream) {
              impl->lock(n, keys, locked_keys_ptr, flags, scores, stream);
            },
            [impl](const size_t n, void** locked_keys_ptr, const void* keys,
                   bool* flags, aclrtStream stream) {
              impl->unlock(n, locked_keys_ptr, keys, flags, stream);
            },
            [impl](const size_t n, const void* keys, void** values,
                   bool* founds, void* scores, aclrtStream stream) {
              const auto& cimpl = *impl;
              cimpl.find_pointers(n, keys, values, founds, scores, stream);
            },
            [impl](const size_t n, const void* keys, void** values,
                   bool* founds, void* scores, aclrtStream stream) {
              impl->find_pointers(n, keys, values, founds, scores, stream);
            },
            [impl]() { return impl->optstate_dim(); },
            [impl]() { return impl->get_emb_cols(); },
            [impl](const size_t n, const size_t offset,
                   const torch::Tensor d_counter, const torch::Tensor keys,
                   const torch::Tensor values,
                   const c10::optional<torch::Tensor>& score) {
              impl->export_batch(n, offset, d_counter, keys, values, score);
            },
            [impl](const uint64_t threshold, const uint64_t n,
                   const uint64_t offset, torch::Tensor num_matched,
                   torch::Tensor keys, torch::Tensor values,
                   const c10::optional<torch::Tensor>& scores,
                   aclrtStream stream) {
              impl->export_batch_matched(threshold, n, offset, num_matched,
                                         keys, values, scores, stream);
            },
            [impl](const uint64_t threshold, torch::Tensor num_matched,
                   aclrtStream stream) {
              impl->count_matched(threshold, num_matched, stream);
            },
            [impl](const size_t n, const torch::Tensor keys,
                   const torch::Tensor values,
                   const c10::optional<torch::Tensor>& score, bool unique_key,
                   bool ignore_evict_strategy) {
              impl->update(n, keys, values, score, unique_key,
                           ignore_evict_strategy);
            },
            [impl](const size_t n, const torch::Tensor keys,
                   const torch::Tensor values,
                   const c10::optional<torch::Tensor>& score, bool unique_key,
                   bool ignore_evict_strategy) {
              impl->load(n, keys, values, score, unique_key,
                         ignore_evict_strategy);
            });
      });
    });
  });
  return table;
}
#else
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
#endif
}  // namespace dyn_emb
