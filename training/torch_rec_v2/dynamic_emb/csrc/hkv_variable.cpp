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
#include <random>
#include <acl/acl.h>
#include "tiling/platform/platform_ascendc.h"
#include <simt_api/common_functions.h>
#include <simt_api/math_functions.h>
#include "kernel_operator.h"
#include "table_vector.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "ops/check_safe_pointers/check_safe_pointers_kernel.h"
#include "ops/load_or_initialize_embeddings/load_or_initialize_embeddings_kernel.h"
#include "ops/initialize_optimizer_state/initialize_optimizer_state_simd_kernel.h"
#include "ops/load_or_initialize_embeddings/load_or_initialize_embeddings_hybrid_kernel.h"
#include "ops/fill_output_with_table_vector/fill_output_with_table_vector_hybrid_kernel.h"
#include "acl_singleton.h"
#include "tiling_helper.h"
#include "utils.h"
#include "initializer_generators.h"
#include "initializer_kernel_ops.h"

namespace dyn_emb {
constexpr uint32_t BLOCK_THREAD_NUM_OPT = 2048;
// 当前无接口获取每个block下最大线程数，查询手册Ascend950PR该值为2048
constexpr uint32_t MAX_THREADS_PER_BLOCK = 2048;

DeviceProp& DeviceProp::getDeviceProp(int device_id)
{
    static DeviceProp device_prop(device_id);
    return device_prop;
}

DeviceProp::DeviceProp(int device_id)
{
    uint32_t deviceCount = 0;
    if (aclrtGetDeviceCount(&deviceCount) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceCount failed");
    }

    if (deviceCount == 0 || device_id < 0 || device_id >= deviceCount) {
        throw std::runtime_error("Can't get device count, or device_id < 0, or device_id >= deviceCount, device_id = " +
                                 std::to_string(device_id) + ", deviceCount = " + std::to_string(deviceCount));
    }

    int64_t vectorCoreCount = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreCount) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceInfo(VECTOR_CORE_NUM) failed");
    }
    this->num_sms = vectorCoreCount;

    int64_t warpSize = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_WARP_SIZE, &warpSize) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceInfo(WARP_SIZE) failed");
    }
    this->warp_size = warpSize;

    int64_t maxThreadPerVectorCore = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, &maxThreadPerVectorCore) !=
        ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceInfo(MAX_THREAD_PER_VECTOR_CORE) failed");
    }
    this->max_thread_per_sm = maxThreadPerVectorCore;
    this->max_thread_per_block = MAX_THREADS_PER_BLOCK;
    this->total_threads = this->num_sms * this->max_thread_per_sm;
}

DeviceCounter::DeviceCounter()
{
    check_ret(aclrtMalloc(reinterpret_cast<void**>(&d_counter), sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc d_counter failed.");
}

DeviceCounter::~DeviceCounter()
{
    check_ret(aclrtFree(d_counter), "aclrtFree d_counter failed.");
}

DeviceCounter& DeviceCounter::reset(const aclrtStream& stream)
{
    (void)stream;
    check_ret(aclrtMemset(d_counter, sizeof(uint64_t), 0, sizeof(uint64_t)), "aclrtMemset d_counter failed.");
    return *this;
}

uint64_t* DeviceCounter::get()
{
    return d_counter;
}

DeviceCounter& DeviceCounter::sync(const aclrtStream& stream)
{
    check_ret(
        aclrtMemcpyAsync(&h_counter, sizeof(uint64_t), d_counter, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST, stream),
        "aclrtMemcpyAsync d_counter to h_counter failed.");
    check_ret(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream failed.");
    return *this;
}

uint64_t DeviceCounter::result()
{
    return h_counter;
}

void DeviceCounter::check_ret(aclError ret, const char* msg)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

template <typename K, typename V, typename S>
struct EvalAndInc {
    __gm__ uint64_t* d_count;
    S threshold;
    EvalAndInc(S threshold, __gm__ uint64_t* d_count) : threshold(threshold), d_count(d_count) {}
    __simt_callee__ void operator()(const K& key, __gm__ V* value, __gm__ S* score, int32_t)
    {
        S score_val = *score;
        bool match = (!npu::hkv::IS_RESERVED_KEY(key) && score_val >= threshold);
        uint32_t vote = asc_ballot(match);
        int32_t group_count = AscendC::Simt::Popc(vote);
        if (threadIdx.x % warpSize == 0) {
            atomicAdd(d_count, group_count);
        }
    }
};

template <class K, class V, class S>
struct ExportIfPredFunctor {
    S threshold;
    ExportIfPredFunctor(S threshold) : threshold(threshold) {}
    template <int GroupSize>
    __forceinline__ __simt_callee__ bool operator()(const K& key, const __gm__ V* value, const S& score)
    {
        return ((!npu::hkv::IS_RESERVED_KEY<K>(key)) && (score >= threshold));
    }
};

template <typename T, typename EmbeddingGenerator, typename TableVector>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_OPT) inline void fill_output_with_table_vectors_kernel_vf(
    uint64_t n, int emb_dim, __gm__ T* outputs, typename TableVector::Args vector_args,
    typename EmbeddingGenerator::Args generator_args)
{
    TableVector vectors(vector_args);
    EmbeddingGenerator emb_gen(generator_args);
    for (int64_t emb_id = AscendC::Simt::GetBlockIdx(); emb_id < n; emb_id += AscendC::Simt::GetBlockNum()) {
        if (vectors.isInitialized(emb_id)) {  // copy embedding from table to outputs.
            for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
                outputs[emb_id * emb_dim + i] = *vectors.data_ptr(emb_id, i);
            }
        } else if (vectors.isValid(emb_id)) {  // initialize the embedding as well as outputs.
            for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
                auto tmp = emb_gen.generate(emb_id);
                outputs[emb_id * emb_dim + i] = TypeConvertFunc<T, float>::convert(tmp);
                *vectors.data_ptr(emb_id, i) = TypeConvertFunc<T, float>::convert(tmp);
            }
        } else {  // vector not exists in table, set the output to 0.
            for (int i = threadIdx.x; i < emb_dim; i += blockDim.x) {
                outputs[emb_id * emb_dim + i] = TypeConvertFunc<T, float>::convert(0.0f);
            }
        }
    }
    emb_gen.destroy();
}

template <typename T, typename EmbeddingGenerator, typename TableVector>
__global__ __vector__ void fill_output_with_table_vectors_kernel(uint64_t n, int emb_dim, __gm__ T* outputs,
                                                                 typename TableVector::Args vector_args,
                                                                 typename EmbeddingGenerator::Args generator_args)
{
    asc_vf_call<fill_output_with_table_vectors_kernel_vf<T, EmbeddingGenerator, TableVector>>(
        dim3{BLOCK_THREAD_NUM_OPT}, n, emb_dim, outputs, vector_args, generator_args);
}

template <typename T, typename OptStateInitializer, typename TableVector>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_OPT) inline void initialize_optimizer_state_vf_vec4(
    uint64_t n, int emb_dim, typename TableVector::Args vector_args, OptStateInitializer optstate_initializer,
    const uint64_t warp_num_all)
{
    TableVector vectors(vector_args);

    T initial_optstate = TypeConvertFunc<T, float>::convert(optstate_initializer.initial_optstate);
    int dim = optstate_initializer.dim;

    const int warp_num_per_block = blockDim.x / warpSize;
    const int warp_id_in_block = threadIdx.x / warpSize;

    for (int64_t emb_id = warp_num_per_block * blockIdx.x + warp_id_in_block; emb_id < n; emb_id += warp_num_all) {
        if ((!vectors.isInitialized(emb_id)) and vectors.isValid(emb_id)) {
            OptStateInitializerInit4(vectors.data_ptr(emb_id, emb_dim), dim, initial_optstate);
        }
    }
}

template <typename T, typename OptStateInitializer, typename TableVector>
__global__ __vector__ void initialize_optimizer_state_kernel_vec4(uint64_t n, int emb_dim,
                                                                  typename TableVector::Args vector_args,
                                                                  OptStateInitializer optstate_initializer)
{
    const int warp_num_per_block = BLOCK_THREAD_NUM_OPT / warpSize;
    const uint64_t warp_num_all = static_cast<uint64_t>(warp_num_per_block) * GetBlockNum();
    asc_vf_call<initialize_optimizer_state_vf_vec4<T, OptStateInitializer, TableVector>>(
        dim3{BLOCK_THREAD_NUM_OPT}, n, emb_dim, vector_args, optstate_initializer, warp_num_all);
}

template <typename T, typename OptStateInitializer, typename TableVector>
__simt_vf__ __aicore__ LAUNCH_BOUND(BLOCK_THREAD_NUM_OPT) inline void initialize_optimizer_state_vf(
    uint64_t n, int emb_dim, typename TableVector::Args vector_args, OptStateInitializer optstate_initializer,
    const uint64_t block_all)
{
    T initial_optstate = TypeConvertFunc<T, float>::convert(optstate_initializer.initial_optstate);
    int dim = optstate_initializer.dim;

    TableVector vectors(vector_args);

    for (int64_t emb_id = blockIdx.x; emb_id < n; emb_id += block_all) {
        if ((!vectors.isInitialized(emb_id)) and vectors.isValid(emb_id)) {
            OptStateInitializerInit(vectors.data_ptr(emb_id, emb_dim), dim, initial_optstate);
        }
    }
}

template <typename T, typename OptStateInitializer, typename TableVector>
__global__ __vector__ void initialize_optimizer_state_kernel(uint64_t n, int emb_dim,
                                                             typename TableVector::Args vector_args,
                                                             OptStateInitializer optstate_initializer)
{
    const uint64_t block_all = GetBlockNum();
    asc_vf_call<initialize_optimizer_state_vf<T, OptStateInitializer, TableVector>>(
        dim3{BLOCK_THREAD_NUM_OPT}, n, emb_dim, vector_args, optstate_initializer, block_all);
}

template <typename ValueType, typename Generator>
void launch_load_or_initialize_embeddings_kernel(size_t n, int dim, int32_t max_cores, void* values, void** value_ptrs,
                                                 bool* d_found, typename Generator::Args generator_args,
                                                 aclrtStream stream, bool use_pure_hbm_kernel)
{
    if (!use_pure_hbm_kernel) {
        auto tiling =
            npu::hkv::GetValueMoveTiling(n, static_cast<uint32_t>(max_cores), static_cast<uint32_t>(dim),
                                         sizeof(ValueType), false, npu::hkv::DOUBLE_BUFFER * npu::hkv::DOUBLE_BUFFER);
        load_or_initialize_embeddings_hybrid_kernel<ValueType, Generator><<<max_cores, tiling.valid_ub_size, stream>>>(
            tiling.former_num, tiling.former_core_move_num, tiling.tail_core_move_num, tiling.tile_size,
            tiling.num_tiles, static_cast<uint32_t>(dim), reinterpret_cast<ValueType*>(values),
            reinterpret_cast<ValueType**>(value_ptrs), d_found, generator_args);
        return;
    }

    load_or_initialize_embeddings_kernel<ValueType, Generator>
        <<<max_cores, 0, stream>>>(n, dim, reinterpret_cast<ValueType*>(values),
                                   reinterpret_cast<ValueType**>(value_ptrs), d_found, generator_args);
}

template <typename ValueType, typename Generator>
void launch_fill_output_with_table_vectors_kernel(size_t n, int dim, int32_t max_cores, void* values, void** value_ptrs,
                                                  bool* d_found, typename Generator::Args generator_args,
                                                  aclrtStream stream, bool use_pure_hbm_kernel)
{
    if (!use_pure_hbm_kernel) {
        auto vector_args =
            typename TableVectorSimd<ValueType>::Args{reinterpret_cast<ValueType**>(value_ptrs), d_found};
        auto tiling =
            npu::hkv::GetValueMoveTiling(n, static_cast<uint32_t>(max_cores), static_cast<uint32_t>(dim),
                                         sizeof(ValueType), false, npu::hkv::DOUBLE_BUFFER * npu::hkv::DOUBLE_BUFFER);
        fill_output_with_table_vectors_hybrid_kernel<ValueType, Generator><<<max_cores, tiling.valid_ub_size, stream>>>(
            tiling.former_num, tiling.former_core_move_num, tiling.tail_core_move_num, tiling.tile_size,
            tiling.num_tiles, static_cast<uint32_t>(dim), reinterpret_cast<ValueType*>(values), vector_args,
            generator_args);
        return;
    }

    using TableVectorType = TableVector<ValueType>;
    auto table_vec_args = typename TableVectorType::Args{reinterpret_cast<ValueType**>(value_ptrs), d_found};
    fill_output_with_table_vectors_kernel<ValueType, Generator, TableVectorType>
        <<<max_cores, 0, stream>>>(n, dim, reinterpret_cast<ValueType*>(values), table_vec_args, generator_args);
}

template <typename T>
void check_safe_pointers_sync(const uint64_t n, const T** ptrs, const SafeCheckMode safe_check_mode,
                              const aclrtStream& stream)
{
    if (n == 0) {
        return;
    }
    static DeviceCounter counter;
    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    check_safe_pointers_kernel<T><<<maxCores, 0, stream>>>(n, ptrs, counter.reset(stream).get());

    auto result = counter.sync(stream).result();
    if (result == 0) {
        std::cout << "#DynamicEmb LOG: All indices in current batch size " << n << " have legal pointers.\n";
        return;
    }
    std::stringstream ss;
    ss << "#DynamicEmb ERROR: Failed to insert " << result << " indices in current batch size " << n << ". "
       << "Consider expanding the capacity/num_embedding.\n";
    if (safe_check_mode == SafeCheckMode::WARNING) {
        std::cerr << ss.str();
    } else if (safe_check_mode == SafeCheckMode::ERROR) {
        throw std::runtime_error(ss.str());
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
HKVVariable<KeyType, ValueType, Strategy>::HKVVariable(
    DataType key_type, DataType value_type, int64_t dim, int64_t init_capacity, size_t max_capacity,
    size_t max_hbm_for_vectors, size_t max_bucket_size, float max_load_factor, int block_size, int io_block_size,
    int device_id, bool io_by_cpu, bool use_constant_memory, int reserved_key_start_bit,
    size_t num_of_buckets_per_alloc, const InitializerArgs& initializer_args_, const SafeCheckMode safe_check_mode,
    const OptimizerType optimizer_type)
    : dim_(dim),
      max_capacity_(max_capacity),
      initializer_args_(initializer_args_),
      curand_states_(nullptr),
      key_type_(key_type),
      value_type_(value_type),
      safe_check_mode_(safe_check_mode),
      optimizer_type_(optimizer_type)
{
    if (dim <= 0) {
        throw std::invalid_argument("dimension must > 0 but got " + std::to_string(dim));
    }

    uint32_t deviceCount;
    NPU_CHECK(aclrtGetDeviceCount(&deviceCount));
    if (device_id < 0 || device_id >= deviceCount) {
        throw std::invalid_argument("Invalid device id, device id is ." + std::to_string(device_id));
    } else {
        NPU_CHECK(aclrtSetDevice(static_cast<int32_t>(device_id)));
        // Init global device property.
        DeviceProp::getDeviceProp(device_id);
    }
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    alloc_curand_states(&curand_states_, stream);

    hkv_table_option_.init_capacity = init_capacity;
    hkv_table_option_.max_capacity = max_capacity;
    hkv_table_option_.dim = dim + get_optimizer_state_dim<ValueType>(optimizer_type, dim);
    int64_t max_hbm_needed = hkv_table_option_.max_capacity * hkv_table_option_.dim * sizeof(ValueType);
    hkv_table_option_.max_hbm_for_vectors = max_hbm_needed < max_hbm_for_vectors ? max_hbm_needed : max_hbm_for_vectors;
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
HKVVariable<KeyType, ValueType, Strategy>::~HKVVariable()
{
    free_curand_states(curand_states_);
    curand_states_ = nullptr;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int64_t HKVVariable<KeyType, ValueType, Strategy>::rows(aclrtStream stream)
{
    return hkv_table_->size(stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
int64_t HKVVariable<KeyType, ValueType, Strategy>::cols()
{
    return dim_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
bool HKVVariable<KeyType, ValueType, Strategy>::is_pure_hbm_mode() const
{
    return hkv_table_->is_pure_hbm_mode();
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
EvictStrategy HKVVariable<KeyType, ValueType, Strategy>::evict_strategy() const
{
    return Strategy;
}

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
const InitializerArgs& HKVVariable<KeyType, ValueType, Strategy>::get_initializer_args() const
{
    return initializer_args_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::insert_and_evict(const size_t n, const void* keys, const void* values,
                                                                 const void* scores, void* evicted_keys,
                                                                 void* evicted_values, void* evicted_scores,
                                                                 uint64_t* d_evicted_counter, aclrtStream stream,
                                                                 bool unique_key, bool ignore_evict_strategy)
{
    hkv_table_->insert_and_evict(n, reinterpret_cast<const KeyType*>(keys), reinterpret_cast<const ValueType*>(values),
                                 reinterpret_cast<const uint64_t*>(scores), reinterpret_cast<KeyType*>(evicted_keys),
                                 reinterpret_cast<ValueType*>(evicted_values),
                                 reinterpret_cast<uint64_t*>(evicted_scores), d_evicted_counter, stream, unique_key,
                                 ignore_evict_strategy);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find(const size_t n, const void* keys, void* values, bool* founds,
                                                     void* scores, aclrtStream stream) const
{
    hkv_table_->find(n, reinterpret_cast<const KeyType*>(keys), reinterpret_cast<ValueType*>(values), founds,
                     reinterpret_cast<uint64_t*>(scores), stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::clear(aclrtStream stream)
{
    hkv_table_->clear(stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::erase(const size_t n, const void* keys, aclrtStream stream)
{
    hkv_table_->erase(n, reinterpret_cast<const KeyType*>(keys), stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::reserve(const size_t new_capacity, aclrtStream stream)
{
    hkv_table_->reserve(new_capacity, stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::accum_or_assign(const size_t n, const void* keys,
                                                                const void* value_or_deltas,
                                                                const bool* accum_or_assigns, const void* scores,
                                                                aclrtStream stream, bool ignore_evict_strategy)
{
    hkv_table_->accum_or_assign(n, reinterpret_cast<const KeyType*>(keys),
                                reinterpret_cast<const ValueType*>(value_or_deltas),
                                reinterpret_cast<const bool*>(accum_or_assigns),
                                reinterpret_cast<const uint64_t*>(scores), stream, ignore_evict_strategy);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_or_insert_pointers(const size_t n, const void* keys,
                                                                        void** value_ptrs, bool* d_found, void* scores,
                                                                        aclrtStream stream, bool unique_key,
                                                                        bool ignore_evict_strategy)
{
    if (n == 0) {
        return;
    }
    int64_t dim = cols();
    hkv_table_->find_or_insert(n, reinterpret_cast<const KeyType*>(keys), reinterpret_cast<ValueType**>(value_ptrs),
                               d_found, reinterpret_cast<uint64_t*>(scores), stream, unique_key, ignore_evict_strategy);
    if (this->safe_check_mode_ != SafeCheckMode::IGNORE) {
        auto hkv_ptrs = reinterpret_cast<const ValueType**>(const_cast<const void**>(value_ptrs));
        check_safe_pointers_sync<ValueType>(n, hkv_ptrs, this->safe_check_mode_, stream);
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::assign(const size_t n, const void* keys, const void* values,
                                                       const void* scores, aclrtStream stream, bool unique_key)
{
    hkv_table_->assign(n, reinterpret_cast<const KeyType*>(keys), reinterpret_cast<const ValueType*>(values),
                       reinterpret_cast<const uint64_t*>(scores), stream, unique_key);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::lock(const size_t n,
                                                     const void* keys,        // (n)
                                                     void** locked_keys_ptr,  // (n)
                                                     bool* flags,             // (n)
                                                     void* scores, aclrtStream stream)
{
    hkv_table_->lock_keys(n, reinterpret_cast<const KeyType*>(keys), reinterpret_cast<KeyType**>(locked_keys_ptr),
                          flags, stream, reinterpret_cast<const uint64_t*>(scores));
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::unlock(const size_t n,
                                                       void** locked_keys_ptr,  // (n)
                                                       const void* keys,        // (n)
                                                       bool* flags,             // (n)
                                                       aclrtStream stream)
{
    hkv_table_->unlock_keys(n, reinterpret_cast<KeyType**>(locked_keys_ptr), reinterpret_cast<const KeyType*>(keys),
                            flags, stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
curandState* HKVVariable<KeyType, ValueType, Strategy>::get_curand_states() const
{
    return curand_states_;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_pointers(const size_t n, const void* keys, void** value_ptrs,
                                                              bool* founds, void* scores, aclrtStream stream) const
{
    if (n == 0) {
        return;
    }

    hkv_table_->find(n, (KeyType*)keys, (ValueType**)value_ptrs, founds, (uint64_t*)scores, stream);
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_pointers(const size_t n, const void* keys, void** value_ptrs,
                                                              bool* founds, void* scores, aclrtStream stream)
{
    if (n == 0) {
        return;
    }

    hkv_table_->find_and_update(n, (KeyType*)keys, (ValueType**)value_ptrs, founds, (uint64_t*)scores, stream);
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
        hkv_table_->export_batch(n, offset, d_counter.data_ptr<size_t>(), reinterpret_cast<KeyType*>(keys.data_ptr()),
                                 reinterpret_cast<ValueType*>(values.data_ptr()),
                                 reinterpret_cast<uint64_t*>(score_.data_ptr()), stream);
    } else {
        hkv_table_->export_batch(n, offset, d_counter.data_ptr<size_t>(), reinterpret_cast<KeyType*>(keys.data_ptr()),
                                 reinterpret_cast<ValueType*>(values.data_ptr()), nullptr, stream);
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::export_batch_matched(const uint64_t threshold, const uint64_t n,
                                                                     const uint64_t offset, torch::Tensor num_matched,
                                                                     torch::Tensor keys, torch::Tensor values,
                                                                     const c10::optional<torch::Tensor>& scores,
                                                                     aclrtStream stream) const
{
    using PredFunc = ExportIfPredFunctor<KeyType, ValueType, uint64_t>;
    PredFunc func(threshold);
    if (scores.has_value()) {
        at::Tensor scores_ = scores.value();
        hkv_table_->export_batch_if_v2(func, n, offset, reinterpret_cast<uint64_t*>(num_matched.data_ptr()),
                                       reinterpret_cast<KeyType*>(keys.data_ptr()),
                                       reinterpret_cast<ValueType*>(values.data_ptr()),
                                       reinterpret_cast<uint64_t*>(scores_.data_ptr()), stream);
    } else {
        hkv_table_->export_batch_if_v2(func, n, offset, reinterpret_cast<uint64_t*>(num_matched.data_ptr()),
                                       reinterpret_cast<KeyType*>(keys.data_ptr()),
                                       reinterpret_cast<ValueType*>(values.data_ptr()), nullptr, stream);
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::count_matched(const uint64_t threshold, torch::Tensor num_matched,
                                                              aclrtStream stream) const
{
    using ExecutionFunc = EvalAndInc<KeyType, ValueType, uint64_t>;
    ExecutionFunc func(threshold, reinterpret_cast<uint64_t*>(num_matched.data_ptr()));
    hkv_table_->for_each(0, hkv_table_->capacity(), func, stream);
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
        hkv_table_->insert_or_assign(
            n, reinterpret_cast<KeyType*>(keys.data_ptr()), reinterpret_cast<ValueType*>(values.data_ptr()),
            reinterpret_cast<uint64_t*>(score_.data_ptr()), stream, unique_key, ignore_evict_strategy);
    } else {
        hkv_table_->insert_or_assign(n, reinterpret_cast<KeyType*>(keys.data_ptr()),
                                     reinterpret_cast<ValueType*>(values.data_ptr()), nullptr, stream, unique_key,
                                     ignore_evict_strategy);
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_and_initialize(
    const size_t n, const void* keys, void** value_ptrs, void* values, bool* d_found,
    const c10::optional<InitializerArgs>& initializer_args, aclrtStream stream)
{
    if (n == 0) {
        return;
    }
    int dim = dim_;
    const_cast<const HKVVariable<KeyType, ValueType, Strategy>*>(this)->find_pointers(n, keys, value_ptrs, d_found,
                                                                                      nullptr, stream);

    auto& init_args = initializer_args.has_value() ? initializer_args.value() : initializer_args_;
    auto& initializer_mode = init_args.mode_;
    int32_t max_cores = AclSingleton::GetInstance().GetMaxCores();
    auto use_pure_hbm_kernel = is_pure_hbm_mode();

    if (initializer_mode == "normal") {
        using Generator = NormalEmbeddingGenerator;
        auto generator_args = typename Generator::Args{curand_states_, init_args.mean_, init_args.std_dev_};
        launch_load_or_initialize_embeddings_kernel<ValueType, Generator>(
            n, dim, max_cores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_mode == "truncated_normal") {
        using Generator = TruncatedNormalEmbeddingGenerator;
        auto generator_args = typename Generator::Args{curand_states_, init_args.mean_, init_args.std_dev_,
                                                       init_args.lower_, init_args.upper_};
        launch_load_or_initialize_embeddings_kernel<ValueType, Generator>(
            n, dim, max_cores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_mode == "uniform") {
        using Generator = UniformEmbeddingGenerator;
        auto generator_args = typename Generator::Args{curand_states_, init_args.lower_, init_args.upper_};
        launch_load_or_initialize_embeddings_kernel<ValueType, Generator>(
            n, dim, max_cores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_mode == "debug") {
        using Generator = MappingEmbeddingGenerator<KeyType>;
        // Debug initializer maps each key to key % 100000 for deterministic output.
        auto generator_args = typename Generator::Args{reinterpret_cast<const KeyType*>(keys), 100000};
        launch_load_or_initialize_embeddings_kernel<ValueType, Generator>(
            n, dim, max_cores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_mode == "constant") {
        using Generator = ConstEmbeddingGenerator;
        auto generator_args = typename Generator::Args{init_args.value_};
        launch_load_or_initialize_embeddings_kernel<ValueType, Generator>(
            n, dim, max_cores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else {
        throw std::runtime_error("Unrecognized initializer {" + initializer_mode + "}");
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

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::find_or_insert(const size_t n, const void* keys, void** value_ptrs,
                                                               void* values, bool* d_found, void* scores,
                                                               aclrtStream stream, bool unique_key,
                                                               bool ignore_evict_strategy)
{
    if (n == 0) {
        return;
    }
    int64_t dim = cols();
    hkv_table_->find_or_insert(n, reinterpret_cast<const KeyType*>(keys), reinterpret_cast<ValueType**>(value_ptrs),
                               d_found, reinterpret_cast<uint64_t*>(scores), stream, unique_key, ignore_evict_strategy);
    if (this->safe_check_mode_ != SafeCheckMode::IGNORE) {
        auto hkv_ptrs = reinterpret_cast<const ValueType**>(const_cast<const void**>(value_ptrs));
        check_safe_pointers_sync<ValueType>(n, hkv_ptrs, this->safe_check_mode_, stream);
    }

    int32_t maxCores = AclSingleton::GetInstance().GetMaxCores();
    using TableVectorType = TableVector<ValueType>;
    auto table_vec_args = typename TableVectorType::Args{reinterpret_cast<ValueType**>(value_ptrs), d_found};
    auto use_pure_hbm_kernel = is_pure_hbm_mode();

    auto& initializer_ = initializer_args_.mode_;
    if (initializer_ == "normal") {
        using Generator = NormalEmbeddingGenerator;
        auto generator_args =
            typename Generator::Args{curand_states_, initializer_args_.mean_, initializer_args_.std_dev_};
        launch_fill_output_with_table_vectors_kernel<ValueType, Generator>(
            n, dim, maxCores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_ == "truncated_normal") {
        using Generator = TruncatedNormalEmbeddingGenerator;
        auto generator_args =
            typename Generator::Args{curand_states_, initializer_args_.mean_, initializer_args_.std_dev_,
                                     initializer_args_.lower_, initializer_args_.upper_};
        launch_fill_output_with_table_vectors_kernel<ValueType, Generator>(
            n, dim, maxCores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_ == "uniform") {
        using Generator = UniformEmbeddingGenerator;
        auto generator_args =
            typename Generator::Args{curand_states_, initializer_args_.lower_, initializer_args_.upper_};
        launch_fill_output_with_table_vectors_kernel<ValueType, Generator>(
            n, dim, maxCores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_ == "debug") {
        using Generator = MappingEmbeddingGenerator<KeyType>;
        auto generator_args = typename Generator::Args{reinterpret_cast<const KeyType*>(keys), 100000};
        launch_fill_output_with_table_vectors_kernel<ValueType, Generator>(
            n, dim, maxCores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else if (initializer_ == "constant") {
        using Generator = ConstEmbeddingGenerator;
        auto generator_args = typename Generator::Args{initializer_args_.value_};
        launch_fill_output_with_table_vectors_kernel<ValueType, Generator>(
            n, dim, maxCores, values, value_ptrs, d_found, generator_args, stream, use_pure_hbm_kernel);
    } else {
        throw std::runtime_error("Unrecognized initializer {" + initializer_ + "}");
    }

    int optstate_dim = get_optimizer_state_dim<ValueType>(optimizer_type_, dim);
    if (optstate_dim == 0) {
        return;
    }

    if (!use_pure_hbm_kernel) {
        const uint32_t buffer_num = 1;  // 不开double_buffer，只有1片内存
        auto tiling = npu::hkv::GetValueMoveTiling(n, maxCores, optstate_dim, sizeof(ValueType), true, buffer_num);
        initialize_optimizer_state_simd_kernel<ValueType><<<maxCores, tiling.valid_ub_size, stream>>>(
            tiling.former_num, tiling.former_core_move_num, tiling.tail_core_move_num, tiling.tile_size,
            tiling.num_tiles, n, dim, reinterpret_cast<ValueType**>(value_ptrs), d_found, optstate_dim,
            initial_optstate_);
    } else {
        using OptStateInitializer = OptStateInitializer<float, int>;
        OptStateInitializer optstate_initializer{optstate_dim, initial_optstate_};
        // 当dim和optstate_dim都是4的倍数时，SIMT算子内部使用Vec4T进行4个元素的向量化优化
        if (dim % 4 == 0 and optstate_dim % 4 == 0) {
            initialize_optimizer_state_kernel_vec4<ValueType, OptStateInitializer, TableVectorType>
                <<<maxCores, 0, stream>>>(n, dim, table_vec_args, optstate_initializer);
        } else {
            initialize_optimizer_state_kernel<ValueType, OptStateInitializer, TableVectorType>
                <<<maxCores, 0, stream>>>(n, dim, table_vec_args, optstate_initializer);
        }
    }
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
void HKVVariable<KeyType, ValueType, Strategy>::set_initial_optstate(const float value)
{
    this->initial_optstate_ = value;
}

template <typename KeyType, typename ValueType, EvictStrategy Strategy>
const float HKVVariable<KeyType, ValueType, Strategy>::get_initial_optstate() const
{
    return this->initial_optstate_;
}

// 2 × 3 × 3 = 18 种组合全部给出实例化
template class HKVVariable<int64_t, float, EvictStrategy::kCustomized>;
template class HKVVariable<int64_t, float, EvictStrategy::kLru>;
template class HKVVariable<int64_t, float, EvictStrategy::kLfu>;

template class HKVVariable<int64_t, half, EvictStrategy::kCustomized>;
template class HKVVariable<int64_t, half, EvictStrategy::kLru>;
template class HKVVariable<int64_t, half, EvictStrategy::kLfu>;

template class HKVVariable<int64_t, bfloat16_t, EvictStrategy::kCustomized>;
template class HKVVariable<int64_t, bfloat16_t, EvictStrategy::kLru>;
template class HKVVariable<int64_t, bfloat16_t, EvictStrategy::kLfu>;

template class HKVVariable<uint64_t, float, EvictStrategy::kCustomized>;
template class HKVVariable<uint64_t, float, EvictStrategy::kLru>;
template class HKVVariable<uint64_t, float, EvictStrategy::kLfu>;

template class HKVVariable<uint64_t, half, EvictStrategy::kCustomized>;
template class HKVVariable<uint64_t, half, EvictStrategy::kLru>;
template class HKVVariable<uint64_t, half, EvictStrategy::kLfu>;

template class HKVVariable<uint64_t, bfloat16_t, EvictStrategy::kCustomized>;
template class HKVVariable<uint64_t, bfloat16_t, EvictStrategy::kLru>;
template class HKVVariable<uint64_t, bfloat16_t, EvictStrategy::kLfu>;

}  // namespace dyn_emb
