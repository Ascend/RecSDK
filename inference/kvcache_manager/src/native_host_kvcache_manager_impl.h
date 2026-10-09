/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <tiling/platform/platform_ascendc.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kvcache_npu_utils.h"

namespace kvcache {

class KVOnloadHandle {
public:
    KVOnloadHandle(int num_layers);
    ~KVOnloadHandle();

    void init();
    void reset();
    void complete_host(int layer_idx, aclrtStream stream);
    void wait_layer(int layer_idx);

public:
    int num_layers;
    std::vector<ACLEventPtr> compl_event;
    std::vector<ACLEventPtr> internal_onload_event;
    // Written under mtx_ by complete_host() so the condition_variable predicate
    // stays consistent; atomic so init()/reset() may clear it without the lock.
    // Matches KVOffloadHandle::host_ready.
    std::vector<std::atomic<int>> host_complete;
    std::mutex mtx_;
    std::condition_variable cv_;

    // Held by the handle because onload_kvcache() enqueues asynchronous work that
    // keeps reading them after the call returns.
    at::Tensor onload_page_indices;
    ACLEventPtr cat_ready_event;

    bool inited;
    bool no_onload;
};

class KVOffloadHandle {
public:
    KVOffloadHandle(int num_layers);
    ~KVOffloadHandle();

    void init(at::Tensor offload_user_ids, at::Tensor offload_start_indices, std::vector<int>&& offload_lengths);
    void complete_host(int layer_idx, aclrtStream stream);
    void complete_host(int layer_idx, aclrtStream stream,
                       std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>&& chunks);
    bool try_wait_layer(int layer_idx);
    float get_launch_time() const
    {
        return time_stamp;
    }
    void set_launch_time(float t)
    {
        time_stamp = t;
    }

    at::Tensor get_user_ids() const
    {
        return offload_user_ids;
    }
    at::Tensor get_start_indices() const
    {
        return offload_start_indices;
    }
    // Returns an owning copy: init() replaces the backing std::vector wholesale,
    // so the non-owning from_blob view used before could outlive its storage.
    at::Tensor get_lengths() const
    {
        const int64_t n = static_cast<int64_t>(offload_lengths.size());
        at::Tensor out = at::empty({n}, at::dtype(torch::kInt32));
        if (n > 0) {
            std::memcpy(out.data_ptr(), offload_lengths.data(), offload_lengths.size() * sizeof(int));
        }
        return out;
    }

public:
    int num_layers;
    ACLEventPtr inference_event;
    std::vector<ACLEventPtr> ready_event;
    std::vector<ACLEventPtr> internal_gather_event;
    std::vector<std::atomic<int>> host_ready;

    at::Tensor offload_user_ids;
    at::Tensor offload_start_indices;
    std::vector<int> offload_lengths;

    // Held by the handle because offload_kvcache() enqueues asynchronous gather
    // work that keeps reading them after the call returns.
    at::Tensor offload_page_indices;
    at::Tensor h_offload_page_indices;

    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> chunks;

    bool inited;
    bool no_offload;
    float time_stamp;
};

class HostKVStorageImpl {
public:
    HostKVStorageImpl(int num_layers, int num_kv_heads, int kv_headdim, int num_tokens_per_page,
                      int64_t num_tokens_per_chunk, int64_t capacity_per_layer, int64_t max_batch_size,
                      int64_t max_sequence_length, int device_idx);
    ~HostKVStorageImpl();

    void register_npu_cache_table(std::vector<at::Tensor> table);
    at::Tensor lookup(at::Tensor user_ids);
    int64_t get_kvdata_length(int64_t user_id);
    std::pair<std::vector<void*>, std::vector<int64_t>> get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx);
    std::vector<at::Tensor> get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat = true);

private:
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> get_empty_pinned_chunks(
        at::Tensor& offload_user_ids, at::Tensor& offload_start_indices,
        const std::vector<int>& offload_num_pages_list);

public:
    void onload_kvcache(at::Tensor onload_user_ids, const std::vector<at::Tensor> onload_page_indices_list,
                        KVOnloadHandle& onloadhandle);
    bool offload_kvcache(at::Tensor offload_user_ids, at::Tensor offload_start_indices,
                         const std::vector<at::Tensor>& offload_page_indices_list, KVOffloadHandle& offloadhandle);
    std::vector<int> finish_offload(KVOffloadHandle& offloadhandle);
    std::vector<int> cancel_offload(KVOffloadHandle& offloadhandle);

    void evict(int64_t uid);
    void evict_all();
    bool retain(int64_t uid);

public:
    const int num_layers;
    const int num_kv_heads;
    const int kv_headdim;
    const int num_tokens_per_page;
    const int64_t num_tokens_per_chunk;
    const int64_t capacity_per_layer;
    const int max_batch_size;
    const int max_sequence_length;
    size_t max_numel_per_layer_buffer;

    size_t num_pages_per_chunk;
    size_t unit_chunk_numel;
    size_t page_numel;
    size_t per_token_numel;
    size_t unit_chunk_bytes;
    size_t page_bytes;
    size_t inner_token_stride;
    size_t k2v_stride;
    size_t page_stride;

    int device_idx;
    // npu_cache_table holds raw data_ptr()s; npu_cache_table_tensors keeps the
    // registered tensors alive so those pointers stay valid.
    std::vector<void*> npu_cache_table;
    std::vector<at::Tensor> npu_cache_table_tensors;

public:
    std::list<int64_t> _lru_list;
    std::unordered_map<int64_t, typename std::list<int64_t>::iterator> _lru_lookup_table;
    std::queue<std::pair<int64_t, int64_t>> _empty_chunks;
    int64_t _num_empty_chunks;
    std::unordered_map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>>
        _uid_to_chunks;  // 存的是 pair<vector<offsets>, vector<psizes>>，一个 user可以有多段不连续chunk，gather/scatter
                         // 时按段处理
    std::unordered_map<int64_t, int64_t> _uid_to_length;

public:
    ACLStreamPtr onload_stream;
    ACLStreamPtr offload_stream;
    ACLStreamPtr scatter_stream;
    ACLStreamPtr gather_stream;
    uint32_t max_cores = 0;

public:
    std::vector<void*> onload_npu_buffers;
    std::vector<void*> offload_npu_buffers;
    std::vector<ACLHostMemPtr> pinned_kvstorage_buffers;
};

}  // namespace kvcache
