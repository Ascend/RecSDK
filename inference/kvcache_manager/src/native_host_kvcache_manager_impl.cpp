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

#include "native_host_kvcache_manager_impl.h"

#include <cstdio>
#include <iostream>

namespace kvcache {

void ScatterPagedKVCache(uint16_t* continuous_kv, int* page_ids, uint32_t num_heads, uint32_t head_dim,
                         uint32_t page_size, uint32_t stride_page, uint32_t stride_k2v, uint32_t stride_n,
                         uint32_t stride_h, uint32_t num_pages, uint16_t* kv_cache, uint32_t num_ctas,
                         aclrtStream stream);

void GatherPagedKVCache(uint16_t* kv_cache, int* page_ids, uint32_t num_heads, uint32_t head_dim, uint32_t page_size,
                        uint32_t stride_page, uint32_t stride_k2v, uint32_t stride_n, uint32_t stride_h,
                        uint32_t num_pages, uint16_t* continuous_kv, uint32_t num_ctas, aclrtStream stream);

namespace {

// Rounds a page count up to whole unit chunks with integer math, replacing
// std::ceil(float) division, which stops being exact past the 24-bit mantissa.
// Non-positive input yields 0, as std::ceil did over the reachable range.
int64_t round_up_to_chunk_pages(int64_t num_pages, int64_t num_pages_per_chunk)
{
    if (num_pages <= 0) {
        return 0;
    }
    return ((num_pages + num_pages_per_chunk - 1) / num_pages_per_chunk) * num_pages_per_chunk;
}

// A chunk shorter than one unit chunk is the tail of a chunk this uid already
// owns, reused in get_empty_pinned_chunks; it was never taken out of
// _num_empty_chunks, so it must not be pushed back or registered again.
bool is_reused_partial_chunk(int64_t chunk_psize, int64_t num_pages_per_chunk)
{
    return chunk_psize < num_pages_per_chunk;
}

}  // namespace

KVOnloadHandle::KVOnloadHandle(int num_layers)
    : num_layers(num_layers),
      compl_event(num_layers),
      internal_onload_event(num_layers),
      host_complete(num_layers),
      inited(false),
      no_onload(true)
{
    for (int i = 0; i < num_layers; i++) {
        host_complete[i].store(0, std::memory_order_release);
    }
}

KVOnloadHandle::~KVOnloadHandle() = default;

void KVOnloadHandle::init()
{
    this->no_onload = false;
    if (!inited) {
        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
            this->compl_event[layer_idx] = make_acl_event();
            this->internal_onload_event[layer_idx] = make_acl_event();
            host_complete[layer_idx].store(0, std::memory_order_release);
        }
        inited = true;
    }
}

void KVOnloadHandle::reset()
{
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        host_complete[layer_idx].store(0, std::memory_order_release);
    }
}

void KVOnloadHandle::complete_host(int layer_idx, aclrtStream stream)
{
    ACL_CHECK(aclrtRecordEvent(compl_event[layer_idx].get(), stream));
    {
        std::unique_lock<std::mutex> lock(mtx_);
        host_complete[layer_idx].store(1, std::memory_order_release);
    }
    cv_.notify_one();
}

void KVOnloadHandle::wait_layer(int layer_idx)
{
    if (no_onload)
        return;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, layer_idx] { return host_complete[layer_idx].load(std::memory_order_acquire) == 1; });
    }
    aclrtStream current = c10_npu::getCurrentNPUStream().stream();
    ACL_CHECK(aclrtStreamWaitEvent(current, compl_event[layer_idx].get()));
}

KVOffloadHandle::KVOffloadHandle(int num_layers)
    : num_layers(num_layers),
      ready_event(num_layers),
      internal_gather_event(num_layers),
      host_ready(num_layers),
      inited(false),
      no_offload(true),
      time_stamp(0.0f)
{
    for (int i = 0; i < num_layers; i++) {
        host_ready[i].store(0, std::memory_order_release);
    }
}

KVOffloadHandle::~KVOffloadHandle() = default;

void KVOffloadHandle::init(at::Tensor offload_user_ids, at::Tensor offload_start_indices,
                           std::vector<int>&& offload_lengths)
{
    this->no_offload = false;
    if (!inited) {
        inference_event = make_acl_event();
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            internal_gather_event[layer_idx] = make_acl_event();
            ready_event[layer_idx] = make_acl_event();
        }
        inited = true;
    }
    this->offload_user_ids = offload_user_ids;
    this->offload_start_indices = offload_start_indices;
    this->offload_lengths = std::move(offload_lengths);
}

void KVOffloadHandle::complete_host(int layer_idx, aclrtStream stream)
{
    ACL_CHECK(aclrtRecordEvent(ready_event[layer_idx].get(), stream));
    host_ready[layer_idx].store(1, std::memory_order_release);
}

void KVOffloadHandle::complete_host(int layer_idx, aclrtStream stream,
                                    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>&& chunks)
{
    this->chunks = std::move(chunks);
    ACL_CHECK(aclrtRecordEvent(ready_event[layer_idx].get(), stream));
    host_ready[layer_idx].store(1, std::memory_order_release);
}

bool KVOffloadHandle::try_wait_layer(int layer_idx)
{
    if (layer_idx == -1)
        layer_idx = static_cast<int>(host_ready.size()) - 1;
    if (host_ready[layer_idx].load(std::memory_order_acquire) == 0) {
        return false;
    }
    aclrtEvent event = ready_event[layer_idx].get();
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    aclError err = aclrtQueryEventStatus(event, &status);
    if (err != ACL_SUCCESS) {
        throw std::runtime_error("aclrtQueryEventStatus failed");
    }
    return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
}

HostKVStorageImpl::HostKVStorageImpl(int num_layers, int num_kv_heads, int kv_headdim, int num_tokens_per_page,
                                     int64_t num_tokens_per_chunk, int64_t capacity_per_layer, int64_t max_batch_size,
                                     int64_t max_sequence_length, int device_idx)
    : num_layers(num_layers),
      num_kv_heads(num_kv_heads),
      kv_headdim(kv_headdim),
      num_tokens_per_page(num_tokens_per_page),
      num_tokens_per_chunk(num_tokens_per_chunk),
      capacity_per_layer(capacity_per_layer),
      max_batch_size(max_batch_size),
      max_sequence_length(max_sequence_length),
      device_idx(device_idx),
      _uid_to_chunks()
{
    ACL_CHECK(aclrtSetDevice(device_idx));

    per_token_numel = 2 * num_kv_heads * kv_headdim;
    page_numel = static_cast<size_t>(num_tokens_per_page) * per_token_numel;
    unit_chunk_numel = static_cast<size_t>(num_tokens_per_chunk) * per_token_numel;
    num_pages_per_chunk = static_cast<size_t>(num_tokens_per_chunk) / static_cast<size_t>(num_tokens_per_page);
    page_bytes = page_numel * sizeof(at::BFloat16);
    unit_chunk_bytes = unit_chunk_numel * sizeof(at::BFloat16);
    inner_token_stride = num_kv_heads * kv_headdim;         // K 或 V 单 token 元素数
    k2v_stride = inner_token_stride * num_tokens_per_page;  // 整个K块大小，调到V
    page_stride = 2 * k2v_stride;                           // K+V 一整页
    max_numel_per_layer_buffer =
        static_cast<size_t>(max_batch_size) * static_cast<size_t>(max_sequence_length) * per_token_numel;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform)
        throw std::runtime_error("get platform failed");
    max_cores = platform->GetCoreNumAiv();
    onload_stream = make_acl_stream();
    offload_stream = make_acl_stream();
    scatter_stream = make_acl_stream();
    gather_stream = make_acl_stream();

    for (int i = 0; i < 2; ++i) {
        void* ptr = nullptr;
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptr), max_numel_per_layer_buffer * sizeof(uint16_t),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        onload_npu_buffers.push_back(ptr);
    }
    for (int i = 0; i < 2; ++i) {
        void* ptr = nullptr;
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&ptr), max_numel_per_layer_buffer * sizeof(uint16_t),
                              ACL_MEM_MALLOC_HUGE_FIRST));
        offload_npu_buffers.push_back(ptr);
    }
    for (int i = 0; i < num_layers; ++i) {
        pinned_kvstorage_buffers.push_back(make_acl_host_mem(capacity_per_layer));
    }
    _num_empty_chunks = capacity_per_layer / static_cast<int64_t>(unit_chunk_bytes);
    _empty_chunks.push({0, _num_empty_chunks});
}

HostKVStorageImpl::~HostKVStorageImpl()
{
    pinned_kvstorage_buffers.clear();
    for (void* ptr : onload_npu_buffers) {
        ACL_CHECK_NOEXCEPT(aclrtFree(ptr));
    }
    for (void* ptr : offload_npu_buffers) {
        ACL_CHECK_NOEXCEPT(aclrtFree(ptr));
    }
    onload_stream.reset();
    offload_stream.reset();
    scatter_stream.reset();
    gather_stream.reset();
}

void HostKVStorageImpl::register_npu_cache_table(std::vector<at::Tensor> table)
{
    if (table.size() != static_cast<size_t>(num_layers)) {
        throw std::runtime_error("num tables != num_layers");
    }
    if (!npu_cache_table.empty()) {
        throw std::runtime_error("npu cache table already registered");
    }
    npu_cache_table.reserve(table.size());
    for (auto& t : table) {
        if (!t.is_contiguous()) {
            throw std::runtime_error("npu cache table tensor must be contiguous");
        }
        npu_cache_table.push_back(t.data_ptr());
    }
    // Keep the tensors alive: the raw data_ptr()s above outlive this call.
    npu_cache_table_tensors = std::move(table);
}

at::Tensor HostKVStorageImpl::lookup(at::Tensor user_ids)
{
    int64_t n = user_ids.size(0);
    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* out_ptr = out.data_ptr<int32_t>();
    auto* uids_ptr = user_ids.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t uid = uids_ptr[i];
        auto it = _uid_to_length.find(uid);
        out_ptr[i] = (it == _uid_to_length.end()) ? 0 : static_cast<int32_t>(it->second);
    }
    return out;
}

int64_t HostKVStorageImpl::get_kvdata_length(int64_t user_id)
{
    auto it = _uid_to_length.find(user_id);
    return (it == _uid_to_length.end()) ? 0 : it->second;
}

std::pair<std::vector<void*>, std::vector<int64_t>> HostKVStorageImpl::get_kvdata(int64_t user_id, int64_t length,
                                                                                  int64_t layer_idx)
{
    std::vector<void*> ptrs;
    std::vector<int64_t> sizes;
    if (length == 0) {
        return {ptrs, sizes};
    }

    // _uid_to_chunks存的是 pair<vector<offsets>, vector<psizes>>，一个 user 可以有多段不连续chunk，gather/scatter
    // 时按段处理
    // find() rather than operator[]: this is a read-only accessor and must not
    // insert an entry for an unknown uid. The empty result is what operator[]
    // produced for that case, so callers see no change.
    auto chunks_it = _uid_to_chunks.find(user_id);
    if (chunks_it == _uid_to_chunks.end()) {
        return {ptrs, sizes};
    }
    const auto& chunk_offsets = chunks_it->second.first;
    const auto& chunk_psizes = chunks_it->second.second;

    int64_t tokens_from_chunks = 0;
    for (size_t i = 0; i < chunk_offsets.size(); i++) {
        void* base = static_cast<char*>(pinned_kvstorage_buffers[layer_idx].get()) + chunk_offsets[i];
        ptrs.push_back(base);
        if (tokens_from_chunks + chunk_psizes[i] * this->num_tokens_per_page >= length) {
            int64_t last_chunk_tokens = length - tokens_from_chunks;
            sizes.push_back(last_chunk_tokens * this->per_token_numel * sizeof(uint16_t));
            break;
        }
        tokens_from_chunks += chunk_psizes[i] * this->num_tokens_per_page;
        sizes.push_back(chunk_psizes[i] * this->num_tokens_per_page * this->per_token_numel * sizeof(uint16_t));
    }
    return {ptrs, sizes};
}

std::vector<at::Tensor> HostKVStorageImpl::get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat)
{
    const int64_t batch_size = static_cast<int64_t>(user_ids.size());

    std::vector<int64_t> seqlens(batch_size, 0);
    int64_t total_seqlen = 0;
    for (int64_t seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        seqlens[seq_idx] = get_kvdata_length(user_ids[seq_idx]);
        // finish_offload only ever stores whole pages in _uid_to_length, so the
        // sum-then-divide below equals the sum of per-sequence page counts.
        if (seqlens[seq_idx] % this->num_tokens_per_page != 0) {
            throw std::runtime_error("get_kvdata_tensor: uid " + std::to_string(user_ids[seq_idx]) + " length " +
                                     std::to_string(seqlens[seq_idx]) + " is not a multiple of num_tokens_per_page");
        }
        total_seqlen += seqlens[seq_idx];
    }
    int64_t num_total_pages = total_seqlen / this->num_tokens_per_page;

    at::Tensor tensor_res = at::empty(
        {this->num_layers, num_total_pages, 2, this->num_tokens_per_page, this->num_kv_heads, this->kv_headdim},
        torch::kBFloat16);

    int64_t seqlen_offset = 0;
    uint16_t* raw_ptr = reinterpret_cast<uint16_t*>(tensor_res.data_ptr());

    for (int64_t seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        uint16_t* seq_ptr = raw_ptr + (seqlen_offset / this->num_tokens_per_page) * this->page_numel;

        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
            auto [chunk_ptrs, chunk_bytes] = get_kvdata(user_ids[seq_idx], seqlens[seq_idx], layer_idx);
            int64_t running_numel = 0;
            for (size_t chk_idx = 0; chk_idx < chunk_ptrs.size(); chk_idx++) {
                std::memcpy(seq_ptr + layer_idx * tensor_res.stride(0) + running_numel, chunk_ptrs[chk_idx],
                            chunk_bytes[chk_idx]);
                running_numel += chunk_bytes[chk_idx] / sizeof(uint16_t);
            }
        }

        seqlen_offset += seqlens[seq_idx];
    }

    std::vector<at::Tensor> res({tensor_res});
    return res;
}

std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> HostKVStorageImpl::get_empty_pinned_chunks(
    at::Tensor& offload_user_ids, at::Tensor& offload_start_indices, const std::vector<int>& offload_num_pages_list)
{
    const int64_t batch_size = offload_user_ids.size(0);
    const int64_t pages_per_chunk = static_cast<int64_t>(this->num_pages_per_chunk);

    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> empty_pinned_chunks;

    // 针对所有用户，计算复用未满的chunk后还需要多少page
    // ((offload_num_pages_list[idx] + num_unaligned_pages - 1) / num_pages_per_chunk) * num_pages_per_chunk
    // 实际为下述公式简化
    // ((offload_num_pages_list[idx] - (num_pages_per_chunk - num_unaligned_pages) + num_pages_per_chunk - 1) /
    // this->num_pages_per_chunk) * num_pages_per_chunk
    int64_t num_pages_required = 0;
    for (int64_t idx = 0; idx < batch_size; idx++) {
        int64_t start_pos = offload_start_indices[idx].item<int64_t>();
        int64_t num_unaligned_pages = (start_pos % this->num_tokens_per_chunk) / this->num_tokens_per_page;
        num_unaligned_pages = (num_unaligned_pages == 0) ? pages_per_chunk : num_unaligned_pages;
        num_pages_required +=
            ((offload_num_pages_list[idx] + num_unaligned_pages - 1) / pages_per_chunk) * pages_per_chunk;
    }

    // 如果pages不足，从LRU列表驱逐不在这次offload列表的用户
    std::vector<int64_t> uids_to_evict;
    std::unordered_set<int64_t> freezed_uids(offload_user_ids.data_ptr<int64_t>(),
                                             offload_user_ids.data_ptr<int64_t>() + batch_size);

    int64_t num_available_pages = _num_empty_chunks * pages_per_chunk;
    for (auto it = std::rbegin(_lru_list); it != std::rend(_lru_list); it++) {
        if (num_pages_required <= num_available_pages)
            break;

        const int64_t uid_to_evict = *it;
        if (freezed_uids.find(uid_to_evict) != freezed_uids.end())
            continue;

        // find() rather than operator[]: a uid whose offload fitted entirely in a
        // reused partial chunk has an LRU entry but no _uid_to_chunks entry, which
        // operator[] would silently create. It is still evicted, to drop that entry.
        auto evict_it = _uid_to_chunks.find(uid_to_evict);
        if (evict_it != _uid_to_chunks.end()) {
            const auto& chunk_psizes = evict_it->second.second;
            for (size_t idx = 0; idx < chunk_psizes.size(); idx++) {
                num_available_pages += chunk_psizes[idx];
            }
        }
        uids_to_evict.push_back(uid_to_evict);
    }
    if (num_available_pages < num_pages_required)
        return empty_pinned_chunks;

    // Actual eviction
    for (auto uid_to_evict : uids_to_evict)
        evict(uid_to_evict);

    // Actual allocation
    for (int64_t idx = 0; idx < batch_size; idx++) {
        int64_t user_id = offload_user_ids[idx].item<int64_t>();
        int64_t start_pos = offload_start_indices[idx].item<int64_t>();
        int64_t offload_num_pages = offload_num_pages_list[idx];

        std::vector<int64_t> chunk_offsets;
        std::vector<int64_t> chunk_psizes;

        if (start_pos % num_tokens_per_chunk != 0) {
            // 如果start_pos未对齐chunk边界，复用最后一个未满的chunk.
            auto partial_num_pages = (start_pos % this->num_tokens_per_chunk) / this->num_tokens_per_page;
            auto chunks_it = _uid_to_chunks.find(user_id);
            if (chunks_it == _uid_to_chunks.end() || chunks_it->second.first.empty()) {
                std::fprintf(stderr,
                             "[get_empty_pinned_chunks][reuse-fail] seq=%lld uid=%lld "
                             "start_pos=%lld unaligned but uid has no prior chunk to reuse "
                             "(would be UB via operator[]/back)\n",
                             static_cast<long long>(idx), static_cast<long long>(user_id),
                             static_cast<long long>(start_pos));
                return empty_pinned_chunks;
            }
            int64_t chunk_offset = chunks_it->second.first.back();
            // chunk_psize是分配出去的总页数，不是已用页
            chunk_offset += chunks_it->second.second.back() * this->page_bytes;
            chunk_offset -= this->unit_chunk_bytes - partial_num_pages * this->page_bytes;
            chunk_offsets.push_back(chunk_offset);
            chunk_psizes.push_back(pages_per_chunk - partial_num_pages);

            start_pos += (pages_per_chunk - partial_num_pages) * this->num_tokens_per_page;
            offload_num_pages -= (pages_per_chunk - partial_num_pages);
        }

        const int64_t padded_num_pages = round_up_to_chunk_pages(offload_num_pages, pages_per_chunk);
        int64_t left_num_pages = padded_num_pages;
        while (left_num_pages > 0) {
            if (_empty_chunks.empty()) {
                throw std::runtime_error("get_empty_pinned_chunks: free chunk queue exhausted with " +
                                         std::to_string(left_num_pages) + " pages still to place");
            }
            auto [chunk_idx, num_unit_chunks] = _empty_chunks.front();
            if (num_unit_chunks * pages_per_chunk > left_num_pages) {
                _empty_chunks.front().first += left_num_pages / pages_per_chunk;
                _empty_chunks.front().second -= left_num_pages / pages_per_chunk;
                chunk_offsets.push_back(static_cast<int64_t>(chunk_idx * this->unit_chunk_bytes));
                chunk_psizes.push_back(left_num_pages);
                left_num_pages = 0;
                break;
            } else {
                _empty_chunks.pop();
                chunk_offsets.push_back(static_cast<int64_t>(chunk_idx * this->unit_chunk_bytes));
                chunk_psizes.push_back(num_unit_chunks * pages_per_chunk);
                left_num_pages -= num_unit_chunks * pages_per_chunk;
            }
        }
        _num_empty_chunks -= padded_num_pages / pages_per_chunk;

        empty_pinned_chunks.push_back(std::make_pair(chunk_offsets, chunk_psizes));
    }

    return empty_pinned_chunks;
}

void HostKVStorageImpl::evict(int64_t uid)
{
    const int64_t pages_per_chunk = static_cast<int64_t>(num_pages_per_chunk);
    auto it = _uid_to_chunks.find(uid);
    // A uid may hold an LRU/length entry with no chunk of its own, so the
    // bookkeeping after this block runs whether or not chunks were found.
    if (it != _uid_to_chunks.end()) {
        for (size_t i = 0; i < it->second.first.size(); ++i) {
            _empty_chunks.push(
                {it->second.first[i] / static_cast<int64_t>(unit_chunk_bytes), it->second.second[i] / pages_per_chunk});
            _num_empty_chunks += it->second.second[i] / pages_per_chunk;
        }
        _uid_to_chunks.erase(it);
    }
    _uid_to_length.erase(uid);
    auto lru_it = _lru_lookup_table.find(uid);
    if (lru_it != _lru_lookup_table.end()) {
        _lru_list.erase(lru_it->second);
        _lru_lookup_table.erase(lru_it);
    }
}

void HostKVStorageImpl::evict_all()
{
    _num_empty_chunks = capacity_per_layer / unit_chunk_bytes;

    std::queue<std::pair<int64_t, int64_t>> chunks;
    chunks.push(std::make_pair(0, _num_empty_chunks));
    std::swap(_empty_chunks, chunks);

    _uid_to_chunks.clear();
    _uid_to_length.clear();
    _lru_list.clear();
    _lru_lookup_table.clear();
}

bool HostKVStorageImpl::retain(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    bool found = (_lru_lookup_table.end() != tableIt);
    if (found) {
        _lru_list.erase(tableIt->second);
    }
    _lru_list.push_front(uid);
    _lru_lookup_table[uid] = _lru_list.begin();
    return found;
}

void HostKVStorageImpl::onload_kvcache(at::Tensor onload_user_ids,                              // on host
                                       const std::vector<at::Tensor> onload_page_indices_list,  // on npu
                                       KVOnloadHandle& onloadhandle)
{
    const int batch_size = onload_user_ids.size(0);

    at::Device device = onload_page_indices_list[0].device();
    c10_npu::OptionalNPUGuard guard;
    guard.set_device(device);
    c10_npu::NPUStream c10_npu_onload_stream =
        c10_npu::getStreamFromExternal(this->onload_stream.get(), device.index());

    onloadhandle.init();

    // Concatenate page indices on the default PyTorch stream.
    // at::cat on a non-default Ascend NPU stream can produce zero-filled
    // output, so we must concatenate before entering the onload stream guard.
    // Both live on the handle: scatter_stream keeps reading them after we return.
    onloadhandle.onload_page_indices = at::cat(onload_page_indices_list, 0);
    at::Tensor& onload_page_indices = onloadhandle.onload_page_indices;
    onloadhandle.cat_ready_event = make_acl_event();
    ACL_CHECK(aclrtRecordEvent(onloadhandle.cat_ready_event.get(), c10_npu::getCurrentNPUStream().stream()));

    c10_npu::NPUStreamGuard onload_stream_guard(c10_npu_onload_stream);
    ACL_CHECK(aclrtStreamWaitEvent(this->onload_stream.get(), onloadhandle.cat_ready_event.get()));

    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        char* npu_onload_buffer = reinterpret_cast<char*>(this->onload_npu_buffers[layer_idx % 2]);
        if (layer_idx >= 2) {
            ACL_CHECK(aclrtStreamWaitEvent(this->onload_stream.get(), onloadhandle.compl_event[layer_idx - 2].get()));
        }

        size_t bytes_offset_in_buffer = 0;
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto [chunk_ptrs, chunk_bytes] =
                this->get_kvdata(onload_user_ids[seq_idx].item<int64_t>(),
                                 onload_page_indices_list[seq_idx].size(0) * this->num_tokens_per_page, layer_idx);
            for (size_t chunk_idx = 0; chunk_idx < chunk_ptrs.size(); chunk_idx++) {
                size_t this_bytes = static_cast<size_t>(chunk_bytes[chunk_idx]);
                ACL_CHECK(aclrtMemcpyAsync(npu_onload_buffer + bytes_offset_in_buffer, this_bytes,
                                           chunk_ptrs[chunk_idx], this_bytes, ACL_MEMCPY_HOST_TO_DEVICE,
                                           this->onload_stream.get()));
                bytes_offset_in_buffer += this_bytes;
            }
        }
        ACL_CHECK(aclrtRecordEvent(onloadhandle.internal_onload_event[layer_idx].get(), this->onload_stream.get()));
        // Scatter: write the contiguous onload buffer back into the paged npu_cache_table
        // using page_ids via AscendC __aicore__ kernel on scatter_stream.
        // Mirrors GPU native_host_kvcache_manager_impl.cpp:412-442.
        ACL_CHECK(
            aclrtStreamWaitEvent(this->scatter_stream.get(), onloadhandle.internal_onload_event[layer_idx].get()));
        {
            ScatterPagedKVCache(reinterpret_cast<uint16_t*>(npu_onload_buffer), onload_page_indices.data_ptr<int>(),
                                static_cast<uint32_t>(this->num_kv_heads), static_cast<uint32_t>(this->kv_headdim),
                                static_cast<uint32_t>(this->num_tokens_per_page),
                                static_cast<uint32_t>(this->page_stride), static_cast<uint32_t>(this->k2v_stride),
                                static_cast<uint32_t>(this->inner_token_stride),
                                static_cast<uint32_t>(this->kv_headdim), onload_page_indices.size(0),
                                reinterpret_cast<uint16_t*>(this->npu_cache_table[layer_idx]), this->max_cores,
                                this->scatter_stream.get());
        }
        ACL_CHECK(aclrtRecordEvent(onloadhandle.compl_event[layer_idx].get(), this->scatter_stream.get()));
        onloadhandle.complete_host(layer_idx, this->scatter_stream.get());
    }
    // For next onload, onload stream wait for current onload completion so npu onload buffers are not in use.
    ACL_CHECK(aclrtStreamWaitEvent(this->onload_stream.get(), onloadhandle.compl_event[this->num_layers - 1].get()));
}
bool HostKVStorageImpl::offload_kvcache(at::Tensor offload_user_ids,                               // on host
                                        at::Tensor offload_start_indices,                          // on host
                                        const std::vector<at::Tensor>& offload_page_indices_list,  // on host
                                        KVOffloadHandle& offloadhandle)
{
    int64_t batch_size = offload_user_ids.size(0);

    at::Device device("npu:" + std::to_string(this->device_idx));
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(device);

    // 元数据计算
    std::vector<int> num_pages_list(batch_size);
    std::vector<int> num_pages_offsets(batch_size);
    std::vector<int> offload_lengths(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        num_pages_list[i] = static_cast<int>(offload_page_indices_list[i].size(0));
        num_pages_offsets[i] = (i == 0) ? 0 : (num_pages_offsets[i - 1] + num_pages_list[i - 1]);
        offload_lengths[i] = num_pages_list[i] * this->num_tokens_per_page;
    }
    offloadhandle.init(offload_user_ids, offload_start_indices, std::move(offload_lengths));

    // 合并 page_indices 到一个 pinned host 张量，并在 device 上预分配目标 tensor
    int64_t cat_dim = 0;
    for (const auto& t : offload_page_indices_list)
        cat_dim += t.size(0);
    // Both live on the handle: gather_stream keeps reading them after we return.
    offloadhandle.h_offload_page_indices =
        at::empty({cat_dim}, at::TensorOptions().device(torch::kCPU).dtype(torch::kInt32).pinned_memory(true));
    at::Tensor& h_offload_page_indices = offloadhandle.h_offload_page_indices;
    at::cat_out(h_offload_page_indices, offload_page_indices_list, 0);
    offloadhandle.offload_page_indices = at::empty({cat_dim}, at::TensorOptions().device(device).dtype(torch::kInt32));
    at::Tensor& offload_page_indices = offloadhandle.offload_page_indices;

    // gather_stream waits for inference to finish, then does H2D + GatherPagedKVCache
    // both on gather_stream — same-stream ordering removes the need for an extra event.
    ACL_CHECK(aclrtRecordEvent(offloadhandle.inference_event.get(), c10_npu::getCurrentNPUStream().stream()));
    ACL_CHECK(aclrtStreamWaitEvent(gather_stream.get(), offloadhandle.inference_event.get()));
    ACL_CHECK(aclrtMemcpyAsync(offload_page_indices.data_ptr(), static_cast<size_t>(cat_dim) * sizeof(int32_t),
                               h_offload_page_indices.data_ptr(), static_cast<size_t>(cat_dim) * sizeof(int32_t),
                               ACL_MEMCPY_HOST_TO_DEVICE, gather_stream.get()));

    auto empty_chunks = get_empty_pinned_chunks(offload_user_ids, offload_start_indices, num_pages_list);
    if (empty_chunks.empty())
        return false;

    constexpr int64_t num_pages_per_layer = 128;
    for (int layer = 0; layer < num_layers; ++layer) {
        char* npu_offload_buffer = static_cast<char*>(this->offload_npu_buffers[layer % 2]);
        if (layer >= 2) {
            ACL_CHECK(aclrtStreamWaitEvent(gather_stream.get(), offloadhandle.ready_event[layer - 2].get()));
        }

        // Gather all pages for this layer using custom AscendC gather kernel on gather_stream.
        {
            GatherPagedKVCache(
                reinterpret_cast<uint16_t*>(this->npu_cache_table[layer]), offload_page_indices.data_ptr<int>(),
                static_cast<uint32_t>(this->num_kv_heads), static_cast<uint32_t>(this->kv_headdim),
                static_cast<uint32_t>(this->num_tokens_per_page), static_cast<uint32_t>(this->page_stride),
                static_cast<uint32_t>(this->k2v_stride), static_cast<uint32_t>(this->inner_token_stride),
                static_cast<uint32_t>(this->kv_headdim), offload_page_indices.size(0),
                reinterpret_cast<uint16_t*>(npu_offload_buffer), this->max_cores, gather_stream.get());
        }

        ACL_CHECK(aclrtRecordEvent(offloadhandle.internal_gather_event[layer].get(), gather_stream.get()));
        ACL_CHECK(aclrtStreamWaitEvent(offload_stream.get(), offloadhandle.internal_gather_event[layer].get()));

        for (int64_t i = 0; i < batch_size; ++i) {
            size_t offset = static_cast<size_t>(num_pages_offsets[i]) * this->page_bytes;
            const auto& chunk_offsets = empty_chunks[i].first;
            const auto& chunk_psizes = empty_chunks[i].second;
            for (size_t c = 0; c < chunk_offsets.size(); ++c) {
                size_t this_bytes = static_cast<size_t>(chunk_psizes[c]) * this->page_bytes;
                void* dst_ptr = static_cast<char*>(this->pinned_kvstorage_buffers[layer].get()) + chunk_offsets[c];
                ACL_CHECK(aclrtMemcpyAsync(dst_ptr, this_bytes, npu_offload_buffer + offset, this_bytes,
                                           ACL_MEMCPY_DEVICE_TO_HOST, offload_stream.get()));
                offset += this_bytes;
            }
        }

        if (layer < num_layers - 1) {
            offloadhandle.complete_host(layer, offload_stream.get());
        } else {
            offloadhandle.complete_host(layer, offload_stream.get(), std::move(empty_chunks));
        }
    }
    ACL_CHECK(aclrtStreamWaitEvent(gather_stream.get(), offloadhandle.ready_event[this->num_layers - 1].get()));
    return true;
}

std::vector<int> HostKVStorageImpl::finish_offload(KVOffloadHandle& offloadhandle)
{
    auto batch_size = offloadhandle.offload_user_ids.size(0);
    std::vector<int> offload_sucess(batch_size, 1);
    if (offloadhandle.chunks.size() < static_cast<size_t>(batch_size)) {
        throw std::runtime_error("finish_offload: handle.chunks size=" + std::to_string(offloadhandle.chunks.size()) +
                                 " < batch_size=" + std::to_string(batch_size) +
                                 " (offload_kvcache likely returned false; " +
                                 "start_index out of bounds or no free chunks)");
    }
    const int64_t pages_per_chunk = static_cast<int64_t>(this->num_pages_per_chunk);
    for (int64_t seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t user_id = offloadhandle.offload_user_ids[seq_idx].item<int64_t>();
        auto [chunk_offsets, chunk_psizes] = offloadhandle.chunks[seq_idx];

        auto offload_start_index = offloadhandle.offload_start_indices[seq_idx].item<int64_t>();
        if (offload_start_index != _uid_to_length[user_id]) {
            // give a warning print
            offload_sucess[seq_idx] = 0;
            for (size_t idx = 0; idx < chunk_offsets.size(); idx++) {
                if (is_reused_partial_chunk(chunk_psizes[idx], pages_per_chunk))
                    continue;
                _empty_chunks.push(std::make_pair(chunk_offsets[idx] / static_cast<int64_t>(this->unit_chunk_bytes),
                                                  chunk_psizes[idx] / pages_per_chunk));
                _num_empty_chunks += chunk_psizes[idx] / pages_per_chunk;
            }
            continue;
        }

        for (size_t idx = 0; idx < chunk_offsets.size(); idx++) {
            if (is_reused_partial_chunk(chunk_psizes[idx], pages_per_chunk))
                continue;
            _uid_to_chunks[user_id].first.push_back(chunk_offsets[idx]);
            _uid_to_chunks[user_id].second.push_back(chunk_psizes[idx]);
        }
        _uid_to_length[user_id] = offload_start_index + offloadhandle.offload_lengths[seq_idx];
        retain(user_id);
    }

    return offload_sucess;
}

std::vector<int> HostKVStorageImpl::cancel_offload(KVOffloadHandle& offloadhandle)
{
    // nop for canceling the launched kernels

    auto batch_size = offloadhandle.offload_user_ids.size(0);
    std::vector<int> offload_sucess(batch_size, 0);

    if (offloadhandle.chunks.size() < static_cast<size_t>(batch_size)) {
        std::fprintf(stderr,
                     "[cancel_offload][skip] chunks.size=%zu < batch_size=%lld "
                     "(offload_kvcache returned false; nothing to reclaim)\n",
                     offloadhandle.chunks.size(), static_cast<long long>(batch_size));
        return offload_sucess;
    }

    const int64_t pages_per_chunk = static_cast<int64_t>(this->num_pages_per_chunk);
    for (int64_t seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        auto [chunk_offsets, chunk_psizes] = offloadhandle.chunks[seq_idx];
        if (chunk_offsets.empty())
            continue;
        for (size_t idx = 0; idx < chunk_offsets.size(); idx++) {
            if (is_reused_partial_chunk(chunk_psizes[idx], pages_per_chunk))
                continue;
            _empty_chunks.push(std::make_pair(chunk_offsets[idx] / static_cast<int64_t>(this->unit_chunk_bytes),
                                              chunk_psizes[idx] / pages_per_chunk));
            _num_empty_chunks += chunk_psizes[idx] / pages_per_chunk;
        }
    }

    return offload_sucess;
}

}  // namespace kvcache
