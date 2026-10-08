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

#include "npu_kvcache_manager_impl.h"

#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>

namespace kvcache {

namespace {

void check_cpu_int64_1d(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype int64");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cpu_integral_1d(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong, name,
                " must have dtype int32 or int64");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_npu_int32_1d(const at::Tensor& tensor, const at::Device& expected_device, const char* name)
{
    TORCH_CHECK(tensor.device() == expected_device, name, " must be on ", expected_device);
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void GetPagedBatchIndicesPositions(int32_t batch_size, int32_t* append_indptr, int32_t* seq_lens_ptr,
                                   int32_t* batch_indices_ptr, int32_t* positions_ptr, aclrtStream stream);

NPUKVCacheManagerImpl::NPUKVCacheManagerImpl(int num_layers, int num_kv_heads, int kv_headdim, int num_tokens_per_page,
                                             int num_tokens_per_chunk, int num_primary_cache_pages,
                                             int num_buffer_pages, int max_batch_size, int max_sequence_length,
                                             int device_idx)
    : num_layers(num_layers),
      num_kv_heads(num_kv_heads),
      kv_headdim(kv_headdim),
      num_tokens_per_page(num_tokens_per_page),
      num_tokens_per_chunk(num_tokens_per_chunk),
      num_primary_cache_pages(num_primary_cache_pages),
      num_buffer_pages(num_buffer_pages),
      total_offloaded_pages(0),
      max_batch_size(max_batch_size),
      max_sequence_length(max_sequence_length),
      device_idx(device_idx)
{
    TORCH_CHECK(num_tokens_per_page > 0, "num_tokens_per_page must be positive");
    TORCH_CHECK(num_tokens_per_chunk > 0, "num_tokens_per_chunk must be positive");
    TORCH_CHECK(num_primary_cache_pages >= 0, "num_primary_cache_pages must be non-negative");
    TORCH_CHECK(num_buffer_pages >= 0, "num_buffer_pages must be non-negative");
    TORCH_CHECK(max_batch_size > 0, "max_batch_size must be positive");
    TORCH_CHECK(max_sequence_length >= 0, "max_sequence_length must be non-negative");

    size_t padded_pages_per_seq = (max_sequence_length + num_tokens_per_page - 1) / num_tokens_per_page;
    this->max_offload_pages = static_cast<size_t>(max_batch_size) * padded_pages_per_seq;

    at::Device device("npu:" + std::to_string(this->device_idx));
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(device);

    for (int page_id = 0; page_id < num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);

    this->alloc_stream = make_acl_stream();
    this->metadata_host_buffer = make_acl_host_mem((5 * max_batch_size + 4) * sizeof(int));
}

NPUKVCacheManagerImpl::~NPUKVCacheManagerImpl() = default;

std::vector<at::Tensor> NPUKVCacheManagerImpl::lookup(at::Tensor uids)
{
    check_cpu_int64_1d(uids, "uids");
    TORCH_CHECK(uids.numel() <= this->max_batch_size, "uids exceeds max_batch_size");

    int batch_size = static_cast<int>(uids.numel());
    const int64_t* user_ids_ptr = uids.data_ptr<int64_t>();
    at::Tensor cached_startpos = at::empty({batch_size}, at::dtype(torch::kInt32).device(at::kCPU));
    at::Tensor cached_lengths = at::empty({batch_size}, at::dtype(torch::kInt32).device(at::kCPU));
    int* cached_startpos_ptr = cached_startpos.data_ptr<int>();
    int* cached_lengths_ptr = cached_lengths.data_ptr<int>();
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids_ptr[seq_idx];
        const auto start_it = _uid_to_paged_cache_startpos.find(uid);
        if (start_it != _uid_to_paged_cache_startpos.end()) {
            const auto length_it = _uid_to_paged_cache_length.find(uid);
            TORCH_CHECK(length_it != _uid_to_paged_cache_length.end(), "inconsistent cache metadata for uid ", uid);
            cached_startpos_ptr[seq_idx] = start_it->second;
            cached_lengths_ptr[seq_idx] = length_it->second;
        } else {
            cached_startpos_ptr[seq_idx] = 0;
            cached_lengths_ptr[seq_idx] = 0;
        }
    }
    return {cached_startpos, cached_lengths};
};

int64_t NPUKVCacheManagerImpl::getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids)
{
    for (auto it = std::begin(_lru_list); it != std::end(_lru_list); ++it) {
        if (_uid_offload_lock.find((int64_t)*it) != _uid_offload_lock.end())
            continue;
        if (extra_freezed_uids.find((int64_t)*it) != extra_freezed_uids.end())
            continue;
        return *it;
    }
    throw std::runtime_error("No evictable UID: all cached UIDs are frozen or offload-locked");
};

void NPUKVCacheManagerImpl::evict(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    if (_lru_lookup_table.end() != tableIt) {
        _lru_list.erase(tableIt->second);
        _lru_lookup_table.erase(tableIt);

        for (auto page_id : _uid_to_page_id[uid]) {
            _empty_pages.push(page_id);
        }

        total_offloaded_pages -= std::max(
            0, (_uid_to_offloaded_length[uid] - _uid_to_paged_cache_startpos[uid]) / this->num_tokens_per_page);

        _uid_to_page_id.erase(uid);
        _uid_to_paged_cache_startpos.erase(uid);
        _uid_to_paged_cache_length.erase(uid);
        _uid_to_offloaded_length.erase(uid);
    }
};

void NPUKVCacheManagerImpl::evict_offloaded(int64_t uid)
{
    int num_offloaded_pages =
        std::max(0, (_uid_to_offloaded_length[uid] - _uid_to_paged_cache_startpos[uid]) / this->num_tokens_per_page);
    if (num_offloaded_pages == 0)
        return;

    int num_pages = _uid_to_page_id[uid].size();
    for (int i = 0; i < num_offloaded_pages; i++) {
        _empty_pages.push(_uid_to_page_id[uid][i]);
    }
    _uid_to_page_id[uid].erase(_uid_to_page_id[uid].begin(), _uid_to_page_id[uid].begin() + num_offloaded_pages);
    _uid_to_paged_cache_startpos[uid] += num_offloaded_pages * this->num_tokens_per_page;
    _uid_to_paged_cache_length[uid] -= num_offloaded_pages * this->num_tokens_per_page;
    total_offloaded_pages -= num_offloaded_pages;
};

void NPUKVCacheManagerImpl::evict_all()
{
    std::queue<int64_t> empty_pages;
    std::swap(_empty_pages, empty_pages);
    _lru_list.clear();
    _lru_lookup_table.clear();
    _uid_to_page_id.clear();
    _uid_to_paged_cache_startpos.clear();
    _uid_to_paged_cache_length.clear();
    _uid_to_offloaded_length.clear();
    _uid_offload_lock.clear();
    total_offloaded_pages = 0;

    for (int page_id = 0; page_id < this->num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);
};

bool NPUKVCacheManagerImpl::retain(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    bool found = (_lru_lookup_table.end() != tableIt);
    if (found) {
        _lru_list.erase(tableIt->second);
    }
    _lru_list.push_back(uid);
    _lru_lookup_table[uid] = std::prev(_lru_list.end());
    return found;
};

std::vector<int> NPUKVCacheManagerImpl::alloc_single_sequence(int64_t uid, int new_total_length,
                                                              int host_cached_startpos, int host_cached_length,
                                                              std::unordered_set<int64_t> freezed_uids)
{
    TORCH_CHECK(new_total_length >= 0 && new_total_length <= this->max_sequence_length,
                "new_total_length must be in [0, max_sequence_length]");
    TORCH_CHECK(host_cached_startpos >= 0, "host_cached_startpos must be non-negative");
    TORCH_CHECK(host_cached_length >= 0 && host_cached_length <= new_total_length,
                "host_cached_length must be in [0, new_total_length]");

    const int64_t num_total_pages =
        (static_cast<int64_t>(new_total_length) + this->num_tokens_per_page - 1) / this->num_tokens_per_page;

    int cur_cached_start = 0;
    int cur_cached_len = 0;
    bool found_in_npu_cache = retain(uid);
    if (found_in_npu_cache) {
        const auto start_it = _uid_to_paged_cache_startpos.find(uid);
        const auto length_it = _uid_to_paged_cache_length.find(uid);
        const auto pages_it = _uid_to_page_id.find(uid);
        TORCH_CHECK(start_it != _uid_to_paged_cache_startpos.end() && length_it != _uid_to_paged_cache_length.end() &&
                        pages_it != _uid_to_page_id.end(),
                    "inconsistent cache state for uid ", uid);
        cur_cached_start = start_it->second;
        cur_cached_len = length_it->second;
    }

    int64_t num_onload_pages = 0;
    if (cur_cached_len == 0) {
        num_onload_pages =
            (static_cast<int64_t>(host_cached_length) + this->num_tokens_per_page - 1) / this->num_tokens_per_page;
    } else if (cur_cached_start > 0) {
        num_onload_pages = cur_cached_start / this->num_tokens_per_page;
    }

    const int64_t num_cur_pages =
        (static_cast<int64_t>(cur_cached_len) + this->num_tokens_per_page - 1) / this->num_tokens_per_page;
    TORCH_CHECK(num_onload_pages + num_cur_pages <= num_total_pages,
                "cached sequence is longer than requested total length for uid ", uid);
    if (num_cur_pages > 0) {
        TORCH_CHECK(static_cast<int64_t>(_uid_to_page_id.at(uid).size()) >= num_cur_pages,
                    "insufficient cached page IDs for uid ", uid);
    }

    const int64_t num_append_pages = num_total_pages - num_onload_pages - num_cur_pages;
    const int64_t num_required_pages = num_onload_pages + num_append_pages;

    {
        for (auto it = std::begin(_lru_list); it != std::end(_lru_list);) {
            if (num_required_pages <= static_cast<int64_t>(_empty_pages.size()))
                break;
            if (this->_uid_offload_lock.find((int64_t)*it) != this->_uid_offload_lock.end()) {
                ++it;
                continue;
            }
            if (freezed_uids.find((int64_t)*it) != freezed_uids.end()) {
                ++it;
                continue;
            }
            evict_offloaded(*it);
            if (_uid_to_paged_cache_length.at(*it) == 0) {
                auto uid_to_evict = *it;
                auto next_it = _lru_list.erase(it);
                _lru_lookup_table.erase(uid_to_evict);

                _uid_to_page_id.erase(uid_to_evict);
                _uid_to_paged_cache_startpos.erase(uid_to_evict);
                _uid_to_paged_cache_length.erase(uid_to_evict);
                _uid_to_offloaded_length.erase(uid_to_evict);
                it = next_it;
            } else {
                ++it;
            }
        }
    }
    while (num_required_pages > static_cast<int64_t>(_empty_pages.size())) {
        int64_t uid_to_evict = getUIdToEvict(freezed_uids);
        evict(uid_to_evict);
    }

    std::vector<int> page_ids(static_cast<size_t>(num_total_pages));
    for (int64_t i = 0; i < num_onload_pages; i++) {
        page_ids[static_cast<size_t>(i)] = _empty_pages.front();
        _empty_pages.pop();
    }
    for (int64_t i = num_onload_pages; i < num_onload_pages + num_cur_pages; i++) {
        page_ids[static_cast<size_t>(i)] = _uid_to_page_id.at(uid)[static_cast<size_t>(i - num_onload_pages)];
    }
    for (int64_t i = num_onload_pages + num_cur_pages; i < num_total_pages; i++) {
        page_ids[static_cast<size_t>(i)] = _empty_pages.front();
        _empty_pages.pop();
    }
    _uid_to_page_id[uid] = page_ids;
    _uid_to_paged_cache_startpos[uid] = 0;
    _uid_to_paged_cache_length[uid] = new_total_length;

    return page_ids;
};

void NPUKVCacheManagerImpl::allocate(at::Tensor user_ids, at::Tensor total_hist_lens, at::Tensor host_cached_lengths,
                                     at::Tensor page_ids_npu_buffer, at::Tensor metadata_npu_buffer)
{
    at::Device device("npu:" + std::to_string(this->device_idx));
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(device);

    check_cpu_int64_1d(user_ids, "user_ids");
    check_cpu_integral_1d(total_hist_lens, "total_hist_lens");
    check_cpu_integral_1d(host_cached_lengths, "host_cached_lengths");
    TORCH_CHECK(host_cached_lengths.scalar_type() == at::kInt, "host_cached_lengths must have dtype int32");
    TORCH_CHECK(total_hist_lens.numel() == user_ids.numel() && host_cached_lengths.numel() == user_ids.numel(),
                "total_hist_lens and host_cached_lengths must match user_ids length");
    TORCH_CHECK(user_ids.numel() <= this->max_batch_size, "batch size exceeds max_batch_size");
    check_npu_int32_1d(page_ids_npu_buffer, device, "page_ids_npu_buffer");
    check_npu_int32_1d(metadata_npu_buffer, device, "metadata_npu_buffer");

    const int batch_size = static_cast<int>(user_ids.numel());
    const int64_t* user_ids_ptr = user_ids.data_ptr<int64_t>();
    const int* host_cached_lengths_ptr = host_cached_lengths.data_ptr<int>();

    std::vector<int> checked_total_history_lengths;
    checked_total_history_lengths.reserve(batch_size);
    std::vector<int> cached_lengths;
    cached_lengths.reserve(batch_size);
    int64_t required_page_ids = 0;
    int64_t total_history_offset = 0;
    int64_t new_tokens_64 = 0;

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        const int64_t uid = user_ids_ptr[seq_idx];
        const int64_t total_history_length_64 = total_hist_lens[seq_idx].item<int64_t>();
        TORCH_CHECK(total_history_length_64 >= 0 && total_history_length_64 <= this->max_sequence_length,
                    "total history length must be in [0, max_sequence_length] for uid ", uid);
        const int total_history_length = static_cast<int>(total_history_length_64);
        const int host_cached_length = host_cached_lengths_ptr[seq_idx];
        TORCH_CHECK(host_cached_length >= 0 && host_cached_length <= total_history_length,
                    "host cached length must be in [0, total history length] for uid ", uid);

        int npu_cached_end = 0;
        const auto start_it = _uid_to_paged_cache_startpos.find(uid);
        const auto length_it = _uid_to_paged_cache_length.find(uid);
        TORCH_CHECK((start_it == _uid_to_paged_cache_startpos.end()) == (length_it == _uid_to_paged_cache_length.end()),
                    "inconsistent cache metadata for uid ", uid);
        if (start_it != _uid_to_paged_cache_startpos.end()) {
            TORCH_CHECK(start_it->second >= 0 && length_it->second >= 0 &&
                            static_cast<int64_t>(start_it->second) + length_it->second <= total_history_length,
                        "NPU cached range exceeds total history length for uid ", uid);
            npu_cached_end = start_it->second + length_it->second;
        }

        const int cached_length = std::max(host_cached_length, npu_cached_end);
        checked_total_history_lengths.push_back(total_history_length);
        cached_lengths.push_back(cached_length);
        required_page_ids += (total_history_length_64 + this->num_tokens_per_page - 1) / this->num_tokens_per_page;
        total_history_offset += total_history_length_64;
        new_tokens_64 += total_history_length_64 - cached_length;
    }

    TORCH_CHECK(required_page_ids <= std::numeric_limits<int>::max(), "page index count exceeds int32 range");
    TORCH_CHECK(total_history_offset <= std::numeric_limits<int>::max(), "total history offset exceeds int32 range");
    TORCH_CHECK(new_tokens_64 <= std::numeric_limits<int>::max(), "new token count exceeds int32 range");
    TORCH_CHECK(page_ids_npu_buffer.numel() >= required_page_ids,
                "page_ids_npu_buffer is too small: expected at least ", required_page_ids, " elements, got ",
                page_ids_npu_buffer.numel());
    const int64_t required_metadata_items = static_cast<int64_t>(batch_size) * 5 + 4 + 2 * new_tokens_64;
    TORCH_CHECK(metadata_npu_buffer.numel() >= required_metadata_items,
                "metadata_npu_buffer is too small: expected at least ", required_metadata_items, " elements, got ",
                metadata_npu_buffer.numel());

    std::vector<int> page_indices;
    page_indices.reserve(static_cast<size_t>(required_page_ids));

    int* host_bufptr = static_cast<int*>(this->metadata_host_buffer.get());

    int* page_indptr = host_bufptr + 0;
    int* last_page_len = host_bufptr + batch_size + 1;
    int* total_history_lengths = host_bufptr + batch_size * 2 + 1;
    int* total_history_offsets = host_bufptr + batch_size * 3 + 1;
    int* new_history_nnz_npu = host_bufptr + batch_size * 4 + 2;
    int* new_history_offsets = host_bufptr + batch_size * 4 + 3;

    page_indptr[0] = 0;
    total_history_offsets[0] = 0;
    new_history_offsets[0] = 0;

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        const int64_t uid = user_ids_ptr[seq_idx];
        this->_uid_to_offloaded_length[uid] = host_cached_lengths_ptr[seq_idx];
    }

    const std::unordered_set<int64_t> freezed_uids(user_ids_ptr, user_ids_ptr + batch_size);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        const int64_t uid = user_ids_ptr[seq_idx];
        const int total_history_length = checked_total_history_lengths[seq_idx];

        std::vector<int> page_ids =
            alloc_single_sequence(uid, total_history_length, 0, host_cached_lengths_ptr[seq_idx], freezed_uids);
        page_indices.insert(page_indices.end(), page_ids.begin(), page_ids.end());
        page_indptr[seq_idx + 1] = page_indptr[seq_idx] + static_cast<int>(page_ids.size());
        last_page_len[seq_idx] = this->_uid_to_paged_cache_length.at(uid) % this->num_tokens_per_page;
        if (last_page_len[seq_idx] == 0)
            last_page_len[seq_idx] = this->num_tokens_per_page;

        total_history_lengths[seq_idx] = total_history_length;
        total_history_offsets[seq_idx + 1] = total_history_offsets[seq_idx] + total_history_length;
        new_history_offsets[seq_idx + 1] =
            new_history_offsets[seq_idx] + total_history_length - cached_lengths[seq_idx];
    }

    if (!page_indices.empty()) {
        const size_t page_ids_size = page_indices.size() * sizeof(int);
        ACL_CHECK(aclrtMemcpyAsync(page_ids_npu_buffer.data_ptr(), page_ids_size, page_indices.data(), page_ids_size,
                                   ACL_MEMCPY_HOST_TO_DEVICE, this->alloc_stream.get()));
    }

    const int new_tokens = new_history_offsets[batch_size];
    *new_history_nnz_npu = new_tokens;

    size_t host_buffer_size = (batch_size * 5 + 4) * sizeof(int);
    ACL_CHECK(aclrtMemcpyAsync(metadata_npu_buffer.data_ptr(), host_buffer_size, this->metadata_host_buffer.get(),
                               host_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE, this->alloc_stream.get()));

    int* npu_bufptr = metadata_npu_buffer.data_ptr<int>();
    int* total_history_lengths_dev = npu_bufptr + batch_size * 2 + 1;
    int* new_history_offsets_dev = npu_bufptr + batch_size * 4 + 3;
    int* batch_indices_dev = npu_bufptr + batch_size * 5 + 4;
    int* position_dev = npu_bufptr + batch_size * 5 + 4 + new_tokens;

    GetPagedBatchIndicesPositions(batch_size, new_history_offsets_dev, total_history_lengths_dev, batch_indices_dev,
                                  position_dev, this->alloc_stream.get());

    ACL_CHECK(aclrtSynchronizeStream(this->alloc_stream.get()));
}

at::Tensor NPUKVCacheManagerImpl::check_for_offload(at::Tensor& user_ids)
{
    check_cpu_int64_1d(user_ids, "user_ids");
    TORCH_CHECK(user_ids.numel() <= this->max_batch_size, "user_ids exceeds max_batch_size");
    const int batch_size = static_cast<int>(user_ids.numel());
    const int64_t* user_ids_ptr = user_ids.data_ptr<int64_t>();

    std::vector<int64_t> offload_user_ids;
    std::unordered_set<int64_t> offload_uids_set;

    int64_t num_pages_to_offload = 0;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        const int64_t uid = user_ids_ptr[seq_idx];
        const auto page_it = _uid_to_page_id.find(uid);
        if (page_it == _uid_to_page_id.end()) {
            continue;
        }
        const auto offloaded_it = _uid_to_offloaded_length.find(uid);
        const auto start_it = _uid_to_paged_cache_startpos.find(uid);
        const auto length_it = _uid_to_paged_cache_length.find(uid);
        TORCH_CHECK(offloaded_it != _uid_to_offloaded_length.end() && start_it != _uid_to_paged_cache_startpos.end() &&
                        length_it != _uid_to_paged_cache_length.end(),
                    "inconsistent offload metadata for uid ", uid);
        const int64_t offloaded_length = offloaded_it->second;

        const int64_t cached_end_index = static_cast<int64_t>(start_it->second) + length_it->second;
        if (cached_end_index - offloaded_length >= this->num_tokens_per_chunk) {
            offload_user_ids.push_back(uid);
            offload_uids_set.insert(uid);
        }

        num_pages_to_offload += (cached_end_index - offloaded_length) / this->num_tokens_per_page;
    }

    for (auto it = std::begin(_lru_list); it != std::end(_lru_list); ++it) {
        int64_t uid = *it;
        if (static_cast<int64_t>(this->total_offloaded_pages) + num_pages_to_offload +
                static_cast<int64_t>(this->_empty_pages.size()) >
            this->num_buffer_pages) {
            break;
        }
        if (offload_uids_set.find(uid) != offload_uids_set.end())
            continue;

        const int offloaded_length = this->_uid_to_offloaded_length.at(uid);
        const int cached_startpos = this->_uid_to_paged_cache_startpos.at(uid);
        if (offloaded_length < cached_startpos)
            continue;

        const int cached_end_index = cached_startpos + this->_uid_to_paged_cache_length.at(uid);
        if (cached_end_index - offloaded_length >= this->num_tokens_per_chunk) {
            num_pages_to_offload += (cached_end_index - offloaded_length) / this->num_tokens_per_page;
            if ((size_t)num_pages_to_offload > this->max_offload_pages) {
                break;
            }
            offload_user_ids.push_back(uid);
            offload_uids_set.insert(uid);
        }
    }
    return at::from_blob(offload_user_ids.data(), {offload_user_ids.size()}, at::dtype(torch::kInt64)).clone();
}

void NPUKVCacheManagerImpl::revoke_onboard_pages(at::Tensor& user_ids, at::Tensor& onboard_start_indices,
                                                 at::Tensor& onboard_lengths)
{
    check_cpu_int64_1d(user_ids, "user_ids");
    check_cpu_integral_1d(onboard_start_indices, "onboard_start_indices");
    check_cpu_integral_1d(onboard_lengths, "onboard_lengths");
    TORCH_CHECK(onboard_start_indices.numel() == user_ids.numel() && onboard_lengths.numel() == user_ids.numel(),
                "onboard metadata must match user_ids length");
    const int batch_size = static_cast<int>(user_ids.numel());
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        const int64_t uid = user_ids[seq_idx].item<int64_t>();
        const int64_t onboard_startpos = onboard_start_indices[seq_idx].item<int64_t>();
        const int64_t onboard_length = onboard_lengths[seq_idx].item<int64_t>();
        TORCH_CHECK(onboard_startpos >= 0 && onboard_length >= 0 &&
                        onboard_startpos <= std::numeric_limits<int64_t>::max() - onboard_length,
                    "invalid onboard range for uid ", uid);

        auto page_it = _uid_to_page_id.find(uid);
        if (page_it == _uid_to_page_id.end()) {
            continue;
        }
        auto& page_ids = page_it->second;
        const int64_t revoke_page_start = onboard_startpos / this->num_tokens_per_page;
        TORCH_CHECK(revoke_page_start <= static_cast<int64_t>(page_ids.size()),
                    "onboard start exceeds cached pages for uid ", uid);
        const int64_t revoke_page_end = std::min((onboard_startpos + onboard_length) / this->num_tokens_per_page,
                                                 static_cast<int64_t>(page_ids.size()));
        const int64_t num_revoke_pages = revoke_page_end - revoke_page_start;
        if (num_revoke_pages <= 0) {
            continue;
        }

        for (int64_t jdx = revoke_page_start; jdx < revoke_page_end; jdx++) {
            _empty_pages.push(page_ids[static_cast<size_t>(jdx)]);
        }
        auto start_it = _uid_to_paged_cache_startpos.find(uid);
        auto length_it = _uid_to_paged_cache_length.find(uid);
        TORCH_CHECK(start_it != _uid_to_paged_cache_startpos.end() && length_it != _uid_to_paged_cache_length.end(),
                    "inconsistent cache metadata for uid ", uid);
        const int64_t revoked_tokens = num_revoke_pages * this->num_tokens_per_page;
        TORCH_CHECK(revoked_tokens <= length_it->second, "revoke range exceeds cached length for uid ", uid);
        TORCH_CHECK(start_it->second <= static_cast<int64_t>(this->max_sequence_length) - revoked_tokens,
                    "revoked cache start exceeds max_sequence_length for uid ", uid);
        start_it->second += static_cast<int>(revoked_tokens);
        length_it->second -= static_cast<int>(revoked_tokens);
        page_ids.erase(page_ids.begin() + revoke_page_start, page_ids.begin() + revoke_page_end);
        if (page_ids.empty()) {
            evict(uid);
        }
    }
}

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> NPUKVCacheManagerImpl::acquire_offload_pages(
    at::Tensor& user_ids, at::Tensor& offloaded_lengths, bool always_offload)
{
    check_cpu_int64_1d(user_ids, "user_ids");
    TORCH_CHECK(user_ids.numel() <= this->max_batch_size, "user_ids exceeds max_batch_size");
    const int batch_size = static_cast<int>(user_ids.numel());
    if (!always_offload && offloaded_lengths.numel() != 0) {
        check_cpu_integral_1d(offloaded_lengths, "offloaded_lengths");
        TORCH_CHECK(offloaded_lengths.numel() == user_ids.numel(),
                    "offloaded_lengths must be empty or match user_ids length");
    }

    if (always_offload) {
        std::vector<int> offload_startpos(static_cast<size_t>(batch_size), 0);
        std::vector<at::Tensor> offload_page_ids_list;
        offload_page_ids_list.reserve(static_cast<size_t>(batch_size));
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            const int64_t uid = user_ids[seq_idx].item<int64_t>();
            const auto page_it = this->_uid_to_page_id.find(uid);
            if (page_it == this->_uid_to_page_id.end()) {
                offload_page_ids_list.push_back(at::empty({0}, at::dtype(torch::kInt32)));
                continue;
            }

            const auto& page_ids = page_it->second;
            if (page_ids.empty()) {
                offload_page_ids_list.push_back(at::empty({0}, at::dtype(torch::kInt32)));
            } else {
                offload_page_ids_list.push_back(
                    at::from_blob(page_ids.data(), {static_cast<int64_t>(page_ids.size())}, at::dtype(torch::kInt32))
                        .clone());
            }
            this->_uid_offload_lock[uid] += 1;
        }
        return std::make_tuple(
            user_ids.clone(),
            at::from_blob(offload_startpos.data(), {offload_startpos.size()}, at::dtype(torch::kInt32)).clone(),
            offload_page_ids_list);
    }

    std::vector<int64_t> offload_user_ids;
    std::vector<int> offload_startpos;
    std::vector<at::Tensor> offload_page_ids_list;

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        const int64_t uid = user_ids[seq_idx].item<int64_t>();
        const auto page_it = this->_uid_to_page_id.find(uid);
        if (page_it == this->_uid_to_page_id.end()) {
            continue;
        }
        const auto start_it = this->_uid_to_paged_cache_startpos.find(uid);
        const auto length_it = this->_uid_to_paged_cache_length.find(uid);
        const auto offloaded_it = this->_uid_to_offloaded_length.find(uid);
        TORCH_CHECK(start_it != this->_uid_to_paged_cache_startpos.end() &&
                        length_it != this->_uid_to_paged_cache_length.end() &&
                        offloaded_it != this->_uid_to_offloaded_length.end(),
                    "inconsistent offload metadata for uid ", uid);

        if (offloaded_lengths.numel() == batch_size) {
            const int64_t value = offloaded_lengths[seq_idx].item<int64_t>();
            TORCH_CHECK(value >= 0 && value <= this->max_sequence_length,
                        "offloaded length must be in [0, max_sequence_length] for uid ", uid);
            offloaded_it->second = static_cast<int>(value);
        }
        const int offloaded_length = offloaded_it->second;
        const int cached_startpos = start_it->second;
        const int cached_end_index = cached_startpos + length_it->second;
        if (offloaded_length < cached_startpos) {
            continue;
        }
        TORCH_CHECK(offloaded_length <= cached_end_index, "offloaded length exceeds cached range for uid ", uid);

        if (cached_end_index - offloaded_length >= this->num_tokens_per_chunk) {
            offload_user_ids.push_back(uid);
            offload_startpos.push_back(offloaded_length);

            const int pages_offload_start = (offloaded_length - cached_startpos) / this->num_tokens_per_page;
            const int pages_offload_num = (cached_end_index - offloaded_length) / this->num_tokens_per_page;
            const auto& page_ids = page_it->second;
            TORCH_CHECK(pages_offload_start >= 0 && pages_offload_num >= 0 &&
                            static_cast<size_t>(pages_offload_start + pages_offload_num) <= page_ids.size(),
                        "offload page range exceeds cached pages for uid ", uid);
            offload_page_ids_list.push_back(
                at::from_blob(page_ids.data() + pages_offload_start, {pages_offload_num}, at::dtype(torch::kInt32))
                    .clone());
            this->_uid_offload_lock[uid] += 1;
        }
    }

    return std::make_tuple(
        at::from_blob(offload_user_ids.data(), {offload_user_ids.size()}, at::dtype(torch::kInt64)).clone(),
        at::from_blob(offload_startpos.data(), {offload_startpos.size()}, at::dtype(torch::kInt32)).clone(),
        offload_page_ids_list);
}

void NPUKVCacheManagerImpl::release_offload_pages(at::Tensor user_ids, at::Tensor offload_start_indices,
                                                  at::Tensor offload_lengths, const std::vector<int>& offloaded)
{
    check_cpu_int64_1d(user_ids, "user_ids");
    const int64_t batch_size = user_ids.numel();
    TORCH_CHECK(static_cast<int64_t>(offloaded.size()) == batch_size, "offloaded must match user_ids length");
    const bool needs_metadata = std::any_of(offloaded.begin(), offloaded.end(), [](int value) { return value != 0; });
    if (needs_metadata) {
        check_cpu_integral_1d(offload_start_indices, "offload_start_indices");
        check_cpu_integral_1d(offload_lengths, "offload_lengths");
        TORCH_CHECK(offload_start_indices.numel() == batch_size && offload_lengths.numel() == batch_size,
                    "offload metadata must match user_ids length when offloaded is true");
    }

    for (int64_t idx = 0; idx < batch_size; idx++) {
        const int64_t uid = user_ids[idx].item<int64_t>();
        auto lock_it = this->_uid_offload_lock.find(uid);
        if (lock_it == this->_uid_offload_lock.end())
            continue;

        if (offloaded[idx]) {
            const auto offloaded_it = _uid_to_offloaded_length.find(uid);
            const auto start_it = _uid_to_paged_cache_startpos.find(uid);
            TORCH_CHECK(
                offloaded_it != _uid_to_offloaded_length.end() && start_it != _uid_to_paged_cache_startpos.end(),
                "inconsistent offload metadata for uid ", uid);
            total_offloaded_pages -= std::max(0, (offloaded_it->second - start_it->second) / this->num_tokens_per_page);
            const int64_t offload_start = offload_start_indices[idx].item<int64_t>();
            const int64_t offload_length = offload_lengths[idx].item<int64_t>();
            TORCH_CHECK(offload_start >= 0 && offload_length >= 0 &&
                            offload_start <= static_cast<int64_t>(this->max_sequence_length) - offload_length,
                        "invalid offload range for uid ", uid);
            offloaded_it->second = static_cast<int>(offload_start + offload_length);
            total_offloaded_pages += std::max(0, (offloaded_it->second - start_it->second) / this->num_tokens_per_page);
        }
        lock_it->second -= 1;
        if (lock_it->second <= 0) {
            this->_uid_offload_lock.erase(lock_it);
        }
    }
}

}  // namespace kvcache
