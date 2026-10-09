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
#include "npu_kvcache_manager_impl.h"

#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <tiling/platform/platform_ascendc.h>

namespace kvcache {
void AppendPagedKVCache(uint16_t* k_data, uint16_t* v_data, int32_t* indices, int32_t* indptr, uint32_t num_heads,
                        uint32_t head_dim, uint32_t page_size, uint32_t stride_page, uint32_t stride_n,
                        uint32_t stride_h, uint16_t* append_key, uint16_t* append_value, int32_t* batch_indices,
                        int32_t* positions, int32_t* offsets, int32_t* nnz_dev, size_t append_k_stride_n,
                        size_t append_k_stride_h, size_t append_v_stride_n, size_t append_v_stride_h, uint32_t num_ctas,
                        aclrtStream stream);
}

namespace {

void check_npu_int32_1d(const at::Tensor& tensor, const at::Device& expected_device, const char* name)
{
    TORCH_CHECK(tensor.device() == expected_device, name, " must be on ", expected_device);
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
    TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// The kernel reinterprets the payload as uint16_t, so only 2-byte dtypes are
// valid. Contiguity is deliberately NOT required: paged caches reach this entry
// point as unbind() views, and the kernel consumes their strides explicitly.
void check_npu_kv_2byte(const at::Tensor& tensor, const at::Device& expected_device, int64_t expected_dim,
                        const char* name)
{
    TORCH_CHECK(tensor.device() == expected_device, name, " must be on ", expected_device);
    TORCH_CHECK(tensor.element_size() == 2, name, " must have a 2-byte dtype (float16 or bfloat16)");
    TORCH_CHECK(tensor.dim() == expected_dim, name, " must have ", expected_dim, " dimensions");
}

}  // namespace

// nnz (the host-side token count) is accepted but unused below: the kernel
// path reads the device-side nnz_npu instead. Kept because
// recsys_kvcache_manager_npu/npu_kvcache_manager.py's put() already calls this
// binding positionally with kvcache_metadata.new_history_nnz in this slot;
// dropping it needs a synchronized change there too.
static void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                                  at::Tensor positions, at::Tensor seqlen_offsets, at::Tensor nnz_npu, unsigned int nnz,
                                  at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor kv_indices,
                                  at::Tensor kv_indptr, at::Tensor kv_last_page_len, int64_t kv_layout)
{
    auto device = append_key.device();
    check_npu_kv_2byte(append_key, device, 3, "append_key");
    check_npu_kv_2byte(append_value, device, 3, "append_value");
    TORCH_CHECK(append_key.sizes() == append_value.sizes(), "append_key and append_value must have the same shape");
    check_npu_kv_2byte(paged_k_cache, device, 4, "paged_k_cache");
    check_npu_kv_2byte(paged_v_cache, device, 4, "paged_v_cache");
    check_npu_int32_1d(batch_indices, device, "batch_indices");
    check_npu_int32_1d(positions, device, "positions");
    check_npu_int32_1d(seqlen_offsets, device, "seqlen_offsets");
    check_npu_int32_1d(nnz_npu, device, "nnz_npu");
    check_npu_int32_1d(kv_indices, device, "kv_indices");
    check_npu_int32_1d(kv_indptr, device, "kv_indptr");
    check_npu_int32_1d(kv_last_page_len, device, "kv_last_page_len");
    TORCH_CHECK(kv_layout == 0 || kv_layout == 1, "kv_layout must be 0 (NHD) or 1 (HND), got ", kv_layout);

    c10_npu::OptionalNPUGuard device_guard(device);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t max_cores = platform->GetCoreNumAiv();

    unsigned int num_heads, page_size, head_dim;
    head_dim = paged_k_cache.size(3);
    if (kv_layout == 1) {
        num_heads = paged_k_cache.size(1);
        page_size = paged_k_cache.size(2);
    } else {
        page_size = paged_k_cache.size(1);
        num_heads = paged_k_cache.size(2);
    }

    auto stride_page = paged_k_cache.stride(0);
    auto stride_n = (kv_layout == 1) ? head_dim : num_heads * head_dim;
    auto stride_h = (kv_layout == 1) ? page_size * head_dim : head_dim;

    auto append_k_strides = append_key.strides();
    auto append_k_stride_n = append_k_strides[0];
    auto append_k_stride_h = append_k_strides[1];
    auto append_v_strides = append_value.strides();
    auto append_v_stride_n = append_v_strides[0];
    auto append_v_stride_h = append_v_strides[1];

    kvcache::AppendPagedKVCache(
        reinterpret_cast<uint16_t*>(paged_k_cache.data_ptr()), reinterpret_cast<uint16_t*>(paged_v_cache.data_ptr()),
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()), num_heads, head_dim,
        page_size, stride_page, stride_n, stride_h, reinterpret_cast<uint16_t*>(append_key.data_ptr()),
        reinterpret_cast<uint16_t*>(append_value.data_ptr()), static_cast<int32_t*>(batch_indices.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()), static_cast<int32_t*>(seqlen_offsets.data_ptr()),
        static_cast<int32_t*>(nnz_npu.data_ptr()), append_k_stride_n, append_k_stride_h, append_v_stride_n,
        append_v_stride_h, max_cores, stream);
}

PYBIND11_MODULE(kvcache_npu_cpp, m)
{
    py::class_<kvcache::HostKVStorageImpl>(m, "HostKVStorageImpl", py::module_local())
        .def(py::init<int, int, int, int, int64_t, int64_t, int64_t, int64_t, int>(), py::arg("num_layers"),
             py::arg("num_kv_heads"), py::arg("kv_headdim"), py::arg("num_tokens_per_page"),
             py::arg("num_tokens_per_chunk"), py::arg("capacity_per_layer"), py::arg("max_batch_size"),
             py::arg("max_sequence_length"), py::arg("device_idx"))
        .def("register_npu_cache_table", &kvcache::HostKVStorageImpl::register_npu_cache_table)
        .def("lookup", &kvcache::HostKVStorageImpl::lookup)
        .def("get_kvdata_tensor", &kvcache::HostKVStorageImpl::get_kvdata_tensor)
        .def("onload_kvcache", &kvcache::HostKVStorageImpl::onload_kvcache)
        .def("offload_kvcache", &kvcache::HostKVStorageImpl::offload_kvcache)
        .def("finish_offload", &kvcache::HostKVStorageImpl::finish_offload)
        .def("cancel_offload", &kvcache::HostKVStorageImpl::cancel_offload)
        .def("evict", &kvcache::HostKVStorageImpl::evict)
        .def("evict_all", &kvcache::HostKVStorageImpl::evict_all);

    py::class_<kvcache::NPUKVCacheManagerImpl>(m, "NPUKVCacheManagerImpl", py::module_local())
        .def(py::init<int, int, int, int, int, int, int, int, int, int>(), py::arg("num_layers"),
             py::arg("num_kv_heads"), py::arg("kv_headdim"), py::arg("num_tokens_per_page"),
             py::arg("num_tokens_per_chunk"), py::arg("num_primary_cache_pages"), py::arg("num_buffer_pages"),
             py::arg("max_batch_size"), py::arg("max_sequence_length"), py::arg("device_idx"))
        .def("lookup", &kvcache::NPUKVCacheManagerImpl::lookup)
        .def("evict", &kvcache::NPUKVCacheManagerImpl::evict)
        .def("evict_offloaded", &kvcache::NPUKVCacheManagerImpl::evict_offloaded)
        .def("evict_all", &kvcache::NPUKVCacheManagerImpl::evict_all)
        .def("retain", &kvcache::NPUKVCacheManagerImpl::retain)
        .def("allocate", &kvcache::NPUKVCacheManagerImpl::allocate)
        .def("revoke_onboard_pages", &kvcache::NPUKVCacheManagerImpl::revoke_onboard_pages)
        .def("check_for_offload", &kvcache::NPUKVCacheManagerImpl::check_for_offload)
        .def("acquire_offload_pages", &kvcache::NPUKVCacheManagerImpl::acquire_offload_pages)
        .def("release_offload_pages", &kvcache::NPUKVCacheManagerImpl::release_offload_pages);

    py::class_<kvcache::KVOnloadHandle>(m, "KVOnloadHandle", py::module_local())
        .def(py::init<int>(), py::arg("num_layers"))
        .def("wait_layer", &kvcache::KVOnloadHandle::wait_layer);

    py::class_<kvcache::KVOffloadHandle>(m, "KVOffloadHandle", py::module_local())
        .def(py::init<int>(), py::arg("num_layers"))
        .def("try_wait_layer", &kvcache::KVOffloadHandle::try_wait_layer)
        .def("get_user_ids", &kvcache::KVOffloadHandle::get_user_ids)
        .def("get_start_indices", &kvcache::KVOffloadHandle::get_start_indices)
        .def("get_lengths", &kvcache::KVOffloadHandle::get_lengths);

    m.def("append_kvcache", &append_paged_kv_cache, "append paged kv cache on NPU",
          py::call_guard<py::gil_scoped_release>());
}
