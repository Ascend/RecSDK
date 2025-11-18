/**
 * @file split_embedding_codegen_forward_unweighted.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "torch/extension.h"
#include "../common/pytorch_npu_helper.hpp"
#include "split_embedding_codegen_common_utils.h"
#include "split_embedding_codegen_forward_unweighted.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;


aclError copy_gm_to_gm(void* source_memory_ptr, // output
                       const std::vector<torch::Tensor>& target_tensors, // 梯度表
                       torch::Tensor size, // unique_offsets_size
                       torch::Tensor grad_accumulate_offsets_size)
{ // grad_accumulate_offsets_size
    // 空指针校验
    if (source_memory_ptr == nullptr) {
        AT_ERROR("Source memory pointer is null");
        return ACL_ERROR_INVALID_PARAM;
    }

    size_t target_tensors_size = target_tensors.size();
    if (target_tensors_size == 0) {
    return ACL_SUCCESS;
    }

    // 检查size张量的长度是否足够
    if (size.size(0) < target_tensors.size() + 1) {
        AT_ERROR("Size tensor is too short");
        return ACL_ERROR_INVALID_PARAM;
    }

    for (size_t i = 0; i < target_tensors.size(); ++i) {
        const auto& target_tensor = target_tensors[i];

        void* tensor_ptr = target_tensor.data_ptr();
        if (tensor_ptr == nullptr) {
            AT_ERROR("Tensor data pointer is null for tensor ", i);
            continue;
        }

        auto offset = size[i].item<int64_t>();
        auto size_bytes = (size[i + 1] - size[i]).item<int64_t>();
        auto grad_accumulate_offset = grad_accumulate_offsets_size[i].item<int64_t>();
        // 检查指针越界
        if (grad_accumulate_offset < 0 || size_bytes < 0) {
            AT_ERROR("Invalid offset or size for tensor ", i);
            continue;
        }

        aclError ret = aclrtMemcpy(
            tensor_ptr + grad_accumulate_offset,
            size_bytes,
            reinterpret_cast<char*>(source_memory_ptr) + offset,
            size_bytes,
            ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            const char* error_msg = aclGetRecentErrMsg();
            AT_ERROR("D2D copy failed for tensor ", i, ": ", error_msg);
        }
    }
    return ACL_SUCCESS;
}


// using namespace fbgemm_gpu;
namespace fbgemm_npu_lookups {
at::Tensor split_embedding_codegen_forward_unweighted_npu(const at::Tensor& dev_weights,
                                                          const at::Tensor& uvm_weights,
                                                          const at::Tensor& lxu_cache_weights,
                                                          const at::Tensor& weights_placements,
                                                          const at::Tensor& weights_offsets,
                                                          const at::Tensor& D_offsets,
                                                          const c10::SymInt total_D,
                                                          const c10::SymInt max_D,
                                                          const at::Tensor& indices,
                                                          const at::Tensor& offsets,
                                                          const int64_t pooling_mode,
                                                          const at::Tensor& lxu_cache_locations,
                                                          const at::Tensor& uvm_cache_stats,
                                                          const int64_t output_dtype,
                                                          const bool is_experimental,
                                                          const at::Tensor& hash_indices,
                                                          const at::Tensor& offset_per_key,
                                                          const at::Tensor& rows_per_table)
{
    const int64_t totalD = total_D.guard_int(__FILE__, __LINE__);
    const int64_t maxD = max_D.guard_int(__FILE__, __LINE__);

    const at::OptionalDeviceGuard guard(device_of(dev_weights));

    validate_forward_data_inputs(dev_weights, weights_offsets, D_offsets, indices,
                                 offsets, hash_indices, offset_per_key, rows_per_table);

    int64_t featCnt = weights_offsets.size(0);
    int32_t totalLen = indices.numel();

    TORCH_CHECK(featCnt > 0, "weights_offsets size must be great than 0.");
    TORCH_CHECK(totalLen > 0, "indices can not be empty tensor.");
    TORCH_CHECK(offsets.size(0) > 1, "offsets dim_0 must be great than 1.");

    int64_t batchSizeRes = (offsets.size(0) - 1) % featCnt;
    TORCH_CHECK(batchSizeRes == 0, "offset size = ", offsets.size(0),
                " is incorrect for feature count = ", featCnt);
    int64_t batchSize = (offsets.size(0) - 1) / featCnt;
    auto output = at::full({batchSize, totalD}, 0.0, dev_weights.options());

    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        output = at::full({totalLen, maxD}, 0.0, dev_weights.options());
    }

    int64_t experimental = static_cast<int64_t>(is_experimental);
    EXEC_NPU_CMD(aclnnSplitEmbeddingCodegenForwardUnweighted, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, indices, offsets, lxu_cache_locations, hash_indices,
                 offset_per_key, rows_per_table, totalD, maxD, pooling_mode, output_dtype, experimental, output);
    return output;
}

}; // namespace fbgemm_npu_lookups

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_forward_unweighted_cuda("
          "    Tensor dev_weights, "
          "    Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    SymInt total_D, "
          "    SymInt max_D, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    Tensor uvm_cache_stats, "
          "    int output_dtype, "
          "    bool is_experimental, "
          "    Tensor hash_indices = None, "
          "    Tensor offset_per_key = None, "
          "    Tensor rows_per_table = None "
          ") -> Tensor");

    m.impl("split_embedding_codegen_forward_unweighted_cuda",
        torch::dispatch(c10::DispatchKey::Autograd,
                        TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_forward_unweighted_npu)));
}
