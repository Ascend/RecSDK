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
#include "split_embedding_codegen_forward_unweighted.h"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;

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
                                                          const Tensor& hash_indices)
{
    const int64_t totalD = total_D.guard_int(__FILE__, __LINE__);
    const int64_t maxD = max_D.guard_int(__FILE__, __LINE__);

    const at::OptionalDeviceGuard guard(device_of(dev_weights));

    int64_t featCnt = weights_placements.size(0);
    int32_t totalLen = indices.numel();
    if (featCnt == 0) {
        return at::Tensor();
    }

    if (totalLen == 0) {
        return at::Tensor();
    }

    int64_t batchSize = (offsets.size(0) - 1) / featCnt;
    auto output = at::full({batchSize, totalD}, 0.0, dev_weights.options());

    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        output = at::full({totalLen, maxD}, 0.0, dev_weights.options());
    }

    int64_t experimental = static_cast<int64_t>(is_experimental);
    EXEC_NPU_CMD(aclnnSplitEmbeddingCodegenForwardUnweighted, dev_weights, uvm_weights,         lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, indices, offsets, lxu_cache_locations, hash_indices,
                 totalD, maxD, pooling_mode, output_dtype, experimental, output);
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
          "    Tensor hash_indices = None "
          ") -> Tensor");

    m.impl("split_embedding_codegen_forward_unweighted_cuda",
        torch::dispatch(c10::DispatchKey::Autograd,
                        TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_forward_unweighted_npu)));
}
