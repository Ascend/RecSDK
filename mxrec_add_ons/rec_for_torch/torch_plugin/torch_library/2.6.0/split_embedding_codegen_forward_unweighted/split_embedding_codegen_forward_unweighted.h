/**
 * @file split_embedding_codegen_forward_unweighted.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef MXREC_ADD_ONS_SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_H
#define MXREC_ADD_ONS_SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_H
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "torch/extension.h"
#include "../common/pytorch_npu_helper.hpp"

enum class OptimizerType {
    ADAGRAD = 1,
    ADAM = 2,
    SGD = 3
};

enum class SparseType {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
    BF16 = 5,
    FP8 = 6,
    INVALID = 7,
};

enum class PoolingMode {
    SUM = 0,
    MEAN = 1,
    NONE = 2
};

namespace fbgemm_npu_lookups {
at::Tensor split_embedding_codegen_forward_unweighted_cuda(const at::Tensor& dev_weights,
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
                                                           const at::Tensor& offset_per_key);

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
                                                          const at::Tensor& offset_per_key);
}; // namespace fbgemm_npu_lookups
#endif // MXREC_ADD_ONS_SPLIT_EMBEDDING_CODEGEN_FORWARD_UNWEIGHTED_H
