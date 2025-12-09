/**
 * @file split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/extension.h>

#include "backward_constant.h"
#include "split_embedding_codegen_forward_unweighted.h"
#include "split_embedding_codegen_common_utils.h"
#include "../common/common_utils.h"
#include "../common/pytorch_npu_helper.hpp"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;
using namespace optim_param_idx_fbgemm_120;

namespace fbgemm_npu_lookups {
Tensor split_embedding_codegen_lookup_rowwise_adagrad_function(
    const Tensor& placeholder_autograd_tensor,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    const Tensor& lxu_cache_locations,
    const bool gradient_clipping,
    const double max_gradient,
    const bool stochastic_rounding,
    Tensor momentum1_dev,
    Tensor momentum1_uvm,
    Tensor momentum1_placements,
    Tensor momentum1_offsets,
    double eps,
    double learning_rate,
    double weight_decay,
    int64_t weight_decay_mode,
    double max_norm,
    const int64_t output_dtype,
    const std::optional<Tensor>& B_offsets,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature,
    const c10::SymInt max_B,
    const c10::SymInt max_B_feature_rank,
    const c10::SymInt vbe_output_size,
    const bool is_experimental_tbe,
    const bool use_uniq_cache_locations_bwd,
    const bool use_homogeneous_placements,
    const std::optional<Tensor>& uvm_cache_stats,
    const std::optional<Tensor>& prev_iter_dev,
    const int64_t iter,
    const bool apply_global_weight_decay,
    const double gwd_lower_bound,
    const bool mixed_D,
    const std::optional<tensor_list>& grad_accumulate,
    const std::optional<at::Tensor>& grad_accumulate_offsets,
    const std::optional<Tensor>& hash_indices,
    const std::optional<at::Tensor>& unique_ids,
    const std::optional<at::Tensor>& unique_offsets,
    const std::optional<at::Tensor>& unique_inverse,
    const std::optional<at::Tensor>& table_grad_accumulate_offsets,
    const std::optional<Tensor>& rows_per_table)
{
    at::Tensor offset_per_key = compute_offset_per_key(offsets, weights_offsets, D_offsets);
    
    auto output = split_embedding_codegen_forward_unweighted_npu(
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        indices,
        offsets,
        pooling_mode,
        lxu_cache_locations,
        uvm_cache_stats.value_or(at::empty({0}, uvm_weights.options().dtype(at::kInt))),
        output_dtype,
        is_experimental_tbe,
        hash_indices.value_or(Tensor()),
        offset_per_key,
        rows_per_table.value_or(Tensor()));
    return output;
}

Tensor split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
    const Tensor& placeholder_autograd_tensor,
    const at::TensorList weights,
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    const int64_t output_dtype,
    const std::vector<std::optional<at::Tensor>>& aux_tensor,
    const std::vector<int64_t>& aux_int,
    const std::vector<double>& aux_float,
    c10::List<bool> aux_bool,
    at::TensorList momentum1,
    Tensor learning_rate_tensor,
    std::vector<int64_t> optim_int,
    std::vector<double> optim_float,
    const c10::SymInt max_B = -1,
    const c10::SymInt max_B_feature_rank = -1,
    const c10::SymInt vbe_output_size = -1)
{
    check_param_len(weights.size(), WEIGHTS_SIZE, "weights");
    auto& dev_weights = weights[DEV_WEIGHTS_INDEX];
    auto& uvm_weights = weights[UVM_WEIGHTS_INDEX];
    auto& weights_placements = weights[WEIGHTS_PLACEMENTS_INDEX];
    auto& weights_offsets = weights[WEIGHTS_OFFSETS_INDEX];
    auto& lxu_cache_weights = weights[LXU_CACHE_WEIGHTS_INDEX];

    // unpacking from aux_tensor
    check_param_len(aux_tensor.size(), AUX_TENSOR_SIZE, "aux_tensor");
    auto lxu_cache_locations = aux_tensor[LXU_CACHE_LOCATIONS_INDEX].value_or(
        at::empty({0}, dev_weights.options().dtype(at::kInt)));
    auto uvm_cache_stats = aux_tensor[UVM_CACHE_STATS_INDEX].value_or(
        at::empty({0}, dev_weights.options().dtype(at::kInt)));
    // unpacking from aux_bool
    check_param_len(aux_bool.size(), AUX_BOOL_SIZE, "aux_bool");
    bool is_experimental_tbe = aux_bool[IS_EXPERIMENTAL_TBE_INDEX];

    at::Tensor offset_per_key = compute_offset_per_key(offsets, weights_offsets, D_offsets);

    auto output = split_embedding_codegen_forward_unweighted_npu(
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        indices,
        offsets,
        pooling_mode,
        lxu_cache_locations,
        uvm_cache_stats,
        output_dtype,
        is_experimental_tbe,
        Tensor(),
        offset_per_key,
        Tensor());
    return output;
}
};  // namespace fbgemm_npu_lookups

// dispatch FBGEMM interface to NPU op
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_lookup_rowwise_adagrad_function("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor(a!) dev_weights, "
          "    Tensor(b!) uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    SymInt total_D, "
          "    SymInt max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    Tensor lxu_cache_locations, "
          "    bool gradient_clipping, "
          "    float max_gradient, "
          "    bool stochastic_rounding, "
          "    Tensor momentum1_dev, "
          "    Tensor momentum1_uvm, "
          "    Tensor momentum1_placements, "
          "    Tensor momentum1_offsets, "
          "    float eps = 0, "
          "    float learning_rate = 0, "
          "    float weight_decay = 0.0, "
          "    int weight_decay_mode = 0.0, "
          "    float max_norm = 0.0, "
          "    int output_dtype=0, "
          "    Tensor? B_offsets=None, "
          "    Tensor? vbe_output_offsets_feature_rank=None, "
          "    Tensor? vbe_B_offsets_rank_per_feature=None, "
          "    SymInt max_B=-1, "
          "    SymInt max_B_feature_rank=-1, "
          "    SymInt vbe_output_size=-1, "
          "    bool is_experimental=False, "
          "    bool use_uniq_cache_locations_bwd=False, "
          "    bool use_homogeneous_placements=False, "
          "    Tensor? uvm_cache_stats=None, "
          "    Tensor? prev_iter_dev=None, int iter=0, "
          "    bool apply_global_weight_decay=False, "
          "    float gwd_lower_bound=0, "
          "    bool mixed_D=True, "
          "    Tensor[]? grad_accumulate = None, "
          "    Tensor? grad_accumulate_offsets = None, "
          "    Tensor? hash_indices = None, "
          "    Tensor? unique_ids = None, "
          "    Tensor? unique_offsets = None, "
          "    Tensor? unique_inverse = None, "
          "    Tensor? table_grad_accumulate_offsets = None, "
          "    Tensor? rows_per_table=None "
          ") -> Tensor");

    m.impl("split_embedding_codegen_lookup_rowwise_adagrad_function",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_rowwise_adagrad_function)));

    m.impl("split_embedding_codegen_lookup_rowwise_adagrad_function",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_rowwise_adagrad_function)));
}

// dispatch FBGEMM1.2.0 interface to NPU op
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.impl("split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_rowwise_adagrad_function_pt2)));

    m.impl("split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_rowwise_adagrad_function_pt2)));
}