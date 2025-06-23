/**
 * @file backward_codegen_sgd_unweighted_exact.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "torch/extension.h"
#include "split_embedding_codegen_forward_unweighted.h"
#include "../common/pytorch_npu_helper.hpp"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;

namespace fbgemm_npu_lookups {

Tensor split_embedding_backward_codegen_sgd_unweighted_exact_cuda(const Tensor& grad_output,
                                                                  const Tensor& dev_weights,
                                                                  const Tensor& uvm_weights,
                                                                  const Tensor& lxu_cache_weights,
                                                                  const Tensor& weights_placements,
                                                                  const Tensor& weights_offsets,
                                                                  const Tensor& D_offsets,
                                                                  const c10::SymInt max_D,
                                                                  const Tensor& hash_size_cumsum,
                                                                  const int64_t total_hash_size_bits,
                                                                  const Tensor& indices,
                                                                  const Tensor& offsets,
                                                                  const int64_t pooling_mode,
                                                                  const Tensor& lxu_cache_locations,
                                                                  const int64_t BT_block_size,
                                                                  const int64_t max_segment_length_per_warp,
                                                                  const bool stochastic_rounding,
                                                                  const int64_t info_B_num_bits,
                                                                  const int64_t info_B_mask_int64,
                                                                  const bool use_uniq_cache_locations,
                                                                  const bool use_homogeneous_placements,
                                                                  const Tensor& hash_indices,
                                                                  const Tensor& unique_ids,
                                                                  const Tensor& unique_offsets,
                                                                  const Tensor& unique_inverse,
                                                                  double learning_rate = 0,
                                                                  bool is_dynamic = false);

class SplitLookupSGD : public torch::autograd::Function<SplitLookupSGD> {
public:
    static constexpr bool isTraceable = true;

    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
                                                  const Tensor& placeholder_autograd_tensor,
                                                  const int64_t output_dtype,
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
                                                  const c10::optional<Tensor>& hash_indices,
                                                  const c10::optional<at::Tensor>& unique_ids,
                                                  const c10::optional<at::Tensor>& unique_offsets,
                                                  const c10::optional<at::Tensor>& unique_inverse,
                                                  const Tensor& offsets,
                                                  const int64_t pooling_mode,
                                                  const std::optional<Tensor>& indice_weights,
                                                  const std::optional<Tensor>& feature_requires_grad,
                                                  const Tensor& lxu_cache_locations,
                                                  std::optional<Tensor> uvm_cache_stats,
                                                  const bool gradient_clipping,
                                                  const double max_gradient,
                                                  const bool stochastic_rounding,
                                                  const bool is_experimental,
                                                  const bool use_uniq_cache_locations_bwd,
                                                  const bool use_homogeneous_placements,
                                                  double learning_rate = 0,
                                                  bool is_dynamic = false)
    {
        const auto T = weights_offsets.size(0);
        if (T == 0) {
            return {at::Tensor()};
        }

        const auto max_B_ = offsets.size(0) / T;
        // NOTE: The `local_uvm_cache_stats` variable held by the nn.Module has dtype int32_t
        const auto uvm_cache_stats_ = uvm_cache_stats.value_or(at::empty({0}, uvm_weights.options().dtype(at::kInt)));

        auto info_B_num_bits = max_B_;
        auto info_B_mask = T;

        ctx->save_for_backward({dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets,
                                D_offsets, hash_size_cumsum, indices, offsets, indice_weights.value_or(Tensor()),
                                feature_requires_grad.value_or(Tensor()), lxu_cache_locations,
                                hash_indices.value_or(Tensor()), unique_ids.value_or(at::Tensor()),
                                unique_offsets.value_or(at::Tensor()), unique_inverse.value_or(at::Tensor())});
        ctx->saved_data["max_D"] = max_D;
        ctx->saved_data["pooling_mode"] = pooling_mode;
        ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
        ctx->saved_data["gradient_clipping"] = gradient_clipping;
        ctx->saved_data["max_gradient"] = max_gradient;
        ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
        ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
        const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
        ctx->saved_data["info_B_mask"] = info_B_mask_int64;
        ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
        ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;
        ctx->saved_data["learning_rate"] = learning_rate;
        ctx->saved_data["is_dynamic"] = is_dynamic;
        const auto& flatten_dev_weights = dev_weights;
        // not surport  indice_weights
        if (!indice_weights) {
            static auto embedding_codegen_forward_op =
                torch::Dispatcher::singleton()
                    .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_unweighted_cuda", "")
                    .typed<decltype(split_embedding_codegen_forward_unweighted_cuda)>();

            return {embedding_codegen_forward_op.call(
                flatten_dev_weights, uvm_weights, lxu_cache_weights, weights_placements,
                weights_offsets, D_offsets, total_D, max_D, indices, offsets,
                pooling_mode, lxu_cache_locations, uvm_cache_stats_, output_dtype, is_experimental,
                hash_indices.value_or(Tensor()), unique_inverse.value_or(at::Tensor()), is_dynamic)};
        }
        return {at::Tensor()};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs)
    {
        const auto saved = ctx->get_saved_variables();
        auto savedItr = std::begin(saved);
        auto dev_weights = *savedItr++;
        auto uvm_weights = *savedItr++;
        auto lxu_cache_weights = *savedItr++;
        auto weights_placements = *savedItr++;
        auto weights_offsets = *savedItr++;
        auto D_offsets = *savedItr++;
        auto hash_size_cumsum = *savedItr++;
        auto indices = *savedItr++;
        auto offsets = *savedItr++;
        auto indice_weights = *savedItr++;
        auto feature_requires_grad = *savedItr++;
        auto lxu_cache_locations = *savedItr++;
        auto hash_indices = *savedItr++;
        auto unique_ids = *savedItr++;
        auto unique_offsets = *savedItr++;
        auto unique_inverse = *savedItr++;
        auto max_D = ctx->saved_data["max_D"].toSymInt();
        auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
        auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
        auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
        auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
        auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
        const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
        const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
        const auto use_uniq_cache_locations_bwd = ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
        const auto use_homogeneous_placements = ctx->saved_data["use_homogeneous_placements"].toBool();
        auto learning_rate = ctx->saved_data["learning_rate"].toDouble();
        auto is_dynamic = ctx->saved_data["is_dynamic"].toBool();

        TORCH_CHECK_EQ(grad_outputs.size(), 1);

        constexpr int32_t BT_block_size = 32;
        constexpr int32_t max_segment_length_per_warp = 32;

        using torch::autograd::Variable;
        auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];

        static auto embedding_codegen_unweighted_backward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_backward_codegen_sgd_unweighted_exact_cuda", "")
                .typed<decltype(split_embedding_backward_codegen_sgd_unweighted_exact_cuda)>();

        const auto grad_dev_weights = embedding_codegen_unweighted_backward_op.call(
            grad_output, dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, pooling_mode, lxu_cache_locations,
            BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
            use_uniq_cache_locations_bwd, use_homogeneous_placements, hash_indices, unique_ids, unique_offsets,
            unique_inverse, learning_rate, is_dynamic);
        return {
            Tensor(),         // placeholder autograd tensor
            Variable(),       // output_dtype
            grad_dev_weights, // dev_weights
            Variable(),       // uvm_weights
            Variable(),       // lxu_cache_weights
            Variable(),       // weights_placements
            Variable(),       // weights_offsets
            Variable(),       // D_offsets
            Variable(),       // total_D
            Variable(),       // max_D
            Variable(),       // hash_size_cumsum
            Variable(),       // total_hash_size_bits
            Variable(),       // indices
            Variable(),       // offsets
            Variable(),       // pooling_mode
            Variable(),       // indice_weights
            Variable(),       // feature_requires_grad
            Variable(),       // lxu_cache_locations
            Variable(),       // uvm_cache_stats
            Variable(),       // gradient_clipping
            Variable(),       // max_gradient
            Variable(),       // stochastic_rounding
            Variable(),       // is_experimental
            Variable(),       // use_uniq_cache_locations_bwd
            Variable(),       // use_homogeneous_placements
            Variable(),       // hash_indices
            Variable(),       // unique_ids
            Variable(),       // unique_offsets
            Variable(),       // unique_inverse
            Variable(),       // learning_rate
            Variable(),       // is_dynamic
        };
    }
};

///@ingroup embedding-cuda
Tensor split_embedding_codegen_lookup_sgd_function(
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
    const c10::optional<Tensor>& hash_indices = c10::optional<Tensor>(),
    const c10::optional<at::Tensor>& unique_ids = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_inverse = c10::optional<at::Tensor>(),
    double learning_rate = 0,
    bool is_dynamic = false,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const std::optional<Tensor>& B_offsets = c10::nullopt,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank = c10::nullopt,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature = c10::nullopt,
    const c10::SymInt max_B = -1,
    const c10::SymInt max_B_feature_rank = -1,
    const c10::SymInt vbe_output_size = -1,
    const bool is_experimental_tbe = false, // formerly named is_experimental
    const bool use_uniq_cache_locations_bwd = false,
    const bool use_homogeneous_placements = false,
    const std::optional<Tensor>& uvm_cache_stats = c10::nullopt,
    const std::optional<Tensor>& prev_iter_dev = c10::nullopt,
    const int64_t iter = 0,
    const bool apply_global_weight_decay = false,
    const double gwd_lower_bound = 0)
{
    // Set to experimental if either the feature is enabled in JK, or the user specifies to use TBEv2
    const auto is_experimental = is_experimental_tbe;

    return SplitLookupSGD::apply(
        placeholder_autograd_tensor, output_dtype, dev_weights, uvm_weights, lxu_cache_weights, weights_placements,
        weights_offsets, D_offsets, total_D, max_D, hash_size_cumsum, total_hash_size_bits, indices, hash_indices,
        unique_ids, unique_offsets, unique_inverse, offsets, pooling_mode, indice_weights, feature_requires_grad,
        lxu_cache_locations, uvm_cache_stats, gradient_clipping, max_gradient, stochastic_rounding, is_experimental,
        use_uniq_cache_locations_bwd, use_homogeneous_placements, learning_rate)[0];
}

at::Tensor split_embedding_backward_codegen_sgd_unweighted_exact_npu(const Tensor& grad_output,
                                                                     const Tensor& dev_weights,
                                                                     const Tensor& uvm_weights,
                                                                     const Tensor& lxu_cache_weights,
                                                                     const Tensor& weights_placements,
                                                                     const Tensor& weights_offsets,
                                                                     const Tensor& D_offsets,
                                                                     const c10::SymInt max_D,
                                                                     const Tensor& hash_size_cumsum,
                                                                     const int64_t total_hash_size_bits,
                                                                     const Tensor& indices,
                                                                     const Tensor& offsets,
                                                                     const int64_t pooling_mode,
                                                                     const Tensor& lxu_cache_locations,
                                                                     const int64_t BT_block_size,
                                                                     const int64_t max_segment_length_per_warp,
                                                                     const bool stochastic_rounding,
                                                                     const int64_t info_B_num_bits,
                                                                     const int64_t info_B_mask_int64,
                                                                     const bool use_uniq_cache_locations,
                                                                     const bool use_homogeneous_placements,
                                                                     const Tensor& hash_indices,
                                                                     const at::Tensor& unique_ids,
                                                                     const at::Tensor& unique_offsets,
                                                                     const at::Tensor& unique_inverse,
                                                                     double learning_rate = 0,
                                                                     bool is_dynamic = false)
{
    const int64_t t_max_D = max_D.guard_int(__FILE__, __LINE__);

    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    int64_t totalEmbed = unique_ids.numel() == 0 ? dev_weights.size(0) : unique_ids.numel() * t_max_D;
    auto output = at::empty({totalEmbed}, dev_weights.options());

    int optim_type = static_cast<int>(OptimizerType::SGD);
    const auto _unused = at::Tensor();
    const int iter = 0;
    const float beta = 0;

    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_output, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, hash_size_cumsum, indices, offsets,
                 lxu_cache_locations, _unused, _unused, _unused, _unused, _unused, _unused, _unused, _unused,
                 hash_indices, unique_ids, unique_offsets, unique_inverse, t_max_D, total_hash_size_bits, pooling_mode,
                 BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
                 use_uniq_cache_locations, use_homogeneous_placements, optim_type, beta, learning_rate, beta, beta,
                 iter, is_dynamic, output, _unused, _unused, dev_weights);

    return at::Tensor();
}

}; // namespace fbgemm_npu_lookups

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_lookup_sgd_function("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor(a!) dev_weights, Tensor(b!) uvm_weights,"
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
          "    Tensor? hash_indices = None, "
          "    Tensor? unique_ids = None, "
          "    Tensor? unique_offsets = None, "
          "    Tensor? unique_inverse = None, "
          "    float learning_rate = 0, "
          "    bool is_dynamic = False,"
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
          "    Tensor? prev_iter_dev=None, "
          "    int iter=0, "
          "    bool apply_global_weight_decay=False, "
          "    float gwd_lower_bound=0 "
          ") -> Tensor");

    m.impl("split_embedding_codegen_lookup_sgd_function",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_sgd_function)));
    m.impl("split_embedding_codegen_lookup_sgd_function",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_sgd_function)));
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_backward_codegen_sgd_unweighted_exact_cuda("
          "    Tensor grad_output, "
          "    Tensor(a!) dev_weights, "
          "    Tensor(b!) uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    SymInt max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    int unused_, "
          "    int max_segment_length_per_warp, "
          "    bool stochastic_rounding, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          "    bool use_uniq_cache_locations, "
          "    bool use_homogeneous_placements, "
          "    Tensor hash_indices = None, "
          "    Tensor unique_ids = None, "
          "    Tensor unique_offsets = None, "
          "    Tensor unique_inverse = None, "
          "    float learning_rate = 0, "
		  "    bool is_dynamic = False "
		  
          ") -> Tensor");

    m.impl("split_embedding_backward_codegen_sgd_unweighted_exact_cuda",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_backward_codegen_sgd_unweighted_exact_npu)));
}