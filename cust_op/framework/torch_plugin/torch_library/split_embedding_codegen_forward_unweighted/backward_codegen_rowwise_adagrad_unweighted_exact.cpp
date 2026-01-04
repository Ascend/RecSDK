/**
 * @file backward_codegen_rowwise_adagrad_unweighted_exact.cpp
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

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;
using namespace optim_param_idx_fbgemm_120;

namespace fbgemm_npu_lookups {

// Forward declaration of the CUDA function
Tensor split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_cuda(
    const Tensor& grad_output, const Tensor& dev_weights, const Tensor& uvm_weights, const Tensor& lxu_cache_weights,
    const Tensor& weights_placements, const Tensor& weights_offsets, const Tensor& D_offsets, const c10::SymInt max_D,
    const Tensor& hash_size_cumsum, const int64_t total_hash_size_bits, const Tensor& indices, const Tensor& offsets,
    const int64_t pooling_mode, const Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets,
    const tensor_list& grad_accumulate, const Tensor& grad_accumulate_offsets, const Tensor& hash_indices,
    const Tensor& unique_ids, const Tensor& unique_offsets, const Tensor& unique_inverse,
    const Tensor& offset_per_key, const Tensor& table_grad_accumulate_offsets, double eps = 0, double learning_rate = 0,
    bool use_optimize = true);

class SplitLookupRowwiseAdagrad : public torch::autograd::Function<SplitLookupRowwiseAdagrad> {
public:
    static constexpr bool isTraceable = true;

    static Tensor forward(
        torch::autograd::AutogradContext* ctx, const Tensor& placeholder_autograd_tensor, const int64_t output_dtype,
        const Tensor& dev_weights, const Tensor& uvm_weights, const Tensor& lxu_cache_weights,
        const Tensor& weights_placements, const Tensor& weights_offsets, const Tensor& D_offsets,
        const c10::SymInt total_D, const c10::SymInt max_D, const Tensor& hash_size_cumsum,
        const c10::optional<Tensor>& rows_per_table, const int64_t total_hash_size_bits, const Tensor& indices,
        const c10::optional<Tensor>& hash_indices, const c10::optional<at::Tensor>& unique_ids,
        const c10::optional<at::Tensor>& unique_offsets, const c10::optional<at::Tensor>& unique_inverse,
        const c10::optional<at::Tensor>& table_grad_accumulate_offsets, const Tensor& offsets,
        const int64_t pooling_mode, const std::optional<Tensor>& indice_weights,
        const std::optional<Tensor>& feature_requires_grad, const Tensor& lxu_cache_locations,
        std::optional<Tensor> uvm_cache_stats, const bool gradient_clipping, const double max_gradient,
        const bool stochastic_rounding, const bool is_experimental, const bool use_uniq_cache_locations_bwd,
        const bool use_homogeneous_placements, Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements,
        Tensor momentum1_offsets, const std::optional<tensor_list>& grad_accumulate = std::nullopt,
        const std::optional<at::Tensor>& grad_accumulate_offsets = std::nullopt, double eps = 0,
        double learning_rate = 0, bool use_optimize = true)
    {
        check_tensor_non_empty(weights_offsets, "weights_offsets");
        check_tensor_non_empty(offsets, "offsets");
        check_tensor_non_empty(D_offsets, "D_offsets");
        const auto T = weights_offsets.size(0);
        TORCH_CHECK(T > 0, "Weights_offsets size must be greater than 0.");
        const auto max_B_ = offsets.size(0) / T;
        // NOTE: The `local_uvm_cache_stats` variable held by the nn.Module has dtype int32_t
        const auto uvm_cache_stats_ = uvm_cache_stats.value_or(at::empty({0}, uvm_weights.options().dtype(at::kInt)));

        auto info_B_num_bits = max_B_;
        auto info_B_mask = T;
        
        // EC查表，计算每张表的indices个数
        at::Tensor offset_per_key = compute_offset_per_key(offsets, weights_offsets, D_offsets);

        std::vector<at::Tensor> saved_tensors;
        saved_tensors.push_back(dev_weights);
        saved_tensors.push_back(uvm_weights);
        saved_tensors.push_back(lxu_cache_weights);
        saved_tensors.push_back(weights_placements);
        saved_tensors.push_back(weights_offsets);
        saved_tensors.push_back(D_offsets);
        saved_tensors.push_back(hash_size_cumsum);
        saved_tensors.push_back(indices);
        saved_tensors.push_back(offsets);
        saved_tensors.push_back(indice_weights.has_value() ? indice_weights.value() : at::Tensor());
        saved_tensors.push_back(feature_requires_grad.has_value() ? feature_requires_grad.value() : at::Tensor());
        saved_tensors.push_back(lxu_cache_locations);
        saved_tensors.push_back(momentum1_dev);
        saved_tensors.push_back(momentum1_uvm);
        saved_tensors.push_back(momentum1_placements);
        saved_tensors.push_back(momentum1_offsets);
        saved_tensors.push_back(hash_indices.has_value() ? hash_indices.value() : at::Tensor());
        saved_tensors.push_back(unique_ids.has_value() ? unique_ids.value() : at::Tensor());
        saved_tensors.push_back(unique_offsets.has_value() ? unique_offsets.value() : at::Tensor());
        saved_tensors.push_back(unique_inverse.has_value() ? unique_inverse.value() : at::Tensor());
        saved_tensors.push_back(offset_per_key);
        saved_tensors.push_back(
            table_grad_accumulate_offsets.has_value() ? table_grad_accumulate_offsets.value() : at::Tensor());
        saved_tensors.push_back(
            grad_accumulate_offsets.has_value() ? grad_accumulate_offsets.value() : at::Tensor());

        if (grad_accumulate.has_value()) {
            saved_tensors.insert(saved_tensors.end(), grad_accumulate.value().begin(), grad_accumulate.value().end());
        }
        
        ctx->save_for_backward(saved_tensors);
        ctx->saved_data["max_D"] = max_D;
        ctx->saved_data["pooling_mode"] = pooling_mode;
        ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
        ctx->saved_data["gradient_clipping"] = gradient_clipping;
        ctx->saved_data["max_gradient"] = max_gradient;
        ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
        ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
        ctx->saved_data["info_B_mask"] = info_B_mask;
        ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
        ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;
        ctx->saved_data["eps"] = eps;
        ctx->saved_data["learning_rate"] = learning_rate;
        ctx->saved_data["use_optimize"] = use_optimize;
        const auto& flatten_dev_weights = dev_weights;
        // not surport  indice_weights
        TORCH_CHECK(!indice_weights, "indice_weights is unsupported.");
        static auto embedding_codegen_forward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_unweighted_cuda", "")
                .typed<decltype(split_embedding_codegen_forward_unweighted_cuda)>();

        return {embedding_codegen_forward_op.call(
            flatten_dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            total_D, max_D, indices, offsets, pooling_mode, lxu_cache_locations, uvm_cache_stats_, output_dtype,
            is_experimental, hash_indices.value_or(Tensor()), offset_per_key, rows_per_table.value_or(Tensor()))};
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
        auto momentum1_dev = *savedItr++;
        auto momentum1_uvm = *savedItr++;
        auto momentum1_placements = *savedItr++;
        auto momentum1_offsets = *savedItr++;
        auto hash_indices = *savedItr++;
        auto unique_ids = *savedItr++;
        auto unique_offsets = *savedItr++;
        auto unique_inverse = *savedItr++;
        auto offset_per_key = *savedItr++;
        auto table_grad_accumulate_offsets = *savedItr++;
        auto grad_accumulate_offsets = *savedItr++;
        tensor_list grad_accumulate(savedItr, std::end(saved));
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
        auto eps = ctx->saved_data["eps"].toDouble();
        auto learning_rate = ctx->saved_data["learning_rate"].toDouble();
        auto use_optimize = ctx->saved_data["use_optimize"].toBool();

        TORCH_CHECK_EQ(grad_outputs.size(), 1);

        constexpr int32_t BT_block_size = 32;
        constexpr int32_t max_segment_length_per_warp = 32;

        using torch::autograd::Variable;
        auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
        static auto embedding_codegen_unweighted_backward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_cuda", "")
                .typed<decltype(split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_cuda)>();

        constexpr int64_t PAD_SIZE = 16;
        auto padded_momentum = at::empty({momentum1_dev.numel(), PAD_SIZE}, momentum1_dev.options());
        padded_momentum.slice(1, 0, 1).squeeze(1).copy_(momentum1_dev);

        const auto grad_dev_weights = embedding_codegen_unweighted_backward_op.call(
            grad_output, dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, pooling_mode, lxu_cache_locations,
            BT_block_size, max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
            use_uniq_cache_locations_bwd, use_homogeneous_placements, padded_momentum, momentum1_uvm,
            momentum1_placements, momentum1_offsets, grad_accumulate, grad_accumulate_offsets, hash_indices, unique_ids,
            unique_offsets, unique_inverse, offset_per_key, table_grad_accumulate_offsets, eps, learning_rate,
            use_optimize);
        momentum1_dev.copy_(padded_momentum.slice(1, 0, 1).squeeze(1));

        // Create a variable_list with correct size (42 elements)
        variable_list result(42);
        result[0] = Tensor();           // placeholder autograd tensor
        result[1] = Variable();         // output_dtype
        result[2] = grad_dev_weights;   // dev_weights
        result[3] = Variable();         // uvm_weights
        result[4] = Variable();         // lxu_cache_weights
        result[5] = Variable();         // weights_placements
        result[6] = Variable();         // weights_offsets
        result[7] = Variable();         // D_offsets
        result[8] = Variable();         // total_D
        result[9] = Variable();         // max_D
        result[10] = Variable();        // hash_size_cumsum
        result[11] = Variable();        // rows_per_table
        result[12] = Variable();        // total_hash_size_bits
        result[13] = Variable();        // indices
        result[14] = Variable();        // offsets
        result[15] = Variable();        // pooling_mode
        result[16] = Variable();        // indice_weights
        result[17] = Variable();        // feature_requires_grad
        result[18] = Variable();        // lxu_cache_locations
        result[19] = Variable();        // uvm_cache_stats
        result[20] = Variable();        // gradient_clipping
        result[21] = Variable();        // max_gradient
        result[22] = Variable();        // stochastic_rounding
        result[23] = Variable();        // is_experimental
        result[24] = Variable();        // use_uniq_cache_locations_bwd
        result[25] = Variable();        // use_homogeneous_placements
        result[26] = Variable();        // momentum1_dev
        result[27] = Variable();        // momentum1_uvm
        result[28] = Variable();        // momentum1_placements
        result[29] = Variable();        // momentum1_offsets
        result[30] = Variable();        // grad_accumulate
        result[31] = Variable();        // hash_indices
        result[32] = Variable();        // unique_ids
        result[33] = Variable();        // unique_offsets
        result[34] = Variable();        // unique_inverse
        result[35] = Variable();        // offset_per_key
        result[36] = Variable();        // table_grad_accumulate_offsets
        result[37] = Variable();        // eps
        result[38] = Variable();        // learning_rate
        result[39] = Variable();        // grad_accumulate_offsets
        result[40] = Variable();        // use_optimize
        result[41] = Variable();        // unused/placeholder
        
        return result;
    }
};

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
    const std::optional<TensorList>& grad_accumulate,
    const std::optional<at::Tensor>& grad_accumulate_offsets,
    const std::optional<Tensor>& hash_indices,
    const std::optional<Tensor>& unique_ids,
    const std::optional<Tensor>& unique_offsets,
    const std::optional<Tensor>& unique_inverse,
    const std::optional<Tensor>& table_grad_accumulate_offsets,
    const std::optional<Tensor>& rows_per_table)
{
    std::optional<std::vector<at::Tensor>> grad_accumulate_vector = std::nullopt;
    if (grad_accumulate.has_value()) {
        grad_accumulate_vector =
            std::vector<at::Tensor>(grad_accumulate.value().begin(), grad_accumulate.value().end());
    }

    return SplitLookupRowwiseAdagrad::apply(
        placeholder_autograd_tensor,
        output_dtype,
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        hash_size_cumsum,
        rows_per_table,
        total_hash_size_bits,
        indices,
        hash_indices,
        unique_ids,
        unique_offsets,
        unique_inverse,
        table_grad_accumulate_offsets,
        offsets,
        pooling_mode,
        indice_weights,
        feature_requires_grad,
        lxu_cache_locations,
        uvm_cache_stats,
        gradient_clipping,
        max_gradient,
        stochastic_rounding,
        is_experimental_tbe,
        use_uniq_cache_locations_bwd,
        use_homogeneous_placements,
        momentum1_dev,
        momentum1_uvm,
        momentum1_placements,
        momentum1_offsets,
        grad_accumulate_vector,
        grad_accumulate_offsets,
        eps,
        learning_rate,
        true);
}

at::Tensor split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_npu(
    const Tensor& grad_output,
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
    Tensor momentum1_dev,
    Tensor momentum1_uvm,
    Tensor momentum1_placements,
    Tensor momentum1_offsets,
    const tensor_list& grad_accumulate,
    const Tensor& grad_accumulate_offsets,
    const Tensor& hash_indices,
    const at::Tensor& unique_ids,
    const at::Tensor& unique_offsets,
    const at::Tensor& unique_inverse,
    const at::Tensor& offset_per_key,
    const at::Tensor& table_grad_accumulate_offsets,
    double eps,
    double learning_rate,
    bool use_optimize)
{
    const int64_t t_max_D = max_D.guard_int(__FILE__, __LINE__);

    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    const auto _unused = Tensor();

    validate_backward_data_inputs(grad_output, dev_weights, weights_offsets, D_offsets, hash_size_cumsum, indices,
                                  offsets, momentum1_dev, _unused, hash_indices, unique_ids, unique_offsets,
                                  unique_inverse, offset_per_key, ROWWISE_ADAGRAD_OPTIM_NUM);

    int64_t totalEmbed = unique_ids.numel() == 0 ? dev_weights.size(0) : unique_ids.numel() * t_max_D;
    auto output = at::empty({totalEmbed}, dev_weights.options().dtype(at::kFloat));

    int optim_type = static_cast<int>(OptimizerType::ROWWISE_ADAGRAD);
    
    double beta = 0;
    int64_t iter = 0;

    const auto grad_output_conti = grad_output.contiguous();

    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_output_conti, dev_weights, uvm_weights,
                 lxu_cache_weights, weights_placements, weights_offsets, D_offsets, hash_size_cumsum, indices, offsets,
                 lxu_cache_locations, momentum1_dev, momentum1_uvm, momentum1_placements, momentum1_offsets,
                 _unused, _unused, _unused, _unused,
                 hash_indices, unique_ids, unique_offsets, unique_inverse, offset_per_key, t_max_D,
                 total_hash_size_bits, pooling_mode, BT_block_size, max_segment_length_per_warp, stochastic_rounding,
                 info_B_num_bits, info_B_mask_int64, use_uniq_cache_locations, use_homogeneous_placements, optim_type,
                 eps, learning_rate, beta, beta, iter, use_optimize, output, momentum1_dev, _unused, dev_weights);

    if (!use_optimize) {
        Tensor unique_offsets_size = unique_offsets * t_max_D * output.element_size();
        Tensor grad_accumulate_offsets_size = grad_accumulate_offsets * output.element_size();
        void* new_tensor_device_ptr = output.data_ptr();
        copy_gm_to_gm(new_tensor_device_ptr, grad_accumulate, unique_offsets_size, grad_accumulate_offsets_size);
    }
    return at::Tensor();
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
    const c10::SymInt max_B,
    const c10::SymInt max_B_feature_rank,
    const c10::SymInt vbe_output_size)
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

    // unpacking from momentum1
    check_param_len(momentum1.size(), MOMENTUM1_SIZE, "momentum1");
    auto& momentum1_dev = momentum1[MOMENTUM1_DEV_INDEX];
    auto& momentum1_uvm = momentum1[MOMENTUM1_UVM_INDEX];
    auto& momentum1_placements = momentum1[MOMENTUM1_PLACEMENTS_INDEX];
    auto& momentum1_offsets = momentum1[MOMENTUM1_OFFSETS_INDEX];

    double learning_rate = learning_rate_tensor.item().toDouble();
    double eps = optim_float[EPS_INDEX];

    return SplitLookupRowwiseAdagrad::apply(
        placeholder_autograd_tensor,
        output_dtype,
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        hash_size_cumsum,
        std::nullopt, // rows_per_table
        total_hash_size_bits,
        indices,
        std::nullopt, // hash_indices
        std::nullopt, // unique_ids
        std::nullopt, // unique_offsets
        std::nullopt, // unique_inverse
        std::nullopt, // table_grad_accumulate_offsets
        offsets,
        pooling_mode,
        indice_weights,
        feature_requires_grad,
        lxu_cache_locations,
        uvm_cache_stats,
        false, // gradient_clipping
        0.0,   // max_gradient
        true,  // stochastic_rounding
        is_experimental_tbe,
        false, // use_uniq_cache_locations_bwd
        true,  // use_homogeneous_placements
        momentum1_dev,
        momentum1_uvm,
        momentum1_placements,
        momentum1_offsets,
        std::nullopt, // grad_accumulate
        std::nullopt, // grad_accumulate_offsets
        eps,
        learning_rate,
        true);
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
          "    Tensor? prev_iter_dev=None, "
          "    int iter=0, "
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

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_cuda("
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
          "    Tensor momentum1_dev, "
          "    Tensor momentum1_uvm, "
          "    Tensor momentum1_placements, "
          "    Tensor momentum1_offsets, "
          "    Tensor[] grad_accumulate, "
          "    Tensor grad_accumulate_offsets, "
          "    Tensor hash_indices = None, "
          "    Tensor unique_ids = None, "
          "    Tensor unique_offsets = None, "
          "    Tensor unique_inverse = None, "
          "    Tensor offset_per_key = None, "
          "    Tensor table_grad_accumulate_offsets = None, "
          "    float eps = 0, "
          "    float learning_rate = 0, "
          "    bool use_optimize = True"
          ") -> Tensor");

    m.impl(
        "split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_cuda",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                        TORCH_FN(fbgemm_npu_lookups::
                                 split_embedding_backward_codegen_rowwise_adagrad_unweighted_exact_npu)));
}

// dispatch FBGEMM 1.2.0 interface to NPU op
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.impl("split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
           torch::dispatch(c10::DispatchKey::Autograd,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_rowwise_adagrad_function_pt2)));

    m.impl("split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_rowwise_adagrad_function_pt2)));
}