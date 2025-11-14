/**
 * @file backward_codegen_adagrad_unweighted_exact.cpp
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
#include "split_embedding_codegen_common_utils.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using Tensor = at::Tensor;
using namespace at;

namespace fbgemm_npu_lookups {
Tensor split_embedding_backward_codegen_adagrad_unweighted_exact_cuda_grad_aggregation(
    const Tensor& grad_output, const Tensor& dev_weights, const Tensor& uvm_weights, const Tensor& lxu_cache_weights,
    const Tensor& weights_placements, const Tensor& weights_offsets, const Tensor& D_offsets, const c10::SymInt max_D,
    const Tensor& hash_size_cumsum, const int64_t total_hash_size_bits, const Tensor& indices, const Tensor& offsets,
    const Tensor& indices_multi_step, const Tensor& offsets_multi_step,
    const int64_t pooling_mode, const Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets,
    const tensor_list& grad_accumulate, const Tensor& grad_accumulate_offsets, const Tensor& hash_indices,
    const Tensor& unique_ids, const Tensor& unique_offsets, const Tensor& unique_inverse,
    const Tensor& offset_per_key, const Tensor& table_grad_accumulate_offsets,
    const Tensor& table_offsets_multi, const Tensor& unique_multi_step,
    const Tensor& unique_offset_multi_step, const Tensor& unique_inverse_multi_step,
    double eps = 0, double learning_rate = 0, bool use_optimize = true);

class SplitLookupAdagrad_grad_aggregation : public torch::autograd::Function<SplitLookupAdagrad_grad_aggregation> {
public:
    static constexpr bool isTraceable = true;

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx, const Tensor& placeholder_autograd_tensor, const int64_t output_dtype,
        const Tensor& dev_weights, const Tensor& uvm_weights, const Tensor& lxu_cache_weights,
        const Tensor& weights_placements, const Tensor& weights_offsets, const Tensor& D_offsets,
        const c10::SymInt total_D, const c10::SymInt max_D, const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits, const Tensor& indices, const c10::optional<Tensor>& hash_indices,
        const c10::optional<at::Tensor>& unique_ids, const c10::optional<at::Tensor>& unique_offsets,
        const c10::optional<at::Tensor>& unique_inverse, const c10::optional<at::Tensor>& table_grad_accumulate_offsets,
        const c10::optional<at::Tensor>& table_offsets_multi,
        const Tensor& offsets,
        const Tensor& indices_multi_step,
        const Tensor& offsets_multi_step,
        const c10::optional<at::Tensor>& unique_multi_step,
        const c10::optional<at::Tensor>& unique_offset_multi_step,
        const c10::optional<at::Tensor>& unique_inverse_multi_step,
        const int64_t pooling_mode, const std::optional<Tensor>& indice_weights,
        const std::optional<Tensor>& feature_requires_grad, const Tensor& lxu_cache_locations,
        std::optional<Tensor> uvm_cache_stats, const bool gradient_clipping, const double max_gradient,
        const bool stochastic_rounding, const bool is_experimental, const bool use_uniq_cache_locations_bwd,
        const bool use_homogeneous_placements, Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements,
        Tensor momentum1_offsets, const std::optional<tensor_list>& grad_accumulate = std::nullopt,
        const std::optional<at::Tensor>& grad_accumulate_offsets =  std::nullopt,
        double eps = 0, double learning_rate = 0, bool use_optimize = true)
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
        int64_t batchs = (offsets.numel() - 1) / weights_offsets.numel();
        at::Tensor table_offsets = torch::arange(D_offsets.size(0), offsets.device()) * batchs;
        at::Tensor offset_per_key = offsets.index_select(0, table_offsets.to(at::kLong));

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
            table_grad_accumulate_offsets.has_value()
            ? table_grad_accumulate_offsets.value()
            : at::Tensor());
        saved_tensors.push_back(table_offsets_multi.has_value() ? table_offsets_multi.value() : at::Tensor());

        saved_tensors.push_back(indices_multi_step);
        saved_tensors.push_back(offsets_multi_step);
        saved_tensors.push_back(unique_multi_step.has_value() ? unique_multi_step.value() : at::Tensor());
        saved_tensors.push_back(unique_offset_multi_step.has_value() ? unique_offset_multi_step.value() : at::Tensor());
        saved_tensors.push_back(unique_inverse_multi_step.has_value()
                                ? unique_inverse_multi_step.value()
                                : at::Tensor());
        saved_tensors.push_back(grad_accumulate_offsets.has_value() ? grad_accumulate_offsets.value() : at::Tensor());
        if (grad_accumulate.has_value()) {
            saved_tensors.insert(
                saved_tensors.end(),
                grad_accumulate.value().begin(),
                grad_accumulate.value().end()
                );
        }
        ctx->save_for_backward(saved_tensors);
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
            is_experimental, hash_indices.value_or(Tensor()), offset_per_key)};
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
        auto table_offsets_multi = *savedItr++;
        auto indices_multi_step = *savedItr++;
        auto offsets_multi_step = *savedItr++;
        auto unique_multi_step = *savedItr++;
        auto unique_offset_multi_step = *savedItr++;
        auto unique_inverse_multi_step = *savedItr++;
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
                .findSchemaOrThrow(
                "fbgemm::split_embedding_backward_codegen_adagrad_unweighted_exact_cuda_grad_aggregation", "")
                .typed<decltype(split_embedding_backward_codegen_adagrad_unweighted_exact_cuda_grad_aggregation)>();

        const auto grad_dev_weights = embedding_codegen_unweighted_backward_op.call(
            grad_output, dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, D_offsets,
            max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, indices_multi_step,
            offsets_multi_step, pooling_mode, lxu_cache_locations, BT_block_size, max_segment_length_per_warp,
            stochastic_rounding, info_B_num_bits, info_B_mask_int64, use_uniq_cache_locations_bwd,
            use_homogeneous_placements, momentum1_dev, momentum1_uvm, momentum1_placements, momentum1_offsets,
            grad_accumulate, grad_accumulate_offsets, hash_indices, unique_ids, unique_offsets, unique_inverse,
            offset_per_key, table_grad_accumulate_offsets, table_offsets_multi, unique_multi_step,
            unique_offset_multi_step, unique_inverse_multi_step, eps, learning_rate, use_optimize);
        return {
            Tensor(),          // placeholder autograd tensor
            Variable(),        // output_dtype
            grad_dev_weights,  // dev_weights
            Variable(),        // uvm_weights
            Variable(),        // lxu_cache_weights
            Variable(),        // weights_placements
            Variable(),        // weights_offsets
            Variable(),        // D_offsets
            Variable(),        // total_D
            Variable(),        // max_D
            Variable(),        // hash_size_cumsum
            Variable(),        // total_hash_size_bits
            Variable(),        // indices
            Variable(),        // offsets
            Variable(),        // pooling_mode
            Variable(),        // indice_weights
            Variable(),        // feature_requires_grad
            Variable(),        // lxu_cache_locations
            Variable(),        // uvm_cache_stats
            Variable(),        // gradient_clipping
            Variable(),        // max_gradient
            Variable(),        // stochastic_rounding
            Variable(),        // is_experimental
            Variable(),        // use_uniq_cache_locations_bwd
            Variable(),        // use_homogeneous_placements
            Variable(),        // momentum1_dev
            Variable(),        // momentum1_uvm
            Variable(),        // momentum1_placements
            Variable(),        // momentum1_offsets
            Variable(),        // grad_accumulate
            Variable(),        // hash_indices
            Variable(),        // unique_ids
            Variable(),        // unique_offsets
            Variable(),        // unique_inverse
            Variable(),        // offset_per_key
            Variable(),        // table_grad_accumulate_offsets,
            Variable(),        // table_offsets_multi
            Variable(),        // eps
            Variable(),        // learning_rate
            Variable(),        // grad_accumulate_offsets
            Variable(),        // use_optimize
            Variable(),        // indices_multi_step
            Variable(),        // offsets_multi_step
            Variable(),        // unique_multi_step
            Variable(),        // unique_offset_multi_step
            Variable(),        // unique_inverse_multi_step
        };
    }
};

///@ingroup embedding-cuda
Tensor split_embedding_codegen_lookup_adagrad_function_grad_aggregation(
    const Tensor& placeholder_autograd_tensor, const Tensor& dev_weights, const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights, const Tensor& weights_placements, const Tensor& weights_offsets,
    const Tensor& D_offsets, const c10::SymInt total_D, const c10::SymInt max_D, const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits, const Tensor& indices, const Tensor& offsets,
    const Tensor& indices_multi_step,
    const Tensor& offsets_multi_step,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights, const std::optional<Tensor>& feature_requires_grad,
    const Tensor& lxu_cache_locations, const bool gradient_clipping, const double max_gradient,
    const bool stochastic_rounding, Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements,
    Tensor momentum1_offsets,
    const c10::optional<tensor_list>& grad_accumulate = std::nullopt,
    const c10::optional<at::Tensor>& grad_accumulate_offsets = std::nullopt,
    const c10::optional<Tensor>& hash_indices = c10::optional<Tensor>(),
    const c10::optional<at::Tensor>& unique_ids = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_inverse = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& table_grad_accumulate_offsets = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& table_offsets_multi = c10::optional<at::Tensor>(),
    const c10::optional<at::Tensor>& unique_multi_step = c10::optional<Tensor>(),
    const c10::optional<at::Tensor>& unique_offset_multi_step = c10::optional<Tensor>(),
    const c10::optional<at::Tensor>& unique_inverse_multi_step = c10::optional<Tensor>(),
    double eps = 0,
    double learning_rate = 0, const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const std::optional<Tensor>& B_offsets = c10::nullopt,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank = c10::nullopt,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature = c10::nullopt, const c10::SymInt max_B = -1,
    const c10::SymInt max_B_feature_rank = -1, const c10::SymInt vbe_output_size = -1,
    const bool is_experimental_tbe = false,  // formerly named is_experimental
    const bool use_uniq_cache_locations_bwd = false, const bool use_homogeneous_placements = false,
    const std::optional<Tensor>& uvm_cache_stats = c10::nullopt,
    const std::optional<Tensor>& prev_iter_dev = c10::nullopt, const int64_t iter = 0,
    const bool apply_global_weight_decay = false, const double gwd_lower_bound = 0,
    bool use_optimize = true)
{
    // Set to experimental if either the feature is enabled in JK, or the user specifies to use TBEv2
    const auto is_experimental = is_experimental_tbe;

    return SplitLookupAdagrad_grad_aggregation::apply(
        placeholder_autograd_tensor, output_dtype, dev_weights, uvm_weights, lxu_cache_weights, weights_placements,
        weights_offsets, D_offsets, total_D, max_D, hash_size_cumsum, total_hash_size_bits, indices, hash_indices,
        unique_ids, unique_offsets, unique_inverse, table_grad_accumulate_offsets,
        table_offsets_multi,
        offsets,
        indices_multi_step,
        offsets_multi_step,
        unique_multi_step,
        unique_offset_multi_step,
        unique_inverse_multi_step,
        pooling_mode, indice_weights,
        feature_requires_grad, lxu_cache_locations, uvm_cache_stats, gradient_clipping, max_gradient,
        stochastic_rounding, is_experimental, use_uniq_cache_locations_bwd, use_homogeneous_placements, momentum1_dev,
        momentum1_uvm, momentum1_placements, momentum1_offsets, grad_accumulate, grad_accumulate_offsets, eps,
        learning_rate, use_optimize)[0];
}

at::Tensor split_embedding_backward_codegen_adagrad_unweighted_exact_npu_grad_aggregation(
    const Tensor& grad_output, const Tensor& dev_weights, const Tensor& uvm_weights, const Tensor& lxu_cache_weights,
    const Tensor& weights_placements, const Tensor& weights_offsets, const Tensor& D_offsets, const c10::SymInt max_D,
    const Tensor& hash_size_cumsum, const int64_t total_hash_size_bits, const Tensor& indices, const Tensor& offsets,
    const Tensor& indices_multi_step,
    const Tensor& offsets_multi_step,
    const int64_t pooling_mode, const Tensor& lxu_cache_locations, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, const bool stochastic_rounding, const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64, const bool use_uniq_cache_locations, const bool use_homogeneous_placements,
    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets,
    const tensor_list& grad_accumulate, const Tensor& grad_accumulate_offsets, const Tensor& hash_indices,
    const at::Tensor& unique_ids, const at::Tensor& unique_offsets, const at::Tensor& unique_inverse,
    const at::Tensor& offset_per_key, const at::Tensor& table_grad_accumulate_offsets,
    const at::Tensor& table_offsets_multi,
    const at::Tensor& unique_multi_step,
    const at::Tensor& unique_offset_multi_step,
    const at::Tensor& unique_inverse_multi_step,
    double eps = 0, double learning_rate = 0,
    bool use_optimize = true)
{
    const int64_t t_max_D = max_D.guard_int(__FILE__, __LINE__);

    const at::OptionalDeviceGuard guard(device_of(dev_weights));
    const auto _unused = Tensor();

    validate_backward_data_inputs(grad_output, dev_weights, weights_offsets, D_offsets, hash_size_cumsum, indices,
                                  offsets, momentum1_dev, _unused, hash_indices, unique_ids, unique_offsets,
                                  unique_inverse, offset_per_key, ADAGRAD_OPTIM_NUM);

    // unique查表，则需要将output的形状设置为(unique_ids.numel() * t_max_D)
    int64_t totalEmbed = unique_ids.numel() == 0 ? dev_weights.size(0) : unique_ids.numel() * t_max_D;
    auto output = at::empty({totalEmbed}, dev_weights.options().dtype(at::kFloat));

    int optim_type = static_cast<int>(OptimizerType::ADAGRAD);
    
    double beta = 0;
    int64_t iter = 0;

    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_output, dev_weights, uvm_weights, lxu_cache_weights,
                 weights_placements, weights_offsets, D_offsets, hash_size_cumsum, indices, offsets,
                 lxu_cache_locations, momentum1_dev, momentum1_uvm, momentum1_placements, momentum1_offsets,
                 _unused, _unused, _unused, _unused,
                 hash_indices, unique_ids, unique_offsets, unique_inverse, offset_per_key, t_max_D,
                 total_hash_size_bits, pooling_mode, BT_block_size, max_segment_length_per_warp, stochastic_rounding,
                 info_B_num_bits, info_B_mask_int64, use_uniq_cache_locations, use_homogeneous_placements, optim_type,
                 eps, learning_rate, beta, beta, iter, use_optimize, output, momentum1_dev, _unused, dev_weights);

    Tensor unique_offsets_size = unique_offsets * t_max_D * output.element_size();
    Tensor grad_accumulate_offsets_size = grad_accumulate_offsets * output.element_size();

    void* new_tensor_device_ptr = output.data_ptr(); // 获取output的设备指针
    copy_gm_to_gm(
        new_tensor_device_ptr,
        grad_accumulate,
        unique_offsets_size,
        grad_accumulate_offsets_size); // 将数据从new_tensor_device_ptr拷贝至grad_accumulate

    std::vector<torch::Tensor> reshaped_tensors;
    Tensor unique_offset_slice = unique_offsets * t_max_D;
    for (size_t i = 0; i < grad_accumulate.size(); ++i) {
        auto slice_size = unique_offset_slice[i+1].item<int64_t>() -
                          unique_offset_slice[i].item<int64_t>() +
                          grad_accumulate_offsets[i].item<int64_t>();
        auto contig = grad_accumulate[i].slice(0, 0, slice_size).view({-1, t_max_D});
        reshaped_tensors.push_back(contig);
    }
    torch::Tensor grad_last_step = torch::cat(reshaped_tensors, 0);

    const int64_t t_max_D_last = max_D.guard_int(__FILE__, __LINE__);
    int64_t uniqueSize_last = static_cast<int64_t>(unique_multi_step.numel());
    int64_t totalEmbed_last = uniqueSize_last == 0 ? dev_weights.size(0) : uniqueSize_last * t_max_D_last;
    auto output_last = at::empty({totalEmbed_last}, dev_weights.options());
    int optim_type_last = static_cast<int>(OptimizerType::ADAGRAD);
    const auto _unused_last = Tensor();
    double beta_last = 0;
    int64_t iter_last = 0;

    bool use_optimize_last_step = true;

    EXEC_NPU_CMD(aclnnBackwardCodegenAdagradUnweightedExact, grad_last_step, dev_weights, uvm_weights,
                 lxu_cache_weights, weights_placements, weights_offsets, D_offsets, hash_size_cumsum,
                 indices_multi_step, offsets_multi_step, lxu_cache_locations, momentum1_dev, momentum1_uvm,
                 momentum1_placements, momentum1_offsets, _unused_last, _unused_last, _unused_last, _unused_last,
                 hash_indices, unique_multi_step, unique_offset_multi_step, unique_inverse_multi_step,
                 table_offsets_multi, t_max_D_last, total_hash_size_bits, pooling_mode, BT_block_size,
                 max_segment_length_per_warp, stochastic_rounding, info_B_num_bits, info_B_mask_int64,
                 use_uniq_cache_locations, use_homogeneous_placements, optim_type_last, eps, learning_rate, beta_last,
                 beta_last, iter_last, use_optimize_last_step, output_last, momentum1_dev, _unused_last, dev_weights);

    return at::Tensor();
}
};  // namespace fbgemm_npu_lookups

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_codegen_lookup_adagrad_function_grad_aggregation("
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
          "    Tensor indices_multi_step, "
          "    Tensor offsets_multi_step, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    Tensor lxu_cache_locations, "
          "    bool gradient_clipping, float max_gradient, "
          "    bool stochastic_rounding, "
          "    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, "
          "    Tensor[]? grad_accumulate = None, "
          "    Tensor? grad_accumulate_offsets = None, "
          "    Tensor? hash_indices = None, "
          "    Tensor? unique_ids = None, "
          "    Tensor? unique_offsets = None, "
          "    Tensor? unique_inverse = None, "
          "    Tensor? table_grad_accumulate_offsets = None, "
          "    Tensor? table_offsets_multi = None, "
          "    Tensor? unique_multi_step = None, "
          "    Tensor? unique_offset_multi_step = None, "
          "    Tensor? unique_inverse_multi_step = None, "
          "    float eps = 0, float learning_rate = 0, "
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
          "    bool use_optimize = True"
          ") -> Tensor");

    m.impl("split_embedding_codegen_lookup_adagrad_function_grad_aggregation",
           torch::dispatch(
                c10::DispatchKey::Autograd,
                TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_adagrad_function_grad_aggregation)));

    m.impl("split_embedding_codegen_lookup_adagrad_function_grad_aggregation",
           torch::dispatch(
                c10::DispatchKey::PrivateUse1,
                TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_lookup_adagrad_function_grad_aggregation)));
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("split_embedding_backward_codegen_adagrad_unweighted_exact_cuda_grad_aggregation("
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
          "    Tensor indices_multi_step, "
          "    Tensor offsets_multi_step, "
          "    int pooling_mode, "
          "    Tensor lxu_cache_locations, "
          "    int unused_, "
          "    int max_segment_length_per_warp, "
          "    bool stochastic_rounding, "
          "    int info_B_num_bits, "
          "    int info_B_mask_int64, "
          "    bool use_uniq_cache_locations, "
          "    bool use_homogeneous_placements, "
          "    Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, "
          "    Tensor momentum1_offsets, "
          "    Tensor[] grad_accumulate, "
          "    Tensor grad_accumulate_offsets, "
          "    Tensor hash_indices = None, "
          "    Tensor unique_ids = None, "
          "    Tensor unique_offsets = None, "
          "    Tensor unique_inverse = None, "
          "    Tensor offset_per_key = None, "
          "    Tensor table_grad_accumulate_offsets = None, "
          "    Tensor table_offsets_multi = None, "
          "    Tensor unique_multi_step = None, "
          "    Tensor unique_offset_multi_step = None, "
          "    Tensor unique_inverse_multi_step = None, "
          "    float eps = 0, float learning_rate = 0, "
          "    bool use_optimize = True"
          ") -> Tensor");

    m.impl(
        "split_embedding_backward_codegen_adagrad_unweighted_exact_cuda_grad_aggregation",
        torch::dispatch(
            c10::DispatchKey::PrivateUse1,
            TORCH_FN(
            fbgemm_npu_lookups::split_embedding_backward_codegen_adagrad_unweighted_exact_npu_grad_aggregation
            )
        ));
}
