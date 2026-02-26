/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <torch/library.h>

#include <ATen/DeviceGuard.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

using tensor_list = std::vector<at::Tensor>;
using BucketizeResult = std::tuple<
    at::Tensor, at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>;
using namespace at;

void validate_supported_inputs(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const at::Tensor& block_sizes,
    const bool bucketize_pos,
    const bool sequence,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<at::Tensor>& batch_size_per_feature,
    const int64_t max_B,
    const c10::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool keep_orig_idx,
    const c10::optional<at::Tensor>& total_num_blocks)
{
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D.");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D.");
    TORCH_CHECK(block_sizes.dim() == 1, "block_sizes must be 1D.");
    TORCH_CHECK(block_sizes.numel() > 0, "block_sizes must be non-empty.");
    TORCH_CHECK(
        lengths.scalar_type() == indices.scalar_type(),
        "lengths and indices must share dtype.");
    TORCH_CHECK(
        block_sizes.scalar_type() == lengths.scalar_type(),
        "block_sizes dtype must match lengths.");
    if (weights.has_value()) {
        TORCH_CHECK(weights.value().dim() == 1, "weights must be 1D.");
        TORCH_CHECK(
            weights.value().numel() == indices.numel(),
            "weights must match indices length.");
        TORCH_CHECK(
            weights.value().scalar_type() == at::ScalarType::Float,
            "weights dtype must be float32.");
    }

    if (batch_size_per_feature.has_value()) {
        TORCH_CHECK(batch_size_per_feature.value().dim() == 1, "batch_size_per_feature must be 1D.");
        TORCH_CHECK(
            batch_size_per_feature.value().scalar_type() == lengths.scalar_type(),
            "batch_size_per_feature dtype must match lengths.");
        TORCH_CHECK(
            batch_size_per_feature.value().numel() == block_sizes.numel(),
            "batch_size_per_feature length must match block_sizes length.");
        TORCH_CHECK(
            max_B > 0,
            "max_B must be positive when batch_size_per_feature is provided.");
    } else {
        TORCH_CHECK(
            lengths.numel() % block_sizes.numel() == 0,
            "lengths.size(0) must be divisible by block_sizes.size(0) when batch_size_per_feature is absent.");
    }

    if (total_num_blocks.has_value()) {
        TORCH_CHECK(total_num_blocks.value().dim() == 1, "total_num_blocks must be 1D.");
        TORCH_CHECK(total_num_blocks.value().scalar_type() == indices.scalar_type(),
            "total_num_blocks dtype must match indices.");
        TORCH_CHECK(total_num_blocks.value().numel() == block_sizes.numel(),
            "total_num_blocks must match block_sizes length.");
    }
}

BucketizeResult block_bucketize_sparse_features_npu(
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const bool bucketize_pos,
    const bool sequence,
    const at::Tensor& block_sizes,
    const int64_t my_size,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<at::Tensor>& batch_size_per_feature,
    const int64_t max_B,
    const c10::optional<std::vector<at::Tensor>>& block_bucketize_pos,
    const bool keep_orig_idx,
    const c10::optional<at::Tensor>& total_num_blocks)
{
    TORCH_CHECK(my_size > 0, "my_size must be positive.");
    if (lengths.scalar_type() == at::kInt) {
        TORCH_CHECK(
            my_size <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
            "my_size must fit into uint32 for int32 tensors.");
    }

    std::vector<at::Tensor> tensors = {lengths, indices, block_sizes};
    std::vector<std::string> names = {"lengths", "indices", "block_sizes"};
    if (weights.has_value()) {
        tensors.push_back(weights.value());
        names.push_back("weights");
    }
    check_tensor_npu_device(tensors, names);
    validate_supported_inputs(
        lengths,
        indices,
        block_sizes,
        bucketize_pos,
        sequence,
        weights,
        batch_size_per_feature,
        max_B,
        block_bucketize_pos,
        keep_orig_idx,
        total_num_blocks);

    const at::OptionalDeviceGuard guard(device_of(lengths));

    auto lengths_contig = lengths.contiguous();
    auto indices_contig = indices.contiguous();
    auto block_sizes_contig = block_sizes.contiguous();

    const auto lengths_size = lengths_contig.numel();
    const auto new_lengths_size = lengths_size * my_size;
    auto new_lengths =
        at::empty({new_lengths_size}, lengths_contig.options());
    auto new_indices =
        at::empty({indices_contig.numel()}, indices_contig.options());

    auto empty_tensor = at::Tensor();

    // Prepare input tensors
    auto weight_contig = weights.has_value() ?
        weights.value().contiguous() : empty_tensor;
    auto batch_size_per_feature_contig = batch_size_per_feature.has_value() ?
        batch_size_per_feature.value().contiguous() : empty_tensor;
    auto total_num_blocks_contig = total_num_blocks.has_value() ?
        total_num_blocks.value().contiguous() : empty_tensor;
    tensor_list block_bucketize_pos_vector = block_bucketize_pos.has_value() ?
        block_bucketize_pos.value() : tensor_list{at::empty({0}, indices_contig.options())};
    at::TensorList block_bucketize_pos_list_tensor = block_bucketize_pos_vector;

    // Prepare output tensors
    auto new_weights_tensor = weights.has_value() ?
        at::empty({weight_contig.numel()}, weight_contig.options()) : empty_tensor;
    auto new_pos_tensor = bucketize_pos ?
        at::empty({indices_contig.numel()}, indices_contig.options()) : empty_tensor;
    auto unbucketize_permute_tensor = sequence ?
        at::empty({indices_contig.numel()}, indices_contig.options()) : empty_tensor;

    EXEC_NPU_CMD(
        aclnnBlockBucketizeSparseFeatures,
        /* input - Tensor */
        lengths_contig,
        indices_contig,
        block_sizes_contig,
        weight_contig,
        batch_size_per_feature_contig,
        total_num_blocks_contig,
        block_bucketize_pos_list_tensor,
        /* input - attr */
        my_size,
        bucketize_pos,
        sequence,
        keep_orig_idx,
        max_B,
        /* output */
        new_lengths,
        new_indices,
        new_weights_tensor,
        new_pos_tensor,
        unbucketize_permute_tensor);

    c10::optional<at::Tensor> new_weights_opt = weights.has_value() ?
        c10::optional<at::Tensor>(new_weights_tensor) : c10::nullopt;
    c10::optional<at::Tensor> new_pos_opt = bucketize_pos ?
        c10::optional<at::Tensor>(new_pos_tensor) : c10::nullopt;
    c10::optional<at::Tensor> unbucketize_output = sequence ?
        c10::optional<at::Tensor>(unbucketize_permute_tensor) : c10::nullopt;

    return {
        new_lengths,
        new_indices,
        new_weights_opt,
        new_pos_opt,
        unbucketize_output};
}

TORCH_LIBRARY_FRAGMENT(mxrec, m) {
    m.def(
        "block_bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, "
        "Tensor block_sizes, SymInt my_size, Tensor? weights=None, Tensor? batch_size_per_feature=None, "
        "SymInt max_B= -1, Tensor[]? block_bucketize_pos=None, bool keep_orig_idx=False, "
        "Tensor? total_num_blocks=None) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m) {
    m.impl(
        "block_bucketize_sparse_features",
        &block_bucketize_sparse_features_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) {
    m.impl(
        "block_bucketize_sparse_features",
        &block_bucketize_sparse_features_npu);
}
