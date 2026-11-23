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
#include "../asynchronous_complete_cumsum/asynchronous_complete_cumsum.h"

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
    const c10::optional<at::Tensor>& total_num_blocks,
    const c10::optional<at::Tensor>& indice_additions,
    const bool use_shift_filter,
    const bool allow_negative_key
    )
{
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D.");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D.");
    TORCH_CHECK(block_sizes.dim() == 1, "block_sizes must be 1D.");
    TORCH_CHECK(block_sizes.numel() > 0, "block_sizes must be non-empty.");
    TORCH_CHECK(
        block_sizes.scalar_type() == indices.scalar_type(),
        "block_sizes dtype must match indices.");
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

    if (indice_additions.has_value()) {
        TORCH_CHECK(indice_additions.value().dim() == 1, "indice_additions must be 1D.");
        TORCH_CHECK(indice_additions.value().scalar_type() == indices.scalar_type(),
            "indice_additions dtype must match indices.");
        TORCH_CHECK(indice_additions.value().numel() == block_sizes.numel(),
            "indice_additions must match block_sizes length.");
    }
}

BucketizeResult block_bucketize_sparse_features_npu_full(
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
    const c10::optional<at::Tensor>& total_num_blocks,
    const c10::optional<at::Tensor>& indice_additions,
    const bool use_shift_filter,
    const bool allow_negative_key)
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
    if (batch_size_per_feature.has_value()) {
        tensors.push_back(batch_size_per_feature.value());
        names.push_back("batch_size_per_feature");
    }
    if (total_num_blocks.has_value()) {
        tensors.push_back(total_num_blocks.value());
        names.push_back("total_num_blocks");
    }
    if (indice_additions.has_value()) {
        tensors.push_back(indice_additions.value());
        names.push_back("indice_additions");
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
        total_num_blocks,
        indice_additions,
        use_shift_filter,
        allow_negative_key);

    const at::OptionalDeviceGuard guard(device_of(lengths));

    auto lengths_contig = lengths.contiguous();
    auto indices_contig = indices.contiguous();
    auto block_sizes_contig = block_sizes.contiguous();

    const auto lengths_size = lengths_contig.numel();
    const auto num_features = block_sizes_contig.numel();
    const auto new_lengths_size = lengths_size * my_size;
    const bool enable_batch_size_per_feature = batch_size_per_feature.has_value();
    TORCH_CHECK(num_features > 0, "num_features must be positive.");
    const int64_t batch_size = enable_batch_size_per_feature ? 0 : (lengths_size / num_features);

    auto new_lengths =
        at::zeros({new_lengths_size}, lengths_contig.options());
    auto new_indices =
        at::empty({indices_contig.numel()}, indices_contig.options());

    auto empty_tensor = at::Tensor();

    auto total_num_blocks_contig = total_num_blocks.has_value() ?
        total_num_blocks.value().contiguous() : empty_tensor;
    auto indice_additions_contig = indice_additions.has_value() ?
        indice_additions.value().contiguous() : empty_tensor;
    tensor_list block_bucketize_pos_vector = block_bucketize_pos.has_value() ?
        block_bucketize_pos.value() : tensor_list{at::empty({0}, indices_contig.options())};
    at::TensorList block_bucketize_pos_list_tensor = block_bucketize_pos_vector;

    auto offsets = at::cumsum(lengths_contig, 0, lengths_contig.scalar_type()).contiguous();

    auto batch_size_offsets = empty_tensor;
    if (enable_batch_size_per_feature) {
        auto bspf_contig = batch_size_per_feature.value();
        batch_size_offsets = asynchronous_complete_cumsum_npu(bspf_contig).contiguous();
    }

    EXEC_NPU_CMD(
        aclnnBlockBucketizeSparseFeaturesComputeNewLengths,
        /* input - required */
        indices_contig,
        block_sizes_contig,
        offsets,
        /* input - optional */
        total_num_blocks_contig,
        batch_size_offsets,
        /* input - dynamic */
        block_bucketize_pos_list_tensor,
        indice_additions_contig,
        /* attr */
        my_size,
        bucketize_pos,
        lengths_size,
        batch_size,
        max_B,
        /* output */
        use_shift_filter,
        allow_negative_key,
        new_lengths);

    auto new_offsets = asynchronous_complete_cumsum_npu(new_lengths).contiguous();

    auto weight_contig = weights.has_value() ?
        weights.value().contiguous() : empty_tensor;
    auto new_weights_tensor = weights.has_value() ?
        at::empty({weight_contig.numel()}, weight_contig.options()) : empty_tensor;
    auto new_pos_tensor = bucketize_pos ?
        at::empty({indices_contig.numel()}, indices_contig.options()) : empty_tensor;
    auto unbucketize_permute_tensor = sequence ?
        at::empty({indices_contig.numel()}, indices_contig.options()) : empty_tensor;

    EXEC_NPU_CMD(
        aclnnBlockBucketizeSparseFeaturesScatterNewIndices,
        /* input - required */
        indices_contig,
        block_sizes_contig,
        offsets,
        new_offsets,
        /* input - optional */
        weight_contig,
        total_num_blocks_contig,
        batch_size_offsets,
        /* input - dynamic */
        block_bucketize_pos_list_tensor,
        indice_additions_contig,
        /* attr */
        my_size,
        bucketize_pos,
        sequence,
        keep_orig_idx,
        lengths_size,
        batch_size,
        max_B,
        use_shift_filter,
        allow_negative_key,
        /* output */
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

BucketizeResult block_bucketize_sparse_features_npu_ori(
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
    return block_bucketize_sparse_features_npu_full(lengths, indices, bucketize_pos, sequence, block_sizes, my_size,
                                                    weights, batch_size_per_feature, max_B, block_bucketize_pos,
                                                    keep_orig_idx, total_num_blocks, c10::nullopt, false, true);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m) {
    m.def(
        "block_bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, "
        "Tensor block_sizes, SymInt my_size, Tensor? weights=None, Tensor? batch_size_per_feature=None, "
        "SymInt max_B= -1, Tensor[]? block_bucketize_pos=None, bool keep_orig_idx=False, "
        "Tensor? total_num_blocks=None, Tensor? indice_additions=None, "
        "bool edison_use_shift_filter=False, bool allow_negative_key=True) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m) {
    m.impl(
        "block_bucketize_sparse_features",
        &block_bucketize_sparse_features_npu_full);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) {
    m.impl(
        "block_bucketize_sparse_features",
        &block_bucketize_sparse_features_npu_ori);
}
