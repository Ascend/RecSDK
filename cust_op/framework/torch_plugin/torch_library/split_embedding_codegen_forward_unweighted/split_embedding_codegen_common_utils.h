/**
 * @file split_embedding_codegen_common_utils.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef MXREC_ADD_ONS_SPLIT_EMBEDDING_CODEGEN_COMMON_UTILS_H
#define MXREC_ADD_ONS_SPLIT_EMBEDDING_CODEGEN_COMMON_UTILS_H
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "torch/extension.h"
#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

const int ADAGRAD_OPTIM_NUM = 1;
const int ADAM_OPTIM_NUM = 2;


namespace fbgemm_npu_lookups {
    inline at::Tensor compute_offset_per_key(
        const at::Tensor& offsets,
        const at::Tensor& weights_offsets,
        const at::Tensor& D_offsets)
    {
        // 计算每张表的indices个数
        int64_t batchs = (offsets.numel() - 1) / weights_offsets.numel();
        at::Tensor table_offsets = torch::arange(D_offsets.size(0), offsets.device()) * batchs;
        return offsets.index_select(0, table_offsets.to(at::kLong));
    }

    inline void validate_forward_data_inputs(
        const at::Tensor& dev_weights,
        const at::Tensor& weights_offsets,
        const at::Tensor& D_offsets,
        const at::Tensor& indices,
        const at::Tensor& offsets,
        const at::Tensor& hash_indices,
        const at::Tensor& offset_per_key,
        const at::Tensor& rows_per_table)
    {
        check_tensor_non_empty(dev_weights, "dev_weights");
        check_tensor_non_empty(weights_offsets, "weights_offsets");
        check_tensor_non_empty(D_offsets, "D_offsets");
        check_tensor_non_empty(indices, "indices");
        check_tensor_non_empty(offsets, "offsets");
        check_tensor_non_empty(offset_per_key, "offset_per_key");
        if (rows_per_table.defined()) {
            check_tensor_non_empty(rows_per_table, "rows_per_table");
        }
        if (hash_indices.defined()) {
            check_tensor_non_empty(hash_indices, "hash_indices");
        }
    }

    inline void validate_backward_data_inputs(
        const at::Tensor& grad_output,
        const at::Tensor& dev_weights,
        const at::Tensor& weights_offsets,
        const at::Tensor& D_offsets,
        const at::Tensor& hash_size_cumsum,
        const at::Tensor& indices,
        const at::Tensor& offsets,
        const at::Tensor& momentum1_dev,
        const at::Tensor& momentum2_dev,
        const at::Tensor& hash_indices,
        const at::Tensor& unique_ids,
        const at::Tensor& unique_offsets,
        const at::Tensor& unique_inverse,
        const at::Tensor& offset_per_key,
        int optim_num = 0)
    {
        check_tensor_non_empty(grad_output, "grad_output");
        check_tensor_non_empty(dev_weights, "dev_weights");
        check_tensor_non_empty(weights_offsets, "weights_offsets");
        check_tensor_non_empty(D_offsets, "D_offsets");
        check_tensor_non_empty(indices, "indices");
        check_tensor_non_empty(offsets, "offsets");
        check_tensor_non_empty(offset_per_key, "offset_per_key");

        if (hash_indices.defined()) {
            check_tensor_non_empty(hash_indices, "hash_indices");
        }
        if (unique_ids.defined()) {
            check_tensor_non_empty(unique_ids, "unique_ids");
            check_tensor_non_empty(unique_offsets, "unique_offsets");
            check_tensor_non_empty(unique_inverse, "unique_inverse");
        }
        if (optim_num >= ADAGRAD_OPTIM_NUM) {
            check_tensor_non_empty(momentum1_dev, "momentum1_dev");
        }
        if (optim_num >= ADAM_OPTIM_NUM) {
            check_tensor_non_empty(momentum2_dev, "momentum2_dev");
        }
    }
    
}; // namespace fbgemm_npu_lookups
#endif // MXREC_ADD_ONS_SPLIT_EMBEDDING_CODEGEN_COMMON_UTILS_H
 