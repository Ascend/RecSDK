/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;

tuple<Tensor, Tensor, c10::optional<Tensor>> permute2d_sparse_data_impl_npu(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values,
    const c10::optional<Tensor> &weights,
    const c10::optional<int64_t> &permuted_lengths_sum)
{
    auto permuteConti = permute.contiguous();
    auto lengthsConti = lengths.contiguous();
    auto valuesConti = values.contiguous();
    auto weightsConti = weights.value_or(at::Tensor()).contiguous();

    const auto T = permute.size(0);
    const auto B = lengths.size(1);
    // When permute num element = 0, or B = 0, permutation will not be performed. Return the input tensor.
    if (permute.numel() == 0 || B == 0) {
        return {
            lengths.clone(),
            values.clone(),
            weights.has_value() ? std::make_optional(weights->clone()) : std::nullopt
        };
    }

    int outValuesLen;
    if (permuted_lengths_sum.has_value() && permuted_lengths_sum.value() > 0) {
        outValuesLen = static_cast<int>(permuted_lengths_sum.value());
    } else {
        outValuesLen = lengthsConti.index_select(0, permuteConti).sum().item<int>();
    }

    at::Tensor outLengths = at::empty({T, B}, lengthsConti.options());
    at::Tensor outValues = at::zeros({outValuesLen}, valuesConti.options());
    at::Tensor outWeights = weights.has_value() ? at::zeros({outValuesLen}, weightsConti.options()) : at::Tensor();
    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti, outValuesLen,
                 outLengths, outValues, outWeights);

    return make_tuple(outLengths, outValues, outWeights);
}

TORCH_LIBRARY(mxrec, m)
{
    m.def("permute_2D_sparse_data(Tensor permute, "
          "                       Tensor lengths, "
          "                       Tensor values, "
          "                       Tensor? weights=None, "
          "                       SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("permute_2D_sparse_data", &permute2d_sparse_data_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("permute_2D_sparse_data", &permute2d_sparse_data_impl_npu);
}
