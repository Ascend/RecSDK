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
    // weight暂不支持
    at::Tensor weightsConti = at::empty({1}, lengths.options());

    const auto T = lengths.size(0);
    const auto B = lengths.size(1);

    at::Tensor outLengths = at::empty({T, B}, lengthsConti.options());
    at::Tensor outValues = at::empty({valuesConti.size(0)}, valuesConti.options());
    at::Tensor outWeights = at::empty({1}, weightsConti.options());

    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti, T,
                 outLengths, outValues, outWeights);

    return make_tuple(outLengths, outValues, at::Tensor());
}

TORCH_LIBRARY(mxrec, m)
{
    m.def("permute_2D_sparse_data(Tensor permute, "
                                 "Tensor lengths, "
                                 "Tensor values, "
                                 "Tensor? weights=None, "
                                 "SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("permute_2D_sparse_data", &permute2d_sparse_data_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("permute_2D_sparse_data", &permute2d_sparse_data_impl_npu);
}
