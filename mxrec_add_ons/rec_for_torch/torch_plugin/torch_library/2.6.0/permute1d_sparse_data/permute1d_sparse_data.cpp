/**
 * @file permute1d_sparse_data.cpp
 *
 * Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using namespace at;
using namespace std;

void check_tensor_non_empty(const Tensor &tensor, const std::string &name)
{
    TORCH_CHECK(tensor.defined(), name, " tensor must be non-empty");
}

void check_tensor_dim(const Tensor &tensor, int64_t expected_dim, const std::string &name)
{
    TORCH_CHECK(tensor.dim() == expected_dim, name, " must be ", expected_dim, "D");
}

void validate_permute1d_sparse_data_inputs(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values,
    const c10::optional<Tensor> &weights,
    const c10::optional<int64_t> &permuted_lengths_sum)
{
    // ============= 空值检查 =============
    check_tensor_non_empty(permute, "permute");
    check_tensor_non_empty(lengths, "lengths");
    check_tensor_non_empty(values, "values");

    // ============= 维度检查 =============
    check_tensor_dim(permute, 1, "permute");
    check_tensor_dim(lengths, 1, "lengths");
    check_tensor_dim(values, 1, "values");

    // ============= 长度一致性检查 =============
    const auto permute_len = permute.size(0);
    const auto lengths_len = lengths.size(0);
    const auto values_len = values.size(0);

    TORCH_CHECK(permute_len == lengths_len,
        "permute (length=", permute_len, ") and lengths (length=", lengths_len, ") must match!");

    // weights是optional的，只有has_value时才检查
    if (weights.has_value()) {
        check_tensor_non_empty(*weights, "weights");
        check_tensor_dim(*weights, 1, "weights");
        const auto weights_len = weights->size(0);
        TORCH_CHECK(weights_len == values_len,
            "weights and values length mismatch: ", weights_len, " vs ", values_len);
    }
}

// permute1d_sparse_data算子NPU实现
tuple<Tensor, Tensor, c10::optional<Tensor>> permute1d_sparse_data_impl_npu(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values,
    const c10::optional<Tensor> &weights,
    const c10::optional<int64_t> &permuted_lengths_sum)
{
    // 输入校验
    validate_permute1d_sparse_data_inputs(permute, lengths, values, weights, permuted_lengths_sum);

    auto permuteConti = permute.contiguous();
    auto lengthsConti = lengths.contiguous().view({-1, 1});
    auto valuesConti = values.contiguous();
    // weight暂不支持
    at::Tensor weightsConti = at::empty({1}, lengths.options());

    const auto T = lengths.size(0);

    at::Tensor outLengths = at::empty({T}, lengthsConti.options());
    at::Tensor outValues = at::empty({valuesConti.size(0)}, valuesConti.options());
    at::Tensor outWeights = at::empty({1}, weightsConti.options());

    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti, T,
        outLengths, outValues, outWeights);

    return make_tuple(outLengths, outValues, at::Tensor());
}

// 在NPU命名空间里面注册permute_1D_sparse_data
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("permute_1D_sparse_data(Tensor permute, "
                                 "Tensor lengths, "
                                 "Tensor values, "
                                 "Tensor? weights=None, "
                                 "SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
}

// 这里表示该算子的 NPU 实现由 permute1d_sparse_data_impl_npu 函数提供
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("permute_1D_sparse_data", &permute1d_sparse_data_impl_npu);
}

// 将同一个算子同时注册到 fbgemm 库的 PrivateUse1 后端
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("permute_1D_sparse_data", &permute1d_sparse_data_impl_npu);
}
