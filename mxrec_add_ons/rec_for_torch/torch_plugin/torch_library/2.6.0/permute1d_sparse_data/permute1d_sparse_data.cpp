/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

constexpr int EXPECTED_DIM_1D = 1;

/**
 * 检查张量是否非空
 * @param tensor 要检查的张量
 * @param name 张量名称(用于错误信息)
 */
void check_tensor_non_empty(const Tensor &tensor, const std::string &name)
{
    TORCH_CHECK(tensor.defined(), name, " tensor must be defined");
    TORCH_CHECK(tensor.numel() > 0, name, " tensor must be non-empty");
}

/**
 * 检查张量维度是否符合预期
 * @param tensor 要检查的张量
 * @param expected_dim 期望的维度
 * @param name 张量名称(用于错误信息)
 */
void check_tensor_dim(const Tensor &tensor, int64_t expected_dim, const std::string &name)
{
    TORCH_CHECK(tensor.dim() == expected_dim, name, " must be ", expected_dim, "D");
}

/**
 * 验证permute1d_sparse_data的输入参数
 * @param permute 排列索引张量
 * @param lengths 长度张量
 * @param values 值张量
 * @param weights 可选权重张量
 * @param permuted_lengths_sum 可选排列后长度和
 */
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
    check_tensor_dim(permute, EXPECTED_DIM_1D, "permute");
    check_tensor_dim(lengths, EXPECTED_DIM_1D, "lengths");
    check_tensor_dim(values, EXPECTED_DIM_1D, "values");

    // ============= 长度一致性检查 =============
    const auto permute_len = permute.size(0);
    const auto lengths_len = lengths.size(0);
    const auto values_len = values.size(0);

    // 检查weights张量(如果存在)
    if (weights.has_value()) {
        check_tensor_non_empty(*weights, "weights");
        check_tensor_dim(*weights, EXPECTED_DIM_1D, "weights");
        const auto weights_len = weights->size(0);
        TORCH_CHECK(weights_len == values_len,
            "weights and values length mismatch: ", weights_len, " vs ", values_len);
    }

    // 检查permuted_lengths_sum(如果存在)
    if (permuted_lengths_sum.has_value()) {
        TORCH_CHECK(permuted_lengths_sum.value() >= 0,
            "permuted_lengths_sum must be non-negative, got ", permuted_lengths_sum.value());
    }
}

/**
 * permute1d_sparse_data算子的NPU实现
 * @param permute 排列索引张量
 * @param lengths 长度张量
 * @param values 值张量
 * @param weights 可选权重张量
 * @param permuted_lengths_sum 可选排列后长度和
 * @return 元组包含(输出长度, 输出值, 输出权重)
 */
tuple<Tensor, Tensor, c10::optional<Tensor>> permute1d_sparse_data_impl_npu(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values,
    const c10::optional<Tensor> &weights,
    const c10::optional<int64_t> &permuted_lengths_sum)
{
    // 输入校验
    validate_permute1d_sparse_data_inputs(permute, lengths, values, weights, permuted_lengths_sum);

    // 确保张量是连续的(减少NPU内核中的内存访问开销)
    auto permuteConti = permute.contiguous();
    auto lengthsConti = lengths.contiguous().view({-1, 1});
    auto valuesConti = values.contiguous();
    auto weightsConti = weights.value_or(at::Tensor()).contiguous();

    const auto pLength = permute.size(0);

    int outValuesLen; // 输出值的长度
    if (permuted_lengths_sum.has_value() && permuted_lengths_sum.value() > 0) {
        // 提供了输出长度，直接使用
        outValuesLen = static_cast<int>(permuted_lengths_sum.value());
    } else {
        // 未提供输出长度，通过permute长度进行计算
        outValuesLen = lengthsConti.index_select(0, permuteConti).sum().item<int>();
    }

    // 初始化输出向量
    at::Tensor outLengths = at::empty({pLength}, lengthsConti.options());
    at::Tensor outValues = at::empty({outValuesLen}, valuesConti.options());
    at::Tensor outWeights = weights.has_value() ? at::empty({outValuesLen}, weightsConti.options()) : at::Tensor();

    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti, outValuesLen,
        outLengths, outValues, outWeights);

    return make_tuple(outLengths, outValues, outWeights);
}

// 在NPU命名空间里面注册permute_1D_sparse_data
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("permute_1D_sparse_data(Tensor permute, "
          "                       Tensor lengths, "
          "                       Tensor values, "
          "                       Tensor? weights=None, "
          "                       SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
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
