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
#include "../common/common_utils.h"
#include "../asynchronous_complete_cumsum/asynchronous_complete_cumsum.h"

using namespace at;
using namespace std;

constexpr int EXPECTED_DIM_1D = 1;
constexpr int threshold_mean_lengths = 30000;
constexpr int threshold_mean_lengths_large = 750000;
constexpr int threshold_T = 56;
 
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

    // ============= NPU设备检查 =============
    std::vector<Tensor> tensors = {permute, lengths, values};
    std::vector<std::string> names = {"permute", "lengths", "values"};
    
    // 如果有权重张量，也加入检查
    if (weights.has_value()) {
        check_tensor_non_empty(weights.value(), "weights");
        check_tensor_dim(weights.value(), EXPECTED_DIM_1D, "weights");
        tensors.push_back(weights.value());
        names.push_back("weights");
    }
    
    check_tensor_npu_device(tensors, names);

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
    auto lengthsConti = lengths.contiguous();
    auto valuesConti = values.contiguous();
    auto weightsConti = weights.value_or(at::Tensor()).contiguous();

    const auto pLength = permute.size(0);
    const auto lengthsSum = values.size(0);
    // 计算每行的平均元素数
    auto mean_lengths = lengthsSum / lengths.size(0);
    bool use_totalOffset = (mean_lengths > threshold_mean_lengths_large) ||
                           (pLength <= 10) ||
                           (mean_lengths > threshold_mean_lengths && pLength < threshold_T);

    at::Tensor totalOffset = at::Tensor();
    at::Tensor lengthsOffset = at::Tensor();
    at::Tensor permutedLengthsOffset = at::Tensor();
    at::Tensor permutedLengths = at::empty({pLength}, lengthsConti.options());
    if (use_totalOffset) {
        // 使用 asynchronous_complete_cumsum 计算累积和
        totalOffset = asynchronous_complete_cumsum_npu(lengthsConti);
        lengthsOffset = at::Tensor();
        permutedLengthsOffset = at::Tensor();
    } else {
        // 直接进行 index_select重排lengthsConti
        permutedLengths = lengthsConti.index_select(0, permuteConti);
        totalOffset = at::Tensor();
        // 使用 asynchronous_complete_cumsum 计算重排后lengthsConti的累积和
        lengthsOffset = asynchronous_complete_cumsum_npu(lengthsConti);
        permutedLengthsOffset = asynchronous_complete_cumsum_npu(permutedLengths);
    }

    int outValuesLen; // 输出值的长度
    if (permuted_lengths_sum.has_value() && permuted_lengths_sum.value() > 0) {
        // 提供了输出长度，直接使用
        int64_t permuted_lengths_sum_value = permuted_lengths_sum.value();
        TORCH_CHECK(permuted_lengths_sum_value <= std::numeric_limits<int>::max(),
            "permuted_lengths_sum limit (0, %d], but get %lld\n", std::numeric_limits<int>::max(),
            permuted_lengths_sum_value);
        outValuesLen = static_cast<int>(permuted_lengths_sum_value);
    } else {
        // 未提供输出长度，通过permute长度进行计算
        int64_t sum_value;
        if (use_totalOffset) {
            sum_value = lengthsConti.index_select(0, permuteConti).sum().item<int64_t>();
        } else {
            sum_value = permutedLengthsOffset[-1].item<int64_t>();
        }
        TORCH_CHECK(sum_value > 0 && sum_value <= std::numeric_limits<int>::max(),
            "sum_value limit (0, %d], but get %lld\n", std::numeric_limits<int>::max(), sum_value);
        outValuesLen = static_cast<int>(sum_value);
    }

    lengthsConti = lengthsConti.view({-1, 1});

    // 初始化输出向量
    at::Tensor outLengths = at::empty({pLength}, lengthsConti.options());
    at::Tensor outValues = at::empty({outValuesLen}, valuesConti.options());
    at::Tensor outWeights = weights.has_value() ? at::empty({outValuesLen}, weightsConti.options()) : at::Tensor();

    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti,
                 totalOffset, lengthsOffset, permutedLengthsOffset, outValuesLen, outLengths,
                 outValues, outWeights);
                
    if (use_totalOffset) {
        return make_tuple(outLengths, outValues, outWeights);
    } else {
        return make_tuple(permutedLengths, outValues, outWeights);
    }
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
 