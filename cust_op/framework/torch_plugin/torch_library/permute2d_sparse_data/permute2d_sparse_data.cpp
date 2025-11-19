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
#include "../common/common_utils.h"
#include "../asynchronous_complete_cumsum/asynchronous_complete_cumsum.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;
constexpr int EXPECTED_DIM_2D = 2;
constexpr int threshold_mean_lengths = 30000;
constexpr int threshold_mean_lengths_large = 750000;
constexpr int threshold_T = 56;
/**
 * 验证permute2d_sparse_data的输入参数
 * @param permute 排列索引张量
 * @param lengths 长度张量
 * @param values 值张量
 */
void validate_permute2d_sparse_data_inputs(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values)
{
    // ============= 空值检查 =============
    check_tensor_non_empty(permute, "permute");
    check_tensor_non_empty(lengths, "lengths");
    check_tensor_non_empty(values, "values");

    // ============= NPU设备检查 =============
    std::vector<Tensor> tensors = {permute, lengths, values};
    std::vector<std::string> names = {"permute", "lengths", "values"};
    check_tensor_npu_device(tensors, names);
}
 
tuple<Tensor, Tensor, c10::optional<Tensor>> permute2d_sparse_data_impl_npu(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values,
    const c10::optional<Tensor> &weights,
    const c10::optional<int64_t> &permuted_lengths_sum)
{
    validate_permute2d_sparse_data_inputs(permute, lengths, values);
    check_tensor_dim(lengths, EXPECTED_DIM_2D, "lengths");
    auto permuteConti = permute.contiguous();
    auto lengthsConti = lengths.contiguous();
    auto valuesConti = values.contiguous();
    auto weightsConti = weights.value_or(at::Tensor()).contiguous();

    const auto T = permute.size(0);
    const auto B = lengths.size(1);
    const auto lengthsSum = values.size(0);

    at::Tensor reduceSumLengths;
    at::Tensor permuteReduceSumLengths;
    at::Tensor totalOffset;
    at::Tensor permutedLengths = at::empty({T, B}, lengthsConti.options());
    at::Tensor lengthsOffset;
    at::Tensor permutedLengthsOffset;

    int cols = lengthsConti.size(1);
    auto ones_matrix = at::ones({cols, 1}, lengthsConti.options().dtype(at::kFloat));
    reduceSumLengths = at::matmul(lengthsConti.to(at::kFloat), ones_matrix).squeeze(1).to(lengthsConti.options());
    // 计算每行的平均元素数
    auto mean_lengths = lengthsSum / lengths.size(0);
    bool use_totalOffset = (mean_lengths > threshold_mean_lengths_large) ||
                           (T <= 10) ||
                           ((mean_lengths > threshold_mean_lengths) && (T < threshold_T));
    if (use_totalOffset) {
        totalOffset = asynchronous_complete_cumsum_npu(reduceSumLengths);
        lengthsOffset = at::Tensor();
        permutedLengthsOffset = at::Tensor();
    } else {
        totalOffset = at::Tensor();
        // 直接进行 index_select重排lengthsConti和reduceSumLengths
        permutedLengths = lengthsConti.index_select(0, permuteConti);
        permuteReduceSumLengths = reduceSumLengths.index_select(0, permuteConti);
        // 确保连续性，避免 NPU 内核中的内存访问开销
        if (!permutedLengths.is_contiguous()) {
            permutedLengths = permutedLengths.contiguous();
        }
        lengthsOffset = asynchronous_complete_cumsum_npu(reduceSumLengths);
        permutedLengthsOffset = asynchronous_complete_cumsum_npu(permuteReduceSumLengths);
    }

    int outValuesLen;
    if (permuted_lengths_sum.has_value() && permuted_lengths_sum.value() > 0) {
        int64_t permuted_lengths_sum_value = permuted_lengths_sum.value();
        TORCH_CHECK(permuted_lengths_sum_value <= std::numeric_limits<int>::max(),
            "permuted_lengths_sum limit (0, %d], but get %lld\n", std::numeric_limits<int>::max(),
            permuted_lengths_sum_value);
        outValuesLen = static_cast<int>(permuted_lengths_sum_value);
    } else {
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

    at::Tensor outLengths = at::empty({T, B}, lengthsConti.options());
    at::Tensor outValues = at::empty({outValuesLen}, valuesConti.options());
    at::Tensor outWeights = weights.has_value() ? at::empty({outValuesLen}, weightsConti.options()) : at::Tensor();

    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti,
                 totalOffset, lengthsOffset, permutedLengthsOffset, outValuesLen,
                 outLengths, outValues, outWeights);

    if (use_totalOffset) {
        return make_tuple(outLengths, outValues, outWeights);
    } else {
        return make_tuple(permutedLengths, outValues, outWeights);
    }
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
