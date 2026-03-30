/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

constexpr int EXPECTED_DIM_1D = 1;
constexpr int EXPECTED_DIM_2D = 2;
constexpr int THRESHOLD_MEAN_LENGTHS = 30000;
constexpr int THRESHOLD_MEAN_LENGTHS_LARGE = 750000;
constexpr int THRESHOLD_MIN_PERMUTE_LENGTHS = 10;
constexpr int THRESHOLD_T = 56;
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
    auto weightsConti = weights.value_or(at::empty({}, at::kFloat)).contiguous();
    bool enableWeights = weights.has_value();

    int32_t weightsColumns = 1;
    const auto T = permute.size(0);
    const auto lengthsRows = lengths.size(0);
    const auto batchSize = lengths.size(1);
    const auto lengthsSum = values.size(0);
    if (lengthsRows == 0 || batchSize == 0 || T == 0) {
        return make_tuple(lengthsConti.clone(), valuesConti.clone(),
                          enableWeights ? c10::make_optional(weightsConti.clone()) : c10::nullopt);
    }

    at::Tensor reduceSumLengths;
    at::Tensor permuteReduceSumLengths = at::Tensor();
    at::Tensor permutedLengths = at::Tensor();
    at::Tensor totalOffset = at::Tensor();
    at::Tensor lengthsOffset = at::Tensor();
    at::Tensor permutedLengthsOffset = at::Tensor();

    int cols = lengthsConti.size(1);
    auto onesMatrix = at::ones({cols, 1}, lengthsConti.options().dtype(at::kFloat));
    reduceSumLengths = at::matmul(lengthsConti.to(at::kFloat), onesMatrix).squeeze(1).to(at::kLong);
    // 计算每行的平均元素数
    auto meanLengths = lengthsSum / lengthsRows;
    bool useTotalOffset = (meanLengths > THRESHOLD_MEAN_LENGTHS_LARGE) ||
                          (T <= THRESHOLD_MIN_PERMUTE_LENGTHS) ||
                          ((meanLengths > THRESHOLD_MEAN_LENGTHS) && (T < THRESHOLD_T));
    if (useTotalOffset) {
        totalOffset = asynchronous_complete_cumsum_npu(reduceSumLengths);
    } else {
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

    int64_t outValuesLen = 0;
    if (permuted_lengths_sum.has_value()) {
        outValuesLen = static_cast<int64_t>(permuted_lengths_sum.value());
    } else {
        if (useTotalOffset) {
            outValuesLen = lengthsConti.index_select(0, permuteConti).sum().item<int64_t>();
        } else {
            outValuesLen = permutedLengthsOffset.index({-1}).item<int64_t>();
        }
    }

    at::Tensor outLengths = at::empty({T, batchSize}, lengthsConti.options());
    at::Tensor outValues = at::empty({outValuesLen}, valuesConti.options());
    at::Tensor outWeights = at::Tensor();

    if (enableWeights) {
        if (weightsConti.dense_dim() > 1) {
            weightsColumns = weightsConti.size(1);
            outWeights = at::empty({outValuesLen, weightsColumns}, weightsConti.options());
        } else {
            outWeights = at::empty({outValuesLen}, weightsConti.options());
        }
    }

    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti, totalOffset,
                 lengthsOffset, permutedLengthsOffset, outValuesLen, enableWeights, outLengths, outValues, outWeights);
    if (useTotalOffset) {
        return make_tuple(outLengths, outValues, outWeights);
    } else {
        return make_tuple(permutedLengths, outValues, outWeights);
    }
}

tuple<Tensor, Tensor, c10::optional<Tensor>> permute2d_sparse_data_input1D_impl_npu(
    const Tensor &permute,
    const Tensor &lengths,
    const Tensor &values,
    const int64_t &stride,
    const c10::optional<Tensor> &weights,
    const c10::optional<int64_t> &permuted_lengths_sum)
{
    check_tensor_dim(lengths, EXPECTED_DIM_1D, "lengths");
    auto [outLengths, outValues, outWeights] =
        permute2d_sparse_data_impl_npu(permute,
                                       lengths.view({-1, stride}),
                                       values,
                                       weights,
                                       permuted_lengths_sum);
    return make_tuple(outLengths.view({-1}), outValues, outWeights);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("permute_2D_sparse_data(Tensor permute, "
          "                       Tensor lengths, "
          "                       Tensor values, "
          "                       Tensor? weights=None, "
          "                       SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
    m.def("permute_sparse_data(Tensor permute, "
          "                       Tensor lengths, "
          "                       Tensor values, "
          "                       Tensor? weights=None, "
          "                       SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
    m.def("permute_2D_sparse_data_input1D(Tensor permute, "
          "                       Tensor lengths, "
          "                       Tensor values, "
          "                       int stride, "
          "                       Tensor? weights=None, "
          "                       SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("permute_2D_sparse_data", &permute2d_sparse_data_impl_npu);
    m.impl("permute_sparse_data", &permute2d_sparse_data_impl_npu);
    m.impl("permute_2D_sparse_data_input1D", &permute2d_sparse_data_input1D_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("permute_2D_sparse_data", &permute2d_sparse_data_impl_npu);
    m.impl("permute_sparse_data", &permute2d_sparse_data_impl_npu);
    m.impl("permute_2D_sparse_data_input1D", &permute2d_sparse_data_input1D_impl_npu);
}
