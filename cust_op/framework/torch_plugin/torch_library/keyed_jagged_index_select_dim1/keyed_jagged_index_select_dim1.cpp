/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

namespace {
constexpr int EXPECTED_DIM_1D = 1;
constexpr int THRESHOLD_MEAN_LENGTHS = 30000;
constexpr int THRESHOLD_MEAN_LENGTHS_LARGE = 750000;
constexpr int THRESHOLD_T = 56;
}  // namespace

void validate_keyed_jagged_index_select_dim1_inputs(const Tensor& values, const Tensor& lengths, const Tensor& offsets,
                                                    const Tensor& indices, const c10::optional<Tensor>& weights,
                                                    const c10::optional<int64_t>& selectedLengthsSum)
{
    check_tensor_dim(values, EXPECTED_DIM_1D, "values");
    check_tensor_dim(lengths, EXPECTED_DIM_1D, "lengths");
    check_tensor_dim(offsets, EXPECTED_DIM_1D, "offsets");
    check_tensor_dim(indices, EXPECTED_DIM_1D, "indices");
    TORCH_CHECK(offsets.dtype() == at::kLong, "offsets must be dtype Long");

    // ============= NPU设备检查 =============
    std::vector<Tensor> tensors = {values, lengths, offsets, indices};
    std::vector<std::string> names = {"values", "lengths", "offsets", "indices"};

    if (weights.has_value()) {
        check_tensor_non_empty(weights.value(), "weights");
        check_tensor_dim(weights.value(), EXPECTED_DIM_1D, "weights");
        tensors.push_back(weights.value());
        names.push_back("weights");
    }
    check_tensor_npu_device(tensors, names);

    // ============ 长度一致性检查 ============
    const auto valuesLen = values.size(0);
    // 检查weights张量(如果存在)
    if (weights.has_value()) {
        check_tensor_non_empty(*weights, "weights");
        check_tensor_dim(*weights, EXPECTED_DIM_1D, "weights");
        const auto weightsLen = weights->size(0);
        TORCH_CHECK(weightsLen == valuesLen, "weights and values length mismatch: ", weightsLen, " vs ", valuesLen);
    }

    // 检查selectedLengthsSum(如果存在)
    if (selectedLengthsSum.has_value()) {
        TORCH_CHECK(selectedLengthsSum.value() >= 0, "selectedLengthsSum must be non-negative, got ",
                    selectedLengthsSum.value());
    }
}

/**
 * permute1d_sparse_data算子的NPU实现
 * @param values 值张量排列索引张量
 * @param lengths 长度张量
 * @param offsets 偏移量
 * @param indices 索引张量
 * @param batchSize  稀疏矩阵batch大小，lengths长度维batchSize的倍数
 * @param weights 可选权重张量
 * @param selectedLengthsSum 可选排列后长度和
 * @return 元组包含(输出长度, 输出值, 输出权重)
 */
std::vector<Tensor> keyed_jagged_index_select_dim1_impl_npu(const Tensor& values, const Tensor& lengths,
                                                            const Tensor& offsets, const Tensor& indices,
                                                            const int64_t& batchSize,
                                                            const c10::optional<Tensor>& weights,
                                                            const c10::optional<int64_t>& selectedLengthsSum)
{
    validate_keyed_jagged_index_select_dim1_inputs(values, lengths, offsets, indices, weights, selectedLengthsSum);
    auto valuesConti = values.contiguous();
    auto lengthsConti = lengths.contiguous();
    auto offsetsConti = offsets.contiguous();
    auto indicesConti = indices.contiguous();
    auto weightsConti = weights.value_or(at::empty({}, at::kFloat)).contiguous();
    bool enableWeights = weights.has_value();
    TORCH_CHECK(batchSize > 0, "batchSize must be greater than 0", ", but get ", batchSize, "\n");
    const auto lengthsSize = lengthsConti.size(0);
    const auto outlengthsSize = lengthsSize / batchSize * indicesConti.size(0);
    TORCH_CHECK(lengthsSize <= std::numeric_limits<int>::max(), "lengthsSize limit (0, ",
                std::numeric_limits<int>::max(), "], but get ", lengthsSize, "\n");
    TORCH_CHECK(outlengthsSize <= std::numeric_limits<int>::max(), "lengthsSize limit (0, ",
                std::numeric_limits<int>::max(), "], but get ", lengthsSize, "\n");
    at::Tensor permute = at::empty({outlengthsSize}, indicesConti.options());
    EXEC_NPU_CMD(aclnnSelectDim1ToPermute, indicesConti, batchSize, lengthsSize, permute);

    const auto pLength = permute.size(0);
    const auto lengthSize = lengths.size(0);
    const auto lengthsSum = values.size(0);
    if (lengthSize == 0 || pLength == 0) {
        return {valuesConti.clone(), lengthsConti.clone(), enableWeights ? weightsConti.clone() : at::Tensor()};
    }
    // 计算每行的平均元素数
    auto meanLengths = lengthsSum / lengthSize;
    bool useOffset =
        (meanLengths > THRESHOLD_MEAN_LENGTHS_LARGE) || (meanLengths > THRESHOLD_MEAN_LENGTHS && pLength < THRESHOLD_T);

    at::Tensor totalOffset = at::Tensor();
    at::Tensor lengthsOffset = at::Tensor();
    at::Tensor permutedLengthsOffset = at::Tensor();
    at::Tensor permutedLengths = at::empty({pLength}, lengthsConti.options());
    auto permuteConti = permute.contiguous();
    if (useOffset) {
        totalOffset = offsetsConti;
    } else {
        // 直接进行 index_select重排lengthsConti
        permutedLengths = lengthsConti.index_select(0, permuteConti);
        // 使用 asynchronous_complete_cumsum 计算重排后lengthsConti的累积和
        lengthsOffset = offsetsConti;
        permutedLengthsOffset = asynchronous_complete_cumsum_npu(permutedLengths).to(at::kLong);
    }

    int64_t outvaluesSize = 0;
    if (selectedLengthsSum.has_value() && selectedLengthsSum.value() > 0) {
        // 提供了输出长度, 直接使用
        outvaluesSize = static_cast<int64_t>(selectedLengthsSum.value());
    } else {
        // 未提供输出长度，通过permute长度进行计算
        if (useOffset) {
            outvaluesSize = lengthsConti.index_select(0, permuteConti).sum().item<int64_t>();
        } else {
            outvaluesSize = permutedLengthsOffset.index({-1}).item<int64_t>();
        }
    }
    lengthsConti = lengthsConti.view({-1, 1});
    // 初始化输出向量
    at::Tensor outlengths = at::empty({outlengthsSize}, lengthsConti.options());
    at::Tensor outvalues = at::empty({outvaluesSize}, valuesConti.options());
    at::Tensor outweights = enableWeights ? at::empty({outvaluesSize}, weightsConti.options()) : at::Tensor();
    EXEC_NPU_CMD(aclnnPermute2dSparseData, permuteConti, lengthsConti, valuesConti, weightsConti, totalOffset,
                 lengthsOffset, permutedLengthsOffset, outvaluesSize, enableWeights, outlengths, outvalues, outweights);

    if (useOffset) {
        return {outvalues, outlengths, outweights};
    } else {
        return {outvalues, permutedLengths, outweights};
    }
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("keyed_jagged_index_select_dim1(Tensor values, "
          "                               Tensor lengths, "
          "                               Tensor offsets, "
          "                               Tensor indices, "
          "                               int batch_size, "
          "                               Tensor? weights=None, "
          "                               int? selected_lengths_sum=None) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("keyed_jagged_index_select_dim1", &keyed_jagged_index_select_dim1_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("keyed_jagged_index_select_dim1", &keyed_jagged_index_select_dim1_impl_npu);
}
