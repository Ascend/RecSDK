/**
 * @file expand_into_jagged_permute.cpp
 *
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

void validate_expand_into_jagged_permute_inputs(
    const at::Tensor& permute,
    const at::Tensor& inputOffset,
    const at::Tensor& outputOffsets,
    const int64_t outputSize)
{
    // ============= 空值检查 =============
    CheckTensorNonEmpty(permute, "permute");
    CheckTensorNonEmpty(inputOffset, "inputOffset");
    CheckTensorNonEmpty(outputOffsets, "outputOffsets");

    CheckTensorDim(permute, EXPECTED_DIM_1D, "permute");
    CheckTensorDim(inputOffset, EXPECTED_DIM_1D, "inputOffset");
    CheckTensorDim(outputOffsets, EXPECTED_DIM_1D, "outputOffsets");

    const auto permute_len = permute.size(0);
    const auto input_offset_len = inputOffset.size(0);
    const auto output_offsets_len = outputOffsets.size(0);

    // 1. 校验inputOffset和outputOffset的shape要相同
    TORCH_CHECK(input_offset_len == output_offsets_len,
                "input_offset_len and output_offsets_len must be the same, but got input_offset_len: ",
                input_offset_len, " and output_offsets_len: ", output_offsets_len);

    TORCH_CHECK(permute_len == input_offset_len - 1,
                "permute_len must equals input_offset_len - 1, but got permute_len: ",
                permute_len, " and input_offset_len: ", input_offset_len);

    // 2. 校验所有输入张量的数据类型相同
    TORCH_CHECK(permute.scalar_type() == inputOffset.scalar_type(),
                "permute and inputOffset must have the same data type, but got permute: ",
                permute.scalar_type(), " and inputOffset: ", inputOffset.scalar_type());

    TORCH_CHECK(permute.scalar_type() == outputOffsets.scalar_type(),
                "permute and outputOffsets must have the same data type, but got permute: ",
                permute.scalar_type(), " and outputOffsets: ", outputOffsets.scalar_type());

    // 3. 校验outputOffset最后一个值等于output_size
    if (outputOffsets.numel() > 0) {
        auto last_offset = outputOffsets[-1].item<int64_t>();
        TORCH_CHECK(last_offset == outputSize,
                    "Last value of outputOffsets (", last_offset,
                    ") must equal outputSize (", outputSize, ")");
    } else {
        TORCH_CHECK(outputSize == 0,
                    "outputSize must be 0 when outputOffsets is empty, but got ", outputSize);
    }
}

at::Tensor expand_into_jagged_permute_impl_npu(const at::Tensor& permute,
                                               const at::Tensor& inputOffset,
                                               const at::Tensor& outputOffsets,
                                               const int64_t outputSize)
{
    validate_expand_into_jagged_permute_inputs(permute,
                                               inputOffset,
                                               outputOffsets,
                                               outputSize);

    const at::OptionalDeviceGuard guard(device_of(permute));
    at::Tensor outputPermuteOut = at::empty({outputSize}, permute.options());

    EXEC_NPU_CMD(aclnnExpandIntoJaggedPermute, permute, inputOffset, outputOffsets, outputSize, outputPermuteOut);

    return outputPermuteOut;
};

// 在NPU命名空间里面注册expand_into_jagged_permute
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("expand_into_jagged_permute(Tensor permute, "
          "                           Tensor inputOffset, "
          "                           Tensor outputOffsets, "
          "                           int outputSize) -> Tensor");
}

// 这里表示该算子的 NPU 实现由 expand_into_jagged_permute_impl_npu 函数提供
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_impl_npu);
}

// 将同一个算子同时注册到 fbgemm 库的 PrivateUse1 后端
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_impl_npu);
}