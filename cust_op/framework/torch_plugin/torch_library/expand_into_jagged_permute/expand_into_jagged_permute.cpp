/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <torch/library.h>
#include <limits>
#include "torch/extension.h"
#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

using namespace at;

constexpr int EXPECTED_DIM_1D = 1;

// 为NPU设备注册实现
at::Tensor expand_into_jagged_permute_impl_npu(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    const c10::SymInt output_size)
{
    // 空值检查
    check_tensor_non_empty(permute, "permute");
    check_tensor_non_empty(input_offsets, "input_offsets");
    check_tensor_non_empty(output_offsets, "output_offsets");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {permute, input_offsets, output_offsets};
    std::vector<std::string> names = {"permute", "input_offsets", "output_offsets"};
    check_tensor_npu_device(tensors, names);

    // 维度检查
    TORCH_CHECK(permute.dim() == EXPECTED_DIM_1D, "The permute should be 1D");
    TORCH_CHECK(input_offsets.dim() == EXPECTED_DIM_1D, "The input_offsets should be 1D");
    TORCH_CHECK(output_offsets.dim() == EXPECTED_DIM_1D, "The output_offsets should be 1D");

    // 参数校验
    TORCH_CHECK(permute.numel() > 0, "permute.numel() must be greater than 0");
    TORCH_CHECK(
        permute.numel() == input_offsets.numel() - 1,
        "permute.numel() (", permute.numel(),
        ") must equal input_offsets.numel() - 1 (", input_offsets.numel() - 1, ")");
    TORCH_CHECK(
        permute.numel() == output_offsets.numel() - 1,
        "permute.numel() (", permute.numel(),
        ") must equal output_offsets.numel() - 1 (", output_offsets.numel() - 1, ")");

    // 数据类型检查（permute、input_offsets、output_offsets 应该类型一致）
    TORCH_CHECK(
        permute.scalar_type() == input_offsets.scalar_type() &&
            permute.scalar_type() == output_offsets.scalar_type(),
        "permute, input_offsets, and output_offsets must have the same dtype");

    // 转换 SymInt 为 int64_t
    const int64_t output_size_int = output_size.guard_int(__FILE__, __LINE__);

    // 确保 tensor 是连续的
    auto permute_contig = permute.contiguous();
    auto input_offsets_contig = input_offsets.contiguous();
    auto output_offsets_contig = output_offsets.contiguous();

    // 创建输出 tensor
    at::Tensor output_permute = at::empty({output_size_int}, permute_contig.options());

    EXEC_NPU_CMD(aclnnExpandIntoJaggedPermute, permute_contig, input_offsets_contig,
                 output_offsets_contig, output_size_int, output_permute);

    return output_permute;
}

// 在 mxrec 空间里注册expand_into_jagged_permute
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("expand_into_jagged_permute(Tensor permute, "
          "                             Tensor input_offset, "
          "                             Tensor output_offset, "
          "                             SymInt output_size) -> Tensor");
}

// 为NPU设备注册实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_impl_npu);
}
