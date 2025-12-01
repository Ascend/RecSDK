/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <iostream>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


constexpr int EXPECTED_DIM_1D = 1;
constexpr int MAXINUM_LENGTH = 2147483647;

// 为NPU设备注册实现
at::Tensor invert_permute_impl_npu(const at::Tensor& x)
{
    // 空值检查
    check_tensor_non_empty(x, "x");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {x};
    std::vector<std::string> names = {"x"};
    check_tensor_npu_device(tensors, names);

    // 维度检查
    TORCH_CHECK(x.dim() == EXPECTED_DIM_1D, "The x should be 1D");

    // 长度检查
    auto x_len = x.size(0);
    TORCH_CHECK(x_len <= MAXINUM_LENGTH, "The maximum length of x is ", MAXINUM_LENGTH);

    auto x_conti = x.contiguous();
    at::Tensor y = at::zeros_like(x, x_conti.options());
    EXEC_NPU_CMD(aclnnInvertPermute, x_conti, y);

    return y;
}

// 在npu命名空间里注册invert_permute
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("invert_permute(Tensor x) -> Tensor");
}

// 为NPU设备注册实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("invert_permute", &invert_permute_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("invert_permute", &invert_permute_impl_npu);
}