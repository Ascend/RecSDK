/**
 * @file in_linear_silu.cpp
 *
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
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_2D = 2;
constexpr uint32_t CONST_4 = 4;
constexpr int EXPECTED_DIM_3D = 3;
constexpr int EXPECTED_DIM_1D = 1;
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> in_linear_silu_impl_npu(const at::Tensor& x,
    const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef attr_dict)
{
    check_tensor_non_empty(x, "x");
    check_tensor_dim(x, EXPECTED_DIM_2D, "x");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {x, weight, bias};
    std::vector<std::string> names = {"x", "weight", "bias"};
    check_tensor_npu_device(tensors, names);

    // // 获取输入Tensor的维度信息(根据实际需求调整)
    auto m = x.size(0);
    auto userNShape = attr_dict[0];
    auto valueNShape = attr_dict[1];
    auto queryNShape = attr_dict[2];
    auto keyNShape = attr_dict[3];

    auto xConti = x.contiguous();
    auto weightConti = weight.contiguous();
    auto biasConti = bias.contiguous();
    auto user = at::empty(at::IntArrayRef{m, userNShape}, x.options());
    auto value = at::empty(at::IntArrayRef{m, valueNShape}, x.options());
    auto query = at::empty(at::IntArrayRef{m, queryNShape}, x.options());
    auto key = at::empty(at::IntArrayRef{m, keyNShape}, x.options());
    EXEC_NPU_CMD(aclnnInLinearSilu, xConti, weightConti, biasConti, attr_dict, user, value, query, key);

    return std::make_tuple(user, value, query, key);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("distance_in_linear_silu(Tensor x, Tensor weight, Tensor bias, int[] attr_dict) -> (Tensor, Tensor, Tensor, "
          "Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("distance_in_linear_silu", &in_linear_silu_impl_npu);
}