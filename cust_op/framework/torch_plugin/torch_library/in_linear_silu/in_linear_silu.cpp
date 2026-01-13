/**
 * @file in_linear_silu.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <torch/torch.h>

#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <stdexcept>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using namespace at;

torch::autograd::variable_list RunInLinearSiluForwardInter(const at::Tensor& x, const at::Tensor& weight,
                                                           const at::Tensor& bias, at::IntArrayRef splitList,
                                                           bool requiresGrad)
{
    at::TensorOptions options = x.options();
    auto m = x.size(0);
    auto n = weight.size(0);
    auto userOut = at::empty({m, splitList[0]}, options);
    auto valueOut = at::empty({m, splitList[1]}, options);
    auto queryOut = at::empty({m, splitList[2]}, options);
    auto keyOut = at::empty({m, splitList[3]}, options);
    auto linearOutputOut = at::empty({m, n}, options);

    auto xCont = x.contiguous();
    auto weightCont = weight.contiguous();
    auto biasCont = bias.contiguous();

    EXEC_NPU_CMD(aclnnInLinearSilu, xCont, weightCont, biasCont, splitList, requiresGrad, userOut, valueOut, queryOut,
                 keyOut, linearOutputOut);
    return {userOut, valueOut, queryOut, keyOut, linearOutputOut};
}

torch::autograd::variable_list RunInLinearSiluForward(const at::Tensor& x, const at::Tensor& weight,
                                                      const at::Tensor& bias, at::IntArrayRef splitList)
{
    bool requiresGrad = false;
    if (x.requires_grad() || weight.requires_grad() || bias.requires_grad()) {
        requiresGrad = true;
    }
    return RunInLinearSiluForwardInter(x, weight, bias, splitList, requiresGrad);
}

torch::autograd::variable_list RunInLinearSiluBackward(const at::Tensor& gradUser, const at::Tensor& gradValue,
                                                       const at::Tensor& gradQuery, const at::Tensor& gradKey,
                                                       const at::Tensor& x, const at::Tensor& weight,
                                                       const at::Tensor& bias, const at::Tensor& linearOutput)
{
    // split 反向 合并梯度
    auto gradActivated = torch::cat({gradUser, gradValue, gradQuery, gradKey}, -1);

    // silu
    auto gradLinear = torch::silu_backward(gradActivated, linearOutput);

    // addmm
    // 初始化梯度张量
    at::Tensor gradX;
    at::Tensor gradWeight;
    at::Tensor gradBias;

    // 计算x的梯度
    if (x.requires_grad()) {
        gradX = torch::mm(gradLinear, weight);
    }
    // 计算weight的梯度
    if (weight.requires_grad()) {
        gradWeight = torch::mm(gradLinear.t(), x);
    }
    // 计算bias的梯度
    if (bias.defined() && bias.requires_grad()) {
        gradBias = gradLinear.sum(0, false);
    }

    return {gradX, gradWeight, gradBias, at::Tensor()};
}

at::Tensor RunSiluBackward(const at::Tensor& gradActivated, const at::Tensor& linearOutput)
{
    return torch::silu_backward(gradActivated, linearOutput);
}

class RunInLinearSiluFunction : public torch::autograd::Function<RunInLinearSiluFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& x,
                                                  const at::Tensor& weight, const at::Tensor& bias,
                                                  at::IntArrayRef splitList)
    {
        bool requiresGrad = false;
        if (x.requires_grad() || weight.requires_grad() || bias.requires_grad()) {
            requiresGrad = true;
        }
        ctx->saved_data["requiresGrad"] = requiresGrad;

        auto output = RunInLinearSiluForwardInter(x, weight, bias, splitList, requiresGrad);
        if (requiresGrad) {
            ctx->save_for_backward({x, weight, bias, output[4]});
        }
        return {output[0], output[1], output[2], output[3], output[4]};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                   torch::autograd::variable_list grad_outputs)
    {
        auto requiresGrad = ctx->saved_data["requiresGrad"].toBool();
        if (!requiresGrad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        // 获取保存的张量
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto linearOutput = saved[3];

        return RunInLinearSiluBackward(grad_outputs[0], grad_outputs[1], grad_outputs[2], grad_outputs[3], x, weight,
                                       bias, linearOutput);
    }
};

torch::autograd::variable_list RunInLinearSilu(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
                                               at::IntArrayRef splitList)
{
    return RunInLinearSiluFunction::apply(x, weight, bias, splitList);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("distance_in_linear_silu(Tensor x, Tensor weight, Tensor bias, int[] split_list) -> Tensor[]");
    m.def("distance_in_linear_silu_forward(Tensor x, Tensor weight, Tensor bias, int[] split_list) -> Tensor[]");
    m.def("distance_in_linear_silu_backward(Tensor gradUser, Tensor gradValue, Tensor gradQuery, Tensor gradKey, "
          "Tensor x, Tensor weight, Tensor bias, Tensor linearOutput) -> Tensor[]");
    m.def("silu_backward(Tensor grad_silu, Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("distance_in_linear_silu_forward", &RunInLinearSiluForward);
    m.impl("distance_in_linear_silu_backward", &RunInLinearSiluBackward);
    m.impl("silu_backward", &RunSiluBackward);
}

TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("distance_in_linear_silu", &RunInLinearSilu);
}