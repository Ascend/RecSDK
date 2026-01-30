/**
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/torch.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

using namespace at;

namespace linear_silu_split_uvqk {
const int MIN_DIM = 16;
const int MAX_DIM = 512 * 16; // max_dim * max_num_head
const int TENSOR_NUM = 4;
const int DIM_2 = 2;
const int DIM_1 = 1;

void IsValidShape(const at::Tensor& x, const at::Tensor& weight,
                  const at::Tensor& bias, at::IntArrayRef splitList)
{
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(weight, "weight");
    check_tensor_non_empty(bias, "bias");

    check_tensor_dim(x, DIM_2, "x");
    check_tensor_dim(weight, DIM_2, "weight");
    check_tensor_dim(bias, DIM_1, "bias");

    TORCH_CHECK(splitList.size() == TENSOR_NUM, "splitList must have 4 elements.");
    int32_t totalDim = 0;
    for (int i = 0; i < TENSOR_NUM; i++) {
        TORCH_CHECK(splitList[i] >= MIN_DIM && splitList[i] <= MAX_DIM && splitList[i] % MIN_DIM == 0,
            "uvqk dim must in range[16, 8192] and multiples of 16.");
        totalDim += splitList[i];
    }

    auto k = x.size(1);
    auto n = weight.size(0);
    TORCH_CHECK(k >= MIN_DIM && k <= MAX_DIM && k % MIN_DIM == 0,
        "x dim[1] must in range[16, 8192] and multiples of 16.");
    TORCH_CHECK(n >= MIN_DIM * TENSOR_NUM && n <= MAX_DIM * TENSOR_NUM && n % MIN_DIM == 0,
        "weight dim[0] must in range[64, 32768] and multiples of 16.");
    TORCH_CHECK(totalDim == n, "weight_dim[0] must equal to sum(splitList).");
    TORCH_CHECK(bias.size(0) == n, "bias dim[0] must equal to weight dim[0].");
    TORCH_CHECK(weight.size(1) == k, "weight dim[1] must equal to x dim[1].");
    TORCH_CHECK((n % (4 * k) == 0), "weight dim[0] must be a multiple of 4 x dim[1]");
}
torch::autograd::variable_list RunLinearSiluSplitUvqkForwardInter(const at::Tensor& x, const at::Tensor& weight,
                                                                  const at::Tensor& bias, at::IntArrayRef splitList)
{
    IsValidShape(x, weight, bias, splitList);

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

    EXEC_NPU_CMD(aclnnInLinearSilu, xCont, weightCont, biasCont, splitList, userOut, valueOut,
                 queryOut, keyOut, linearOutputOut);
    return {userOut, valueOut, queryOut, keyOut, linearOutputOut};
}

at::Tensor RunConcatSiluGrad(const at::Tensor& grad1, const at::Tensor& grad2, const at::Tensor& grad3,
                             const at::Tensor& grad4, const at::Tensor& siluInput)
{
    at::TensorOptions options = grad1.options();
    auto m = siluInput.size(0);
    auto n = siluInput.size(1);
    auto gradSiluInput = at::empty({m, n}, options);

    auto grad1Cont = grad1.contiguous();
    auto grad2Cont = grad2.contiguous();
    auto grad3Cont = grad3.contiguous();
    auto grad4Cont = grad4.contiguous();
    auto siluInputCont = siluInput.contiguous();

    EXEC_NPU_CMD(aclnnConcatSiluGrad, grad1Cont, grad2Cont, grad3Cont, grad4Cont, siluInputCont, gradSiluInput);
    return gradSiluInput;
}

at::Tensor RunConcatSiluGradGolden(const at::Tensor& grad1, const at::Tensor& grad2, const at::Tensor& grad3,
                                   const at::Tensor& grad4, const at::Tensor& siluInput)
{
    // split 反向 合并梯度
    auto grad_activated = torch::cat({grad1, grad2, grad3, grad4}, -1);

    // silu
    auto grad_linear = torch::silu_backward(grad_activated, siluInput);
    return grad_linear;
}

torch::autograd::variable_list RunLinearSiluSplitUvqkForward(const at::Tensor& x, const at::Tensor& weight,
                                                             const at::Tensor& bias, at::IntArrayRef splitList)
{
    bool requiresGrad = false;
    if (x.requires_grad() || weight.requires_grad() || bias.requires_grad()) {
        requiresGrad = true;
    }
    return RunLinearSiluSplitUvqkForwardInter(x, weight, bias, splitList);
}

torch::autograd::variable_list RunLinearSiluSplitUvqkBackward(const at::Tensor& grad_user, const at::Tensor& grad_value,
                                                              const at::Tensor& grad_query, const at::Tensor& grad_key,
                                                              const at::Tensor& x, const at::Tensor& weight,
                                                              const at::Tensor& bias, const at::Tensor& linear_output)
{
    // split 反向 合并梯度
    auto grad_linear = RunConcatSiluGrad(grad_user, grad_value, grad_query, grad_key, linear_output);

    // addmm
    // 初始化梯度张量
    at::Tensor grad_x;
    at::Tensor grad_weight;
    at::Tensor grad_bias;

    // 计算x的梯度
    if (x.requires_grad()) {
        grad_x = torch::mm(grad_linear, weight);
    }
    // 计算weight的梯度
    if (weight.requires_grad()) {
        grad_weight = torch::mm(grad_linear.t(), x);
    }

    // 计算bias的梯度
    if (bias.defined() && bias.requires_grad()) {
        grad_bias = grad_linear.sum(0, false);
    }
    return {grad_x, grad_weight, grad_bias, at::Tensor()};
}

at::Tensor RunSiluBackward(const at::Tensor& grad_activated, const at::Tensor& linear_output)
{
    return torch::silu_backward(grad_activated, linear_output);
}

class RunLinearSiluSplitUvqkFunction : public torch::autograd::Function<RunLinearSiluSplitUvqkFunction> {
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
        auto output = RunLinearSiluSplitUvqkForwardInter(x, weight, bias, splitList);
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
        auto linear_output = saved[3];
        return RunLinearSiluSplitUvqkBackward(grad_outputs[0], grad_outputs[1], grad_outputs[2], grad_outputs[3], x,
                                              weight, bias, linear_output);
    }
};

torch::autograd::variable_list RunLinearSiluSplitUvqk(const at::Tensor& x, const at::Tensor& weight,
                                                      const at::Tensor& bias, at::IntArrayRef splitList)
{
    return RunLinearSiluSplitUvqkFunction::apply(x, weight, bias, splitList);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("linear_silu_split_uvqk(Tensor x, Tensor weight, Tensor bias, int[] split_list) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("linear_silu_split_uvqk", &RunLinearSiluSplitUvqk);
}

TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("linear_silu_split_uvqk", &RunLinearSiluSplitUvqk);
}

}  // namespace linear_silu_split_uvqk
