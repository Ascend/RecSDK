/**
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <random>
#include <iostream>

#include "../common/pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;

namespace mxrec_npu_api {

at::Tensor dropout_gen_mask_impl(const at::Tensor& query, double keep_prob, int64_t seed, const int64_t offset,
                                 const int64_t numels)
{
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    c10::TensorOptions options = query.options();
    at::Tensor mask = at::zeros(at::IntArrayRef{length}, options.dtype(at::kByte));
    c10::SmallVector<int64_t> shapeSize = {numels};
    at::IntArrayRef shapeArray = at::IntArrayRef(shapeSize);
    double prob;
    at::Scalar probScalar;
    if (query.scalar_type() == at::kHalf) {
        probScalar = at::Scalar(at::Half(1.0) - at::Half(keep_prob));
    } else if (query.scalar_type() == at::kBFloat16) {
        probScalar = at::Scalar(at::BFloat16(1.0) - at::BFloat16(keep_prob));
    } else {
        probScalar = at::Scalar(float(1.0) - float(keep_prob));
    }
    prob = probScalar.toDouble();
    aclDataType probDataType = aclDataType::ACL_FLOAT;
    EXEC_NPU_CMD(aclnnDropoutGenMaskV2, shapeArray, prob, seed, offset, probDataType, mask);
    return mask;
}

std::mt19937& rng()
{
    static std::mt19937 gen{std::random_device{}()};
    return gen;
}

// 为NPU设备注册前向实现
tensor_list norm_multiply_dropout_impl_npu(const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                                           const at::Tensor& bias, double eps, double dropout_ratio)
{
    at::Tensor result = at::empty_like(x);
    std::vector<int64_t> x_sizes = x.sizes().vec();
    at::Tensor mean = at::empty({x_sizes[0]}, x.options().dtype(at::kFloat));
    at::Tensor var = at::empty({x_sizes[0]}, x.options().dtype(at::kFloat));

    float keep_prob = 1 - dropout_ratio;
    int64_t numels = x.numel();
    std::mt19937 rng(std::random_device{}());
    int64_t seed = rng();
    at::Tensor drop_mask = dropout_gen_mask_impl(x, keep_prob, seed, 0, numels);
    int64_t mask_numels = drop_mask.numel();

    // 调用EXEC_NPU_CMD接口，完成输出结果的计算
    EXEC_NPU_CMD(aclnnNormMultiplyDropout, x, u, weight, bias, drop_mask, eps, dropout_ratio, result, mean, var);
    return {result, mean, var, drop_mask};
}

// 为NPU设备注册反向实现
tensor_list norm_multiply_dropout_backward_impl_npu(const at::Tensor& d_out, const at::Tensor& x, const at::Tensor& u,
                                                    const at::Tensor& weight, const at::Tensor& bias,
                                                    const at::Tensor& mean, const at::Tensor& var,
                                                    const at::Tensor& mask, double eps, double dropout_ratio)
{
    at::Tensor d_u = at::empty_like(x);
    at::Tensor d_x = at::empty_like(x);
    at::Tensor d_weight = at::zeros_like(weight, at::TensorOptions().dtype(at::kFloat));
    at::Tensor d_bias = at::zeros_like(bias, at::TensorOptions().dtype(at::kFloat));

    // 调用EXEC_NPU_CMD接口，完成输出结果的计算
    EXEC_NPU_CMD(aclnnNormMultiplyDropoutGrad, d_out, x, u, weight, bias, mean, var, mask, eps, dropout_ratio, d_u, d_x,
                 d_weight, d_bias);
    return {d_x, d_u, d_weight, d_bias};
}

// 通过继承torch::autograd::Function类实现前反向绑定
class NormMultiplyDropout : public torch::autograd::Function<NormMultiplyDropout> {
public:
    static constexpr bool isTraceable = true;

    static tensor_list forward(AutogradContext* ctx, const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                               const at::Tensor& bias, double eps, double dropout_ratio)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto outs = norm_multiply_dropout_impl_npu(x, u, weight, bias, eps, dropout_ratio);
        at::Tensor result = outs[0];
        at::Tensor mean = outs[1];
        at::Tensor var = outs[2];
        at::Tensor mask = outs[3];

        ctx->save_for_backward({x, u, weight, bias, mean, var, mask, at::tensor(eps), at::tensor(dropout_ratio)});
        return {result, mean, var};
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto u = saved[1];
        auto weight = saved[2];
        auto bias = saved[3];
        auto mean = saved[4];
        auto var = saved[5];
        auto drop_mask = saved[6];
        auto eps = saved[7].item<double>();
        auto dropout_ratio = saved[8].item<double>();
        tensor_list bk_out = norm_multiply_dropout_backward_impl_npu(grad_output, x, u, weight, bias, mean, var,
                                                                     drop_mask, eps, dropout_ratio);
        return {bk_out[0], bk_out[1], bk_out[2], bk_out[3], at::Tensor(), at::Tensor()};
    }
};

// 使用的时候调用apply()方法
tensor_list norm_multiply_dropout_impl_autograd(const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                                                const at::Tensor& bias, double eps, double dropout_ratio)
{
    return NormMultiplyDropout::apply(x, u, weight, bias, eps, dropout_ratio);
}

// 注册接口
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("norm_multiply_dropout(Tensor x, Tensor u, Tensor weight, Tensor bias, float eps, float dropout_ratio) -> "
          "Tensor[]");
}

// 转发NPU实现和反向
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("norm_multiply_dropout", TORCH_FN(norm_multiply_dropout_impl_autograd));
}
TORCH_LIBRARY_IMPL(mxrec, Autograd, m)
{
    m.impl("norm_multiply_dropout", TORCH_FN(norm_multiply_dropout_impl_autograd));
}
}  // namespace mxrec_npu_api
