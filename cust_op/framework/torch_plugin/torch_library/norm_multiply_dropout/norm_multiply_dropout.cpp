/**
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cmath>
#include <iostream>
#include <random>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/common_utils.h"
#include "../common/pytorch_npu_helper.hpp"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using torch::autograd::variable_list;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;

namespace mxrec_npu_api {
constexpr int64_t ONE_DIM = 1;
constexpr int64_t TWO_DIM = 2;
constexpr int64_t MAX_DIM0 = 1e6;
constexpr int64_t BASE_DIM1 = 512;
constexpr int64_t MAX_DIM1 = 1024;
constexpr int64_t MASK_BITS = 8;
constexpr int64_t MASK_ALIGN = 128;
constexpr double DROPOUT_RATIO_ZERO_EPS = 1e-10;
constexpr double DROPOUT_RATIO_MIN = 0.0;
constexpr double DROPOUT_RATIO_MAX = 1.0;
constexpr double EPS_MIN = 1e-10;
constexpr double EPS_MAX = 1e-4;

at::Tensor dropout_gen_mask_impl(const at::Tensor& query, double keep_prob, int64_t seed, const int64_t offset,
                                 const int64_t numels)
{
    // kernel内调用dropout时使用bit模式，生成的mask为输入shape的1/8大小的uint8类型tensor
    int64_t length = (numels + MASK_ALIGN - 1) / MASK_ALIGN * MASK_ALIGN / MASK_BITS;
    c10::TensorOptions options = query.options();
    at::Tensor mask = at::zeros(at::IntArrayRef{length}, options.dtype(at::kByte));
    c10::SmallVector<int64_t> shape_size = {numels};
    at::IntArrayRef shapeArray = at::IntArrayRef(shape_size);
    double prob;
    at::Scalar prob_scalar;
    if (query.scalar_type() == at::kHalf) {
        prob_scalar = at::Scalar(at::Half(1.0) - at::Half(keep_prob));
    } else if (query.scalar_type() == at::kBFloat16) {
        prob_scalar = at::Scalar(at::BFloat16(1.0) - at::BFloat16(keep_prob));
    } else {
        prob_scalar = at::Scalar(static_cast<float>(1.0) - static_cast<float>(keep_prob));
    }
    prob = prob_scalar.toDouble();
    aclDataType probDataType = aclDataType::ACL_FLOAT;
    EXEC_NPU_CMD(aclnnDropoutGenMaskV2, shapeArray, prob, seed, offset, probDataType, mask);
    return mask;
}

std::mt19937& rng()
{
    static std::mt19937 gen{std::random_device{}()};
    return gen;
}

tensor_list norm_multiply_dropout_impl_npu(const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                                           const at::Tensor& bias, double eps, double dropout_ratio)
{
    at::Tensor result = at::empty_like(x);
    std::vector<int64_t> x_sizes = x.sizes().vec();
    at::Tensor mean = at::empty({x_sizes[0]}, x.options().dtype(at::kFloat));
    at::Tensor var = at::empty({x_sizes[0]}, x.options().dtype(at::kFloat));

    bool isNeedDrop = std::fabs(dropout_ratio) > DROPOUT_RATIO_ZERO_EPS;
    at::Tensor drop_mask;
    if (isNeedDrop) {
        int64_t numels = x.numel();
        std::mt19937 rng(std::random_device{}());
        int64_t seed = rng();
        float keep_prob = 1 - dropout_ratio;
        drop_mask = dropout_gen_mask_impl(x, keep_prob, seed, 0, numels);
    } else {
        drop_mask = at::empty({1}, x.options().dtype(at::kByte));
    }

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
    EXEC_NPU_CMD(aclnnNormMultiplyDropoutBackward, d_out, x, u, weight, bias, mean, var, mask, eps, dropout_ratio, d_u,
                 d_x, d_weight, d_bias);
    return {d_x, d_u, d_weight, d_bias};
}

// 通过继承torch::autograd::Function类实现前反向绑定
class NormMultiplyDropout : public torch::autograd::Function<NormMultiplyDropout> {
public:
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
        return {result};
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

void check_params(const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                  const at::Tensor& bias, double eps, double dropout_ratio)
{
    check_tensor_dim(x, TWO_DIM, "norm_multiply_dropout input x");
    check_tensor_dim(u, TWO_DIM, "norm_multiply_dropout input u");
    check_tensor_dim(weight, ONE_DIM, "norm_multiply_dropout input weight");
    check_tensor_dim(bias, ONE_DIM, "norm_multiply_dropout input bias");
    auto x_dim0 = x.size(0);
    auto x_dim1 = x.size(1);
    TORCH_CHECK(x_dim0 == u.size(0), "param input x dim0: ", x_dim0, " is not equal to u dim0: ", u.size(0));
    TORCH_CHECK(x_dim1 == u.size(1), "param input x dim1: ", x_dim1, " is not equal to u dim1: ", u.size(1));
    TORCH_CHECK(x_dim1 == weight.size(0) && weight.size(0) == bias.size(0),
                "param input x dim1: ", x_dim1, ", weight dim0: ", weight.size(0),
                ", bias dim0:", bias.size(0), " must be same but not equal.");
    TORCH_CHECK(x_dim0 >= 1 && x_dim0 <= MAX_DIM0, "x dim0 must in range:[1,", MAX_DIM0, "].");
    TORCH_CHECK(x_dim1 == BASE_DIM1 || x_dim1 == MAX_DIM1, "x dim1 must equal with ", BASE_DIM1, " or ", MAX_DIM1);
    TORCH_CHECK(eps >= EPS_MIN && eps <= EPS_MAX,
                "eps must in range:[", EPS_MIN, ",", EPS_MAX, "].");
    TORCH_CHECK(dropout_ratio >= DROPOUT_RATIO_MIN && dropout_ratio <= DROPOUT_RATIO_MAX,
                "dropout_ratio must in range:[", DROPOUT_RATIO_MIN, ",", DROPOUT_RATIO_MAX, "].");
}

// 使用的时候调用apply()方法
tensor_list norm_multiply_dropout_impl_autograd(const at::Tensor& x, const at::Tensor& u, const at::Tensor& weight,
                                                const at::Tensor& bias, double eps, double dropout_ratio)
{
    check_params(x, u, weight, bias, eps, dropout_ratio);
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
