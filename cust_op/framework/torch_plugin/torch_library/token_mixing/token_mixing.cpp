/**
 * @file token_mixing.cpp
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
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_3D = 3;
constexpr int EXPECTED_DIM_1D = 1;
at::Tensor token_mixing_impl_npu(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta,
                                 const double epsilon)
{
    check_tensor_non_empty(x, "x");
    check_tensor_dim(x, EXPECTED_DIM_3D, "x");
    check_tensor_non_empty(gamma, "gamma");
    check_tensor_dim(gamma, EXPECTED_DIM_1D, "gamma");
    check_tensor_non_empty(beta, "beta");
    check_tensor_dim(beta, EXPECTED_DIM_1D, "beta");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {x, gamma, beta};
    std::vector<std::string> names = {"x", "gamma", "beta"};
    check_tensor_npu_device(tensors, names);
    at::Tensor x_t = x.permute({0, 2, 1});
    auto x_conti = x.contiguous();
    auto x_t_conti = x_t.contiguous();
    auto gamma_conti = gamma.contiguous();
    auto beta_conti = beta.contiguous();
    at::Tensor y = at::empty({x.size(0), x.size(2), x.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnTokenMixing, x_conti, x_t_conti, gamma_conti, beta_conti, epsilon, y);
    return y;
}

// 通过继承torch::autograd::Function类实现前向绑定
class TokenMixing : public torch::autograd::Function<TokenMixing> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor x, at::Tensor gamma, at::Tensor beta,
                              const double epsilon)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->saved_data["epsilon"] = epsilon;
        auto y = token_mixing_impl_npu(x, gamma, beta, epsilon);
        ctx->save_for_backward({x, gamma, beta});
        return y;
    }
};

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("token_mixing(Tensor x, Tensor gamma, Tensor beta, float epsilon=1e-7) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("token_mixing", &token_mixing_impl_npu);
}