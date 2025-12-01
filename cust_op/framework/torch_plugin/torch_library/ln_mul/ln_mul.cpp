/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// 为NPU设备注册前向实现
at::Tensor ln_mul_impl_npu(const at::Tensor& x, const at::Tensor& u, const at::Tensor& gamma, const at::Tensor& beta)
{
    check_tensor_non_empty(x, "x");
    check_tensor_non_empty(u, "u");
    check_tensor_non_empty(gamma, "gamma");
    check_tensor_non_empty(beta, "beta");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {x, u, gamma, beta};
    std::vector<std::string> names = {"x", "u", "gamma", "beta"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(u.dim() == 2, "The u should be 2D");
    TORCH_CHECK(gamma.dim() == 1, "The gamma should be 1D");
    TORCH_CHECK(beta.dim() == 1, "The beta should be 1D");
    TORCH_CHECK(x.sizes() == u.sizes(), "x and u must have identical shape");

    auto x_conti = x.contiguous();
    auto u_conti = u.contiguous();
    auto gamma_conti = gamma.contiguous();
    auto beta_conti = beta.contiguous();
    at::Tensor y = at::empty_like(x_conti);
    EXEC_NPU_CMD(aclnnLnMul, x_conti, u_conti, gamma_conti, beta_conti, y);
    return y;
}

// 通过继承torch::autograd::Function类实现前向绑定
class LnMul : public torch::autograd::Function<LnMul> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor x, at::Tensor u, at::Tensor gamma, at::Tensor beta)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = ln_mul_impl_npu(x, u, gamma, beta);
        ctx->save_for_backward({x, u, gamma, beta});
        return y;
    }
};

// 在npu命名空间里注册ln_mul的schema
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("ln_mul(Tensor x, Tensor u, Tensor gamma, Tensor beta) -> Tensor");
}

// 为NPU设备注册前向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("ln_mul", &ln_mul_impl_npu);
}