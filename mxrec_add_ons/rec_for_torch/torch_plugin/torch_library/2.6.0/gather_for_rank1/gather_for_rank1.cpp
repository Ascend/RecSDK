/**
 * @file gather_for_rank1.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// 为NPU设备注册前向实现
at::Tensor gather_for_rank1_impl_npu(const at::Tensor& x, const at::Tensor& index)
{
    TORCH_CHECK(x.dim() == 1, "The x should be 1D");
    TORCH_CHECK(index.dim() == 1, "The index should be 1D");
    auto xConti = x.contiguous();
    auto indexConti = index.contiguous();
    at::Tensor y = at::empty({indexConti.size(0)}, xConti.options());
    EXEC_NPU_CMD(aclnnGatherForRank1, xConti, indexConti, y);
    return y;
}

// 为NPU设备注册反向实现
tensor_list gather_for_rank1_backward_impl_npu(
    const at::Tensor& gradY,
    const at::Tensor& x,
    const at::Tensor& index)
{
    auto dense_gradY = gradY.contiguous();
    auto dense_index = index.contiguous();
    at::Tensor gradX = at::zeros_like(x);
    at::Tensor gradIndex = at::zeros_like(index);
    EXEC_NPU_CMD(aclnnIndexSelectForRank1Backward, dense_gradY, x, dense_index, gradX, gradIndex);
    return {gradX, gradIndex};
}

// 通过继承torch::autograd::Function类实现前反向绑定
class GatherForRank1 : public torch::autograd::Function<GatherForRank1> {
public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor x, at::Tensor index)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({x, index});
        return gather_for_rank1_impl_npu(x, index);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto index = saved[1];
        return gather_for_rank1_backward_impl_npu(grad_output, x, index);
    }
};

// 使用的时候调用apply()方法
at::Tensor gather_for_rank1_impl_autograd(const at::Tensor& x, const at::Tensor& index)
{
    return GatherForRank1::apply(x, index);
}

// 在npu命名空间里注册gather_for_rank1和gather_for_rank1_backward两个schema
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("gather_for_rank1(Tensor x, Tensor index) -> Tensor");
    m.def("index_select_for_rank1_backward(Tensor grad, Tensor x, Tensor index) -> Tensor[]");
}

// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("gather_for_rank1", &gather_for_rank1_impl_npu);
    m.impl("index_select_for_rank1_backward", &gather_for_rank1_backward_impl_npu);
}

// 给op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("gather_for_rank1", &gather_for_rank1_impl_autograd);
}
