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
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor segment_sum_csr_impl_npu(int64_t batch_size, const at::Tensor& csr_seg, const at::Tensor& values)
{
    check_tensor_non_empty(csr_seg, "csr_seg");
    check_tensor_dim(csr_seg, 1, "segment_sum_csr csr_seg");
    check_tensor_dim(values, 1, "segment_sum_csr values");
    if (batch_size != 0) {
        TORCH_CHECK(values.size(0) % batch_size == 0, "param values dim 0: ", values.size(0),
                    " is not multiple of batch_size: ", batch_size);
    }
    auto csr_seg_conti = csr_seg.contiguous();
    auto values_conti = values.contiguous();

    // NPU kernel 仅支持浮点类型。对整数类型，先转 FP32 计算，再 cast 回原始类型。
    if (values_conti.dtype() == at::kInt || values_conti.dtype() == at::kLong) {
        auto values_fp32 = values_conti.to(at::kFloat);
        at::Tensor y_fp32 = at::empty({csr_seg_conti.size(0) - 1}, values_fp32.options());
        EXEC_NPU_CMD(aclnnSegmentSumCsr, csr_seg_conti, values_fp32, batch_size, y_fp32);
        return y_fp32.to(values_conti.dtype());
    }

    at::Tensor y = at::empty({csr_seg_conti.size(0) - 1}, values_conti.options());
    EXEC_NPU_CMD(aclnnSegmentSumCsr, csr_seg_conti, values_conti, batch_size, y);
    return y;
}

// 通过继承torch::autograd::Function类实现前向绑定
class SegmentSumCsr : public torch::autograd::Function<SegmentSumCsr> {
public:
    static at::Tensor forward(AutogradContext* ctx, int64_t batch_size, at::Tensor csr_seg, at::Tensor values)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        auto y = segment_sum_csr_impl_npu(batch_size, csr_seg, values);
        ctx->save_for_backward({csr_seg, values});
        return y;
    }
};

// 在npu命名空间里注册segment_sum_csr的schema
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("segment_sum_csr(SymInt batch_size, Tensor csr_seg, Tensor values) -> Tensor");
}

// 为NPU设备注册前向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("segment_sum_csr", &segment_sum_csr_impl_npu);
}