/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor &offset)
{
    const at::OptionalDeviceGuard guard(device_of(offset));
    auto offset_contin = offset.contiguous();
    auto output = at::empty({offset.size(0) + 1}, offset.options());

    EXEC_NPU_CMD(aclnnAsynchronousCompleteCumsum, offset_contin, output);
    return output;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("asynchronous_complete_cumsum(Tensor offset) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_npu);
}
