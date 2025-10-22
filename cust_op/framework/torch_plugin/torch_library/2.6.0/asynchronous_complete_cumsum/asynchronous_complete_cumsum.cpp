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
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor &offset)
{
    const at::OptionalDeviceGuard guard(device_of(offset));
    check_tensor_non_empty(offset, "offset");
    
    // 检查NPU设备（单个张量）
    std::vector<at::Tensor> tensors = {offset};
    std::vector<std::string> names = {"offset"};
    check_tensor_npu_device(tensors, names);
    
    auto offset_contin = offset.contiguous();
    int64_t offset_size = offset.size(0);
    TORCH_CHECK(offset_size > 0 && offset_size < std::numeric_limits<int64_t>::max(),
        "offset.size(0) limit (0, %lld), but get %lld\n", std::numeric_limits<int64_t>::max(), offset_size);
    auto output = at::empty({offset_size + 1}, offset.options());

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
