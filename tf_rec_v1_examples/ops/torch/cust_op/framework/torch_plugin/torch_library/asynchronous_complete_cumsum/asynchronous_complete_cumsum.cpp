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

at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor &offset)
{
    const at::OptionalDeviceGuard guard(device_of(offset));

    // 检查NPU设备（单个张量）
    std::vector<at::Tensor> tensors = {offset};
    std::vector<std::string> names = {"offset"};
    check_tensor_npu_device(tensors, names);

    auto offset_contin = offset.contiguous();
    int64_t offset_size = offset.size(0);
    TORCH_CHECK(offset_size >= 0 && offset_size < std::numeric_limits<int64_t>::max(),
                "offset.size(0) limit [0, ", std::numeric_limits<int64_t>::max(), "), but get ", offset_size, "\n");
    auto output = at::empty({offset_size + 1}, offset.options());
    if (offset_size == 0) {
        output.zero_();
        return output;
    }

    EXEC_NPU_CMD(aclnnAsynchronousCompleteCumsum, offset_contin, output);
    return output;
}

at::Tensor asynchronous_inclusive_cumsum_npu(const at::Tensor &offset)
{
    if (offset.numel() == 0) {
        return at::empty_like(offset);
    }
    auto complete_result = asynchronous_complete_cumsum_npu(offset);
    return complete_result.narrow(0, 1, complete_result.size(0) - 1);
}

at::Tensor asynchronous_exclusive_cumsum_npu(const at::Tensor &offset)
{
    if (offset.numel() == 0) {
        return at::empty_like(offset);
    }
    auto complete_result = asynchronous_complete_cumsum_npu(offset);
    return complete_result.narrow(0, 0, complete_result.size(0) - 1);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("asynchronous_inclusive_cumsum(Tensor offset) -> Tensor");
    m.def("asynchronous_exclusive_cumsum(Tensor offset) -> Tensor");
    m.def("asynchronous_complete_cumsum(Tensor offset) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("asynchronous_inclusive_cumsum", &asynchronous_inclusive_cumsum_npu);
    m.impl("asynchronous_exclusive_cumsum", &asynchronous_exclusive_cumsum_npu);
    m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("asynchronous_inclusive_cumsum", &asynchronous_inclusive_cumsum_npu);
    m.impl("asynchronous_exclusive_cumsum", &asynchronous_exclusive_cumsum_npu);
    m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_npu);
}
