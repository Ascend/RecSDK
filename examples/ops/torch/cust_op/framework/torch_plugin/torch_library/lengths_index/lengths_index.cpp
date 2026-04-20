/**
* Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
#include "../asynchronous_complete_cumsum/asynchronous_complete_cumsum.h"
#include <cstdio>

at::Tensor lengths_index_npu(const at::Tensor &lengths,
                             const c10::optional<int64_t> &shape)
{
    const at::OptionalDeviceGuard guard(device_of(lengths));

    // 检查输入张量在 NPU 设备上
    std::vector<at::Tensor> tensors = {lengths};
    std::vector<std::string> names = {"lengths"};
    check_tensor_npu_device(tensors, names);

    auto lengths_contig = lengths.contiguous();
    int64_t num_seq = lengths_contig.size(0);

    // 计算 output_size
    int64_t output_size;
    at::Tensor offsets;


    if (shape.has_value()) {
        // 有 shape：用 exclusive_cumsum，output_size = shape
        output_size = shape.value();
        TORCH_CHECK(output_size >= 0, "shape must be >= 0, but got ", output_size);
        offsets = asynchronous_exclusive_cumsum_npu(lengths_contig);
    } else {
        // 无 shape：用 complete_cumsum，从 offsets[-1] 获取 output_size
        offsets = asynchronous_complete_cumsum_npu(lengths_contig);
        output_size = offsets[num_seq].item<int64_t>();
    }


    // 创建输出张量
    auto output = at::empty({output_size}, lengths.options());

    // 如果 output_size 为 0，直接返回空张量
    if (output_size == 0) {
        return output;
    }

    // 调用 NPU 算子 aclnnLengthsIndex
    // 参数: offsets, output_size, num_seq, output
    EXEC_NPU_CMD(aclnnLengthsIndex, offsets, output_size, num_seq, output);
    std::cout << "output=" << output << std::endl;

    return output;
}


// 注册算子到 fbgemm 和 mxrec 命名空间
// 支持两种形式: lengths_index(lengths) 或 lengths_index(lengths, shape)
// shape 可以是 int 标量或 None
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
m.def("lengths_index(Tensor lengths, int? shape = None) -> Tensor");
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
m.def("lengths_index(Tensor lengths, int? shape = None) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
m.impl("lengths_index", &lengths_index_npu);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
m.impl("lengths_index", &lengths_index_npu);
}