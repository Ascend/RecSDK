/**
 * @file dense_to_jagged.cpp
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
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// 目前只支持3维的dense
std::tuple<at::Tensor, tensor_list> dense_to_jagged_forward_npu(const at::Tensor& dense,
    const tensor_list& offsets, const c10::optional<int64_t> total_L)
{
    TORCH_CHECK(dense.dim() == 3,
        "dense must be 3-dimensional (B, MaxT, D), but got ", dense.dim(), "D input");
    TORCH_CHECK(offsets.size() == 1,
        "Only single-dimension jagged tensors supported (offsets.size() must be 1)");

    const at::OptionalDeviceGuard guard(device_of(dense));
    auto D = dense.size(-1);
    auto dense_contin = dense.contiguous();

    // 从offsets计算预期的total_L
    int64_t expected_total_L = offsets.back()[-1].item<int64_t>();

    // 校验输入的total_L
    if (total_L.has_value()) {
        TORCH_CHECK(
            total_L.value() == expected_total_L,
            "total_L (", total_L.value(), ") does not match the value calculated from offsets (",
            expected_total_L, ")"
        );
    }

    int64_t totalLength = total_L.value_or(expected_total_L);
    auto output = at::empty({totalLength, D}, dense.options());
    EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets[0], totalLength, output);
    return {output, offsets};
};

std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu(const at::Tensor& dense,
                                                        const tensor_list& offsets,
                                                        const c10::optional<int64_t> total_L)
{
    return dense_to_jagged_forward_npu(dense, offsets, total_L);
};

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("dense_to_jagged_forward(Tensor dense, "
          "                        Tensor[] offsets, "
          "                        SymInt? total_L=None) -> (Tensor, Tensor[])");

    m.def("dense_to_jagged(Tensor dense, "
          "                Tensor[] offsets, "
          "                SymInt? total_L=None) -> (Tensor, Tensor[])");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
}