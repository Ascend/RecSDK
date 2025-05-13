/**
* @file jagged_to_padded_dense.cpp
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

namespace fbgemm_npu {
at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense,
                                       const tensor_list& offsets,
                                       const c10::optional<int64_t>& total_L)
{
    const at::OptionalDeviceGuard guard(device_of(dense));
    auto D = dense.size(-1);
    auto dense_contin = dense.contiguous();

    int64_t totalLComputed;
    if (total_L.has_value()) {
        totalLComputed = total_L.value();
    } else {
        totalLComputed = (int64_t)offsets.back().max().item<int64_t>();
    }

    auto output = at::empty({totalLComputed, D}, dense.options());
    EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets[0], totalLComputed, output);
    return output;
};
at::Tensor jagged_to_padded_dense_forward_npu(const at::Tensor& values,
                                              const tensor_list& offsets,
                                              const int64_t max_lengths,
                                              const double padding_value)
{
    const at::OptionalDeviceGuard guard(device_of(values));
    auto values_contin = values.contiguous();
    auto D = values.size(-1);
    auto output =
        at::full({offsets[0].size(0) - 1, max_lengths, values.size(1)}, padding_value, values.options());
    EXEC_NPU_CMD(aclnnJaggedToPaddedDense, values_contin, offsets[0], max_lengths, padding_value, output);
    return output;
};

at::Tensor jagged_to_padded_dense_backward_npu(const at::Tensor& grad_output,
                                               const tensor_list& offsets,
                                               const int64_t total_L)
{
    return dense_to_jagged_forward_npu(grad_output, offsets, total_L);
};

at::Tensor jagged_to_padded_dense_npu(const at::Tensor& values,
                                      const tensor_list& offsets,
                                      const int64_t max_lengths,
                                      const double padding_value)
{
    return jagged_to_padded_dense_forward_npu(values, offsets, max_lengths, padding_value);
};

}  // namespace fbgemm_npu

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("jagged_to_padded_dense(Tensor values, \
                                  Tensor[] offsets, \
                                  int max_lengths, \
                                  float padding_value) -> Tensor");
    m.def("jagged_to_padded_dense_forward(Tensor values, \
                                          Tensor[] offsets, \
                                          int max_lengths, \
                                          float padding_value) -> Tensor");
    m.def("jagged_to_padded_dense_backward(Tensor grad, Tensor[] offsets, int total_L) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("jagged_to_padded_dense", &fbgemm_npu::jagged_to_padded_dense_npu);
    m.impl("jagged_to_padded_dense_forward", &fbgemm_npu::jagged_to_padded_dense_forward_npu);
    m.impl("jagged_to_padded_dense_backward", &fbgemm_npu::jagged_to_padded_dense_backward_npu);
}
