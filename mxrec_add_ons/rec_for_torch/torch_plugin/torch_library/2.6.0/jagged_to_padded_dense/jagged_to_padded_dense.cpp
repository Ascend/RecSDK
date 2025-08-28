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
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_1D = 1;
constexpr int EXPECTED_DIM_2D = 2;

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

at::Tensor jagged_to_padded_dense_forward_npu_v1(const at::Tensor& values,
                                                 const tensor_list& offsets,
                                                 const int64_t max_lengths,
                                                 const double padding_value)
{
    check_tensor_dim(values, EXPECTED_DIM_2D, "values");
    TORCH_CHECK(offsets.size() == 1,
        "offsets must contain exactly 1 tensor, but got ", offsets.size(), " tensors");

    const auto& offset_tensor = offsets[0];
    check_tensor_non_empty(offset_tensor, "offset_tensor");
    check_tensor_dim(offset_tensor, EXPECTED_DIM_1D, "offset_tensor");
    TORCH_CHECK(max_lengths > 0, "max_lengths must be positive, but got ", max_lengths);

    const at::OptionalDeviceGuard guard(device_of(values));
    auto values_contin = values.contiguous();
    auto D = values.size(-1);
    auto output =
        at::full({offsets[0].size(0) - 1, max_lengths, values.size(1)}, padding_value, values.options());
    EXEC_NPU_CMD(aclnnJaggedToPaddedDense, values_contin, offsets[0], max_lengths, padding_value, output);
    return output;
};

at::Tensor jagged_to_padded_dense_npu_v1(const at::Tensor& values,
                                         const tensor_list& offsets,
                                         const int64_t max_lengths,
                                         const double padding_value)
{
    return jagged_to_padded_dense_forward_npu_v1(values, offsets, max_lengths, padding_value);
};

at::Tensor jagged_to_padded_dense_forward_npu_v2(const at::Tensor& values,
                                                 const tensor_list& offsets,
                                                 const at::IntArrayRef max_lengths,
                                                 const double padding_value)
{
    // 1. 检查 max_lengths 必须只有一个元素
    TORCH_CHECK(
        max_lengths.size() == 1,
        "max_lengths must contain exactly one element, but got ", max_lengths.size()
    );

    // 2. 提取第一个元素（转为 int64_t）
    const int64_t max_length = max_lengths[0];
    return jagged_to_padded_dense_forward_npu_v1(values, offsets, max_length, padding_value);
};

at::Tensor jagged_to_padded_dense_npu_v2(const at::Tensor& values,
                                         const tensor_list& offsets,
                                         const at::IntArrayRef max_lengths,
                                         const double padding_value)
{
    return jagged_to_padded_dense_forward_npu_v2(values, offsets, max_lengths, padding_value);
};

at::Tensor jagged_to_padded_dense_backward_npu(const at::Tensor& grad_output,
                                               const tensor_list& offsets,
                                               const int64_t total_L)
{
    return dense_to_jagged_forward_npu(grad_output, offsets, total_L);
};

// 自动求导Function类
class JaggedToPaddedDenseV1 : public torch::autograd::Function<JaggedToPaddedDenseV1> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                             const at::Tensor& values,
                             const tensor_list& offsets,
                             const int64_t max_lengths,
                             const double padding_value)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({values, offsets[0]});
        ctx->saved_data["max_lengths"] = max_lengths;
        ctx->saved_data["padding_value"] = padding_value;
        return jagged_to_padded_dense_forward_npu_v1(values, offsets, max_lengths, padding_value);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto values = saved[0];
        auto offsets_tensor = saved[1];
        tensor_list offsets = {offsets_tensor};

        int64_t totalL = values.size(0);
        auto grad_input = jagged_to_padded_dense_backward_npu(grad_output, offsets, totalL);
        return {grad_input, Variable(), Variable(), Variable()};
    }
};

class JaggedToPaddedDenseV2 : public torch::autograd::Function<JaggedToPaddedDenseV2> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                             const at::Tensor& values,
                             const tensor_list& offsets,
                             const at::IntArrayRef max_lengths,
                             const double padding_value)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({values, offsets[0]});
        ctx->saved_data["max_lengths"] = max_lengths[0];  // 保存第一个元素
        ctx->saved_data["padding_value"] = padding_value;
        return jagged_to_padded_dense_forward_npu_v2(values, offsets, max_lengths, padding_value);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto values = saved[0];
        auto offsets_tensor = saved[1];
        tensor_list offsets = {offsets_tensor};

        int64_t totalL = values.size(0);
        auto grad_input = jagged_to_padded_dense_backward_npu(grad_output, offsets, totalL);
        return {grad_input, Variable(), Variable(), Variable()};
    }
};

// 自动求导接口
at::Tensor jagged_to_padded_dense_npu_v1_autograd(const at::Tensor& values,
                                                  const tensor_list& offsets,
                                                  const int64_t max_lengths,
                                                  const double padding_value)
{
    return JaggedToPaddedDenseV1::apply(values, offsets, max_lengths, padding_value);
}

at::Tensor jagged_to_padded_dense_npu_v2_autograd(const at::Tensor& values,
                                                  const tensor_list& offsets,
                                                  const at::IntArrayRef max_lengths,
                                                  const double padding_value)
{
    return JaggedToPaddedDenseV2::apply(values, offsets, max_lengths, padding_value);
}

}  // namespace fbgemm_npu

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("jagged_to_padded_dense.v1(Tensor values, "
          "                          Tensor[] offsets, "
          "                          int max_lengths, "
          "                          float padding_value) -> Tensor");
    // 新增int[]的max_lengths
    m.def("jagged_to_padded_dense.v2(Tensor values, "
          "                          Tensor[] offsets, "
          "                          int[] max_lengths, "
          "                          float padding_value) -> Tensor");

    m.def("jagged_to_padded_dense_forward.v1(Tensor values, "
          "                                  Tensor[] offsets, "
          "                                  int max_lengths, "
          "                                  float padding_value) -> Tensor");
    // 新增int[]的max_lengths
    m.def("jagged_to_padded_dense_forward.v2(Tensor values, "
          "                                  Tensor[] offsets, "
          "                                  int[] max_lengths, "
          "                                  float padding_value) -> Tensor");

    m.def("jagged_to_padded_dense_backward(Tensor grad, "
          "                                Tensor[] offsets, "
          "                                int total_L) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("jagged_to_padded_dense.v1",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v1)));
    m.impl("jagged_to_padded_dense.v2",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v2)));
    m.impl("jagged_to_padded_dense_forward.v1",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_forward_npu_v1)));
    m.impl("jagged_to_padded_dense_forward.v2",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_forward_npu_v2)));
    m.impl("jagged_to_padded_dense_backward", &fbgemm_npu::jagged_to_padded_dense_backward_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("jagged_to_padded_dense.v1",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v1)));
    m.impl("jagged_to_padded_dense.v2",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v2)));
    m.impl("jagged_to_padded_dense_forward.v1",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_forward_npu_v1)));
    m.impl("jagged_to_padded_dense_forward.v2",
        torch::dispatch(c10::DispatchKey::PrivateUse1,
                      TORCH_FN(fbgemm_npu::jagged_to_padded_dense_forward_npu_v2)));
    m.impl("jagged_to_padded_dense_backward", &fbgemm_npu::jagged_to_padded_dense_backward_npu);
}

// 注册自动求导实现
TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("jagged_to_padded_dense.v1", TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v1_autograd));
    m.impl("jagged_to_padded_dense.v2", TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v2_autograd));
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("jagged_to_padded_dense.v1", TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v1_autograd));
    m.impl("jagged_to_padded_dense.v2", TORCH_FN(fbgemm_npu::jagged_to_padded_dense_npu_v2_autograd));
}