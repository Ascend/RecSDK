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
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int EXPECTED_DIM_1D = 1;
constexpr int EXPECTED_DIM_2D = 2;
constexpr int EXPECTED_DIM_3D = 3;

at::Tensor jagged_to_padded_dense_forward_npu(const at::Tensor& values,
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
    
    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {values, offset_tensor};
    std::vector<std::string> names = {"values", "offset_tensor"};
    check_tensor_npu_device(tensors, names);
    
    TORCH_CHECK(max_lengths > 0, "max_lengths must be positive, but got ", max_lengths);

    const at::OptionalDeviceGuard guard(device_of(values));
    auto values_contin = values.contiguous();
    auto D = values.size(-1);
    auto output =
        at::empty({offsets[0].size(0) - 1, max_lengths, values.size(1)}, values.options());
    int64_t padding_value_int64 = static_cast<int64_t>(padding_value);
    EXEC_NPU_CMD(aclnnJaggedToPaddedDense, values_contin, offsets[0], max_lengths,
        padding_value, padding_value_int64, output);
    return output;
};

// 目前只支持3维的dense
at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense,
                                       const tensor_list& offsets,
                                       const c10::optional<int64_t> total_L)
{
    check_tensor_dim(dense, EXPECTED_DIM_3D, "dense");
    TORCH_CHECK(offsets.size() == 1,
        "Only single-dimension jagged tensors supported (offsets.size() must be 1)");

    const auto& offset_tensor = offsets[0];
    
    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {dense, offset_tensor};
    std::vector<std::string> names = {"dense", "offset_tensor"};
    check_tensor_npu_device(tensors, names);

    const at::OptionalDeviceGuard guard(device_of(dense));
    auto D = dense.size(-1);
    auto dense_contin = dense.contiguous();

    check_tensor_non_empty(offsets.back(), "offsets.back()");
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
    return output;
};

std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu(const at::Tensor& dense,
                                                        const tensor_list& offsets,
                                                        const c10::optional<int64_t> total_L)
{
    return {dense_to_jagged_forward_npu(dense, offsets, total_L), offsets};
};

// 反向算子 - 使用jagged_to_padded_dense作为反向
at::Tensor dense_to_jagged_backward_npu(const at::Tensor& values,
                                        const tensor_list& offsets,
                                        const int64_t max_lengths,
                                        const double padding_value)
{
    return jagged_to_padded_dense_forward_npu(values, offsets, max_lengths, padding_value);
};

// 自动求导Function类
class DenseToJaggedFunction : public torch::autograd::Function<DenseToJaggedFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                              const at::Tensor& dense,
                              const tensor_list& offsets,
                              const c10::optional<int64_t> total_L)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({dense, offsets[0]});

        return dense_to_jagged_forward_npu(dense, offsets, total_L);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto dense = saved[0];
        auto offsets_tensor = saved[1];

        tensor_list offsets = {offsets_tensor};
        int64_t maxLen = dense.size(1);

        // 调用jagged_to_padded_dense作为反向
        auto grad_dense = dense_to_jagged_backward_npu(
            grad_output, offsets, maxLen, 0.0);

        // 返回梯度：grad_dense, None, None
        return {grad_dense, Variable(), Variable()};
    }
};

// 自动求导接口
at::Tensor dense_to_jagged_autograd(const at::Tensor& dense,
                                    const tensor_list& offsets,
                                    const c10::optional<int64_t> total_L)
{
    return DenseToJaggedFunction::apply(dense, offsets, total_L);
}

std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu_autograd(const at::Tensor& dense,
                                                                 const tensor_list& offsets,
                                                                 const c10::optional<int64_t> total_L)
{
    return {dense_to_jagged_autograd(dense, offsets, total_L), offsets};
};

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("dense_to_jagged_forward(Tensor dense, "
          "                        Tensor[] offsets, "
          "                        SymInt? total_L=None) -> Tensor");

    m.def("dense_to_jagged(Tensor dense, "
          "                Tensor[] offsets, "
          "                SymInt? total_L=None) -> (Tensor, Tensor[])");

    m.def("dense_to_jagged_backward(Tensor values, "
          "                         Tensor[] offsets, "
          "                         int max_lengths, "
          "                         float padding_value) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
    m.impl("dense_to_jagged_backward", &dense_to_jagged_backward_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("dense_to_jagged_forward", &dense_to_jagged_forward_npu);
    m.impl("dense_to_jagged", &dense_to_jagged_npu);
    m.impl("dense_to_jagged_backward", &dense_to_jagged_backward_npu);
}

// 注册自动求导实现
TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("dense_to_jagged", &dense_to_jagged_npu_autograd);
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradPrivateUse1, m)
{
    m.impl("dense_to_jagged", &dense_to_jagged_npu_autograd);
}