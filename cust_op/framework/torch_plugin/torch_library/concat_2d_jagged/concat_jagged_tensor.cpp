/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <iostream>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::Variable;
using tensor_list = std::vector<at::Tensor>;
using namespace at;
using namespace std;

constexpr int EXPECTED_DIM_2D = 2;
constexpr int MAX_OFFSET_LEN = 1024;

template <typename T>
std::vector<T> tensor_to_vector(const at::Tensor& tensor)
{
    std::vector<T> result;
    // 确保张量在cpu且内存连续
    at::Tensor tensor_cpu = tensor.is_cpu() ? tensor : tensor.cpu();
    at::Tensor tensor_cpu_contiguous = tensor_cpu.flatten().contiguous();
    // 校验dtype 匹配
    const auto& dtype = tensor_cpu_contiguous.dtype();
    if constexpr (std::is_same_v<T, int64_t>) {
        TORCH_CHECK(dtype == at::kLong, "Tensor dtype must int64 (kLong)");
    } else if constexpr (std::is_same_v<T, float>) {
        TORCH_CHECK(dtype == at::kFloat, "Tensor dtype must float32 (kFloat)");
    } else if constexpr (std::is_same_v<T, int32_t>) {
        TORCH_CHECK(dtype == at::kInt, "Tensor dtype must int32 (kInt)");
    }
    // 获取指针并拼接至vector
    const T* data_ptr = tensor_cpu_contiguous.data_ptr<T>();
    size_t elem_count = tensor_cpu_contiguous.numel();
    result.insert(result.end(), data_ptr, data_ptr + elem_count);
    return result;
}

c10::optional<at::IntArrayRef> vec_to_intarray(const std::vector<int64_t>& vec)
{
    if (!vec.empty()) {
        at::IntArrayRef arr_ref(vec);
        return c10::optional<at::IntArrayRef>(arr_ref);
    }
}

// 为NPU设备注册实现
at::Tensor concat_2d_jagged_npu(
    const int64_t &maxSeqlen,
    const Tensor &valuesA,
    const Tensor &valuesB,
    const Tensor &offsetA,
    const Tensor &offsetB,
    const bool isReplace = false,
    const int64_t nPrefixFromRight = 0)
{
    // check values same dim
    TORCH_CHECK(valuesA.size(1) == valuesB.size(1), "values must be the same dimensional.");
    // check offset dim
    TORCH_CHECK(offsetA.dim() == 1, "offsetA must be a 1-dimensional tensor.");
    TORCH_CHECK(offsetB.dim() == 1, "offsetB must be a 1-dimensional tensor.");
    // check offset size
    TORCH_CHECK(offsetA.size(0) == offsetB.size(0),
                "offsetA and offsetB must have the same length.");
    TORCH_CHECK(offsetA.size(0) >= 2 && offsetA.size(0) <= MAX_OFFSET_LEN,
                "offset must have length >= 2 and <= ", MAX_OFFSET_LEN);
    // check values size
    TORCH_CHECK(valuesA.size(0) >= offsetA[-1].item<int64_t>(),
                "The length of valuesA should be greater than the maximum value of offsetA");
    TORCH_CHECK(valuesB.size(0) >= offsetB[-1].item<int64_t>(),
                "The length of valuesB should be greater than the maximum value of offsetB");

    TORCH_CHECK(valuesA.size(0) >= 1, "values must have length >= 1.");
    // check type
    TORCH_CHECK(valuesA.dtype() == at::kFloat || valuesA.dtype() == at::kHalf || valuesA.dtype() == at::kBFloat16,
                "valuesA must have be kFloat or kHalf or kBFloat16 dtype.");
    TORCH_CHECK(valuesB.dtype() == at::kFloat || valuesB.dtype() == at::kHalf || valuesB.dtype() == at::kBFloat16,
                "valuesB must have be kFloat or kHalf or kBFloat16 dtype.");
    TORCH_CHECK(offsetA.dtype() == at::kLong || offsetA.dtype() == at::kInt,
                "offsetA must have be kLong or kInt dtype.");
    TORCH_CHECK(offsetB.dtype() == at::kLong || offsetB.dtype() == at::kInt,
                "offsetB must have be kLong or kInt dtype.");

    TORCH_CHECK(valuesA.dtype() == valuesB.dtype(), "values must have same dtype.");

    // offsetlen
    int64_t offsetlen = offsetA.size(0);

    // offset
    at::Tensor offsets = at::cat({offsetA, offsetB}, 0);
    at::Tensor offsets_int64 = offsets.to(at::kLong);
    std::vector<int64_t> offsetVector = tensor_to_vector<int64_t>(offsets_int64);
    auto offsetArray = vec_to_intarray(offsetVector);

    // tensor list
    std::vector<at::Tensor> values_vec = {valuesA, valuesB};
    at::TensorList values = at::TensorList(values_vec);
    int64_t jtNum = 2;
    // result
    int64_t resultRows = valuesA.size(0) + valuesB.size(0);
    int64_t resultCols = valuesA.size(1);
    auto result = at::empty({resultRows, resultCols}, valuesA.options());
    EXEC_NPU_CMD(aclnnConcatJaggedTensor, values, offsetArray, offsetlen, jtNum, result);
    return result;
}


tuple<Tensor, Tensor> split_2d_jagged_npu(
    const Tensor &values,
    const SymInt &maxSeqlen,
    const Tensor &offsetA,
    const Tensor &offsetB,
    const SymInt dense_size = 0,
    const SymInt nPrefixToRight = 0)
{
    // check offset
    TORCH_CHECK(offsetA.dim() == 1, "offsetA must be a 1-dimensional tensor.");
    TORCH_CHECK(offsetB.dim() == 1, "offsetB must be a 1-dimensional tensor.");
    // check offset size
    TORCH_CHECK(offsetA.size(0) == offsetB.size(0),
                "offsetA and offsetB must have the same length.");
    TORCH_CHECK(offsetA.size(0) >= 2 && offsetA.size(0) <= MAX_OFFSET_LEN,
                "offset must have length >= 2 and <= ", MAX_OFFSET_LEN);

    // check values
    TORCH_CHECK(values.dim() == EXPECTED_DIM_2D, "values must be a 2-dimensional tensor.");

    // check type
    TORCH_CHECK(values.dtype() == at::kFloat || values.dtype() == at::kHalf || values.dtype() == at::kBFloat16,
                "values must have be kFloat or kHalf or kBFloat16 dtype.");
    TORCH_CHECK(offsetA.dtype() == at::kLong || offsetA.dtype() == at::kInt,
                "offsetA must have be kLong or kInt dtype.");
    TORCH_CHECK(offsetB.dtype() == at::kLong || offsetB.dtype() == at::kInt,
                "offsetB must have be kLong or kInt dtype.");

    // offsetlen
    int64_t offsetlen = offsetA.size(0);

    // offset
    at::Tensor offsets = at::cat({offsetA, offsetB}, 0);
    at::Tensor offsets_int64 = offsets.to(at::kLong);
    std::vector<int64_t> offsetVector = tensor_to_vector<int64_t>(offsets_int64);
    auto offsetArray = vec_to_intarray(offsetVector);

    int64_t jtNum = 2;
    // outputs
    int64_t RowsA = offsetA[-1].item<int64_t>();
    int64_t RowsB = offsetB[-1].item<int64_t>();
    int64_t resultCols = values.size(1);
    tensor_list outputs;
    auto outputA = at::empty({RowsA, resultCols}, values.options());
    auto outputB = at::empty({RowsB, resultCols}, values.options());
    outputs.push_back(outputA);
    outputs.push_back(outputB);
    at::TensorList outputs_list = at::TensorList(outputs);
    EXEC_NPU_CMD(aclnnConcatJaggedTensorGrad, values, offsetArray, offsetlen, jtNum, outputs_list);

    return make_tuple(outputA, outputB);
}

// 通过继承torch::autograd::funcation类实现前反向绑定
class ConcatJaggedFunction : public torch::autograd::Function<ConcatJaggedFunction> {
public:
    static at::Tensor forward(AutogradContext* ctx,
                              const int64_t &maxSeqlen,
                              const Tensor &valuesA,
                              const Tensor &valuesB,
                              const Tensor &offsetA,
                              const Tensor &offsetB,
                              const bool isReplace = false,
                              const int64_t nPrefixFromRight = 0)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({offsetA, offsetB});
        ctx->saved_data["maxSeqlen"] = maxSeqlen;
        return concat_2d_jagged_npu(maxSeqlen, valuesA, valuesB, offsetA, offsetB, isReplace, nPrefixFromRight);
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        at::Tensor offsetA = saved[0];
        at::Tensor offsetB = saved[1];
        const int64_t maxSeqlen = ctx->saved_data["maxSeqlen"].toInt();
        auto grad_output = grad_outputs[0];
        auto tensors = split_2d_jagged_npu(grad_output, maxSeqlen, offsetA, offsetB, 0, 0);
        at::Tensor tensor_a = std::get<0>(tensors);
        at::Tensor tensor_b = std::get<1>(tensors);
        return {Variable(), tensor_a, tensor_b, Variable(), Variable(), Variable(), Variable()};
    }
};

class SplitJaggedFunction : public torch::autograd::Function<SplitJaggedFunction> {
public:
    static std::vector<at::Tensor> forward(AutogradContext* ctx,
                                         const Tensor &values,
                                         const SymInt &maxSeqlen,
                                         const Tensor &offsetA,
                                         const Tensor &offsetB,
                                         const SymInt dense_size = 0,
                                         const SymInt nPrefixToRight = 0)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({offsetA, offsetB});
        ctx->saved_data["maxSeqlen"] = maxSeqlen;
        auto tensors = split_2d_jagged_npu(values, maxSeqlen, offsetA, offsetB, dense_size, nPrefixToRight);
        at::Tensor tensor_a = std::get<0>(tensors);
        at::Tensor tensor_b = std::get<1>(tensors);
        return {tensor_a, tensor_b};
    }

    static torch::autograd::variable_list backward(AutogradContext* ctx, torch::autograd::variable_list grad_outputs)
    {
        auto tensor_back_a = grad_outputs[0];
        auto tensor_back_b = grad_outputs[1];
        auto saved = ctx->get_saved_variables();
        at::Tensor offsetA = saved[0];
        at::Tensor offsetB = saved[1];
        const int64_t maxSeqlen = ctx->saved_data["maxSeqlen"].toInt();
        auto concat_tensor = concat_2d_jagged_npu(maxSeqlen, tensor_back_a, tensor_back_b, offsetA, offsetB, false, 0);
        return {concat_tensor, Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

// 使用的时候调用apply()方法
at::Tensor concat_2d_jagged_autograd(
    const int64_t &maxSeqlen,
    const Tensor &valuesA,
    const Tensor &valuesB,
    const Tensor &offsetA,
    const Tensor &offsetB,
    const bool isReplace = false,
    const int64_t nPrefixFromRight = 0)
    {
        return ConcatJaggedFunction::apply(maxSeqlen, valuesA, valuesB, offsetA, offsetB, false, 0);
    }


tuple<Tensor, Tensor> split_2d_jagged_autograd(
    const Tensor &values,
    const SymInt &maxSeqlen,
    const Tensor &offsetA,
    const Tensor &offsetB,
    const SymInt dense_size = 0,
    const SymInt nPrefixToRight = 0)
    {
        auto result = SplitJaggedFunction::apply(values, maxSeqlen, offsetA, offsetB, 0, 0);
        return {result[0], result[1]};
    }


// 在npu命名空间里注册concat_2d_jagged
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("concat_2d_jagged(SymInt maxSeqlen, "
          "                 Tensor valuesA, "
          "                 Tensor valuesB, "
          "                 Tensor offsetA, "
          "                 Tensor offsetB, "
          "                 bool isReplace = False, "
          "                 SymInt nPrefixFromRight = 0) -> Tensor ");
    m.def("split_2d_jagged(Tensor values, "
          "                SymInt maxSeqlen, "
          "                Tensor offsetA, "
          "                Tensor offsetB, "
          "                SymInt dense_size = 0, "
          "                SymInt nPrefixToRight = 0) -> (Tensor, Tensor) ");
}


// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("concat_2d_jagged", &concat_2d_jagged_npu);
    m.impl("split_2d_jagged", &split_2d_jagged_npu);
}

// 注册自动求导实现
TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("concat_2d_jagged", &concat_2d_jagged_autograd);
    m.impl("split_2d_jagged", &split_2d_jagged_autograd);
}
