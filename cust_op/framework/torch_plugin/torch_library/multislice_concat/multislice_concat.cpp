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
using namespace at;
using namespace std;

/**
 * 验证multislice_concat的输入参数
 * @param input 输入数据
 * @param concat_num 该算子的输出数据个数
 * @param concat_size 每个输出由多少个slice数据concat组成
 * @param slice_begin 每个slice数据的起始位置
 * @param slice_length 每个slice数据的长度，数据个数
 */
void validate_multislice_concat_inputs(const Tensor& input, const int64_t concat_num, const IntArrayRef& concat_size,
                                       const IntArrayRef& slice_begin, const IntArrayRef& slice_length)
{
    // input校验
    constexpr size_t INPUT_TENSOR_DIM = 2;
    check_tensor_non_empty(input, "input");
    check_tensor_dim(input, INPUT_TENSOR_DIM, "input");
    TORCH_CHECK(input.size(0) <= std::numeric_limits<uint16_t>::max(), "input tensor row[", input.size(0),
                "] must be <= ", std::numeric_limits<uint16_t>::max());
    TORCH_CHECK(input.size(1) <= std::numeric_limits<uint16_t>::max(), "input tensor column[", input.size(1),
                "] must be <= ", std::numeric_limits<uint16_t>::max());
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::Float ||
                    input.scalar_type() == at::ScalarType::BFloat16,
                "input tensor type must be half/float32/bfloat16");

    // concat_num校验
    constexpr uint16_t MAX_CONCAT_TENSOR_NUM = 256;
    TORCH_CHECK((concat_num > 0 && concat_num <= MAX_CONCAT_TENSOR_NUM), "concat_num[", concat_num, "]  must be (0, ",
                MAX_CONCAT_TENSOR_NUM, "]");

    // concat_size校验
    TORCH_CHECK(!concat_size.empty(), "concat_size must be non-empty");
    TORCH_CHECK(concat_size.size() >= concat_num, "length of concatSize [", concat_size.size(),
                "] must be >= concat_num[", concat_num, "]");

    // slice_begin校验
    TORCH_CHECK(!slice_begin.empty(), "slice_begin must be non-empty");

    // slice_length校验
    TORCH_CHECK(!slice_length.empty(), "slice_length must be non-empty");
}

/**
 * multislice_concat算子的NPU实现
 * @param input 输入数据
 * @param concat_num 该算子的输出数据个数
 * @param concat_size 每个输出由多少个slice数据concat组成
 * @param slice_begin 每个slice数据的起始位置
 * @param slice_length 每个slice数据的长度，数据个数
 * @return 多个输出数据
 */
std::vector<at::Tensor> multislice_concat_impl_npu(const Tensor& input, const int64_t concat_num,
                                                   const IntArrayRef& concat_size, const IntArrayRef& slice_begin,
                                                   const IntArrayRef& slice_length)
{
    // 输入校验
    validate_multislice_concat_inputs(input, concat_num, concat_size, slice_begin, slice_length);

    // 确保张量是连续的(减少NPU内核中的内存访问开销)
    auto input_conti = input.contiguous();
    const auto row_num = input_conti.size(0);
    const auto column_num = input_conti.size(1);

    std::vector<at::Tensor> outputs;
    outputs.reserve(concat_num);

    int32_t slice_offset = 0;
    int32_t output_column = 0;
    size_t slice_begin_size = slice_begin.size();
    size_t slice_length_size = slice_length.size();
    constexpr int32_t MAX_SLICE_NUM = 3600;
    for (int32_t i = 0; i < concat_num; i++) {
        int32_t cur_concat_size = concat_size[i];
        TORCH_CHECK(slice_offset + cur_concat_size <= MAX_SLICE_NUM, "all slice num must be <= ", MAX_SLICE_NUM);
        TORCH_CHECK(cur_concat_size > 0 && cur_concat_size <= MAX_SLICE_NUM, "concat_size[", i, "]=", concat_size[i],
                    " must be (0, ", MAX_SLICE_NUM, "]");
        TORCH_CHECK(slice_begin_size >= slice_offset + cur_concat_size, "length of slice_begin[", slice_begin_size,
                    "] must be >= all slice num");
        TORCH_CHECK(slice_length_size >= slice_offset + cur_concat_size, "length of slice_begin[", slice_length_size,
                    "] must be >= all slice num");
        output_column = 0;
        for (int32_t concat_index = 0; concat_index < cur_concat_size; concat_index++) {
            auto cur_slice_length = slice_length[slice_offset + concat_index];
            output_column +=
                cur_slice_length != -1 ? cur_slice_length : column_num - slice_begin[slice_offset + concat_index];
        }
        at::Tensor output_2d = at::empty({row_num, output_column}, input_conti.options());
        slice_offset += cur_concat_size;
        outputs.push_back(output_2d);
    }
    at::TensorList outputs_list = at::TensorList(outputs);
    EXEC_NPU_CMD(aclnnMultisliceConcat, input_conti, concat_num, concat_size, slice_begin, slice_length, outputs_list);

    return outputs;
}

// 在NPU命名空间里面注册multislice_concat
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("multislice_concat(Tensor input, "
          "                  int concat_num, "
          "                  int[] concat_size, "
          "                  int[] slice_begin, "
          "                  int[] slice_length) -> (Tensor[])");
}

// 这里表示该算子的 NPU 实现由 multislice_concat_impl_npu 函数提供
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("multislice_concat", &multislice_concat_impl_npu);
}
