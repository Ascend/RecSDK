/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

namespace npu_reverse_sequence {

constexpr int64_t INPUT_DIM = 3;
constexpr int64_t SEQ_LENGTHS_DIM = 1;
constexpr int64_t EMB_DIM_MIN = 16;
constexpr int64_t EMB_DIM_MAX = 1024;

constexpr int64_t MAX_SEQ_LENGTH_DIM_INDEX = 1;
constexpr int64_t EMB_DIM_INDEX = 2;

constexpr int64_t BATCH_SIZE_MAX = 10240;
constexpr int64_t MAX_SEQ_LENGTH_MAX = 102400;

class ReverseSequenceOp : public torch::autograd::Function<ReverseSequenceOp> {
public:
    static at::Tensor forward(AutogradContext* ctx, const at::Tensor& input, const at::Tensor& seq_lengths)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({seq_lengths});

        auto output = input.clone();
        EXEC_NPU_CMD(aclnnReverseSequence, input, seq_lengths, output);
        return output;
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto seq_lengths = saved[0];
        check_tensor_non_empty(seq_lengths, "seq_lengths");
        auto grad_output_conti = grad_output.contiguous();
        auto output = grad_output_conti.clone();
        EXEC_NPU_CMD(aclnnReverseSequence, grad_output_conti, seq_lengths, output);
        return {output, Variable()};
    }
};

Tensor npu_reverse_sequence_impl(const Tensor& input, const Tensor& seq_lengths)
{
    check_tensor_dim(input, INPUT_DIM, "reverse_sequence input");
    check_tensor_dim(seq_lengths, SEQ_LENGTHS_DIM, "reverse_sequence seq_lengths");
    TORCH_CHECK(input.size(0) == seq_lengths.size(0),
        "param input dim 0: ", input.size(0), " is not equal to seq_lengths dim 0: ", seq_lengths.size(0));
    TORCH_CHECK(input.size(0) <= BATCH_SIZE_MAX,
                "input dim[0] is batch_size, must be less or equal to 10240, but got ", input.size(0));
    TORCH_CHECK(input.size(MAX_SEQ_LENGTH_DIM_INDEX) <= MAX_SEQ_LENGTH_MAX,
                "input dim[0] is batch_size, must be less or equal to 10240, but got ", input.size(1));
    int64_t emb_dim = input.size(EMB_DIM_INDEX);
    TORCH_CHECK(emb_dim >= EMB_DIM_MIN && emb_dim <= EMB_DIM_MAX && emb_dim % EMB_DIM_MIN == 0,
                "emb_dim must be in range:[", EMB_DIM_MIN, ", ", EMB_DIM_MAX, "] and is multiple of ", EMB_DIM_MIN,
                ", but got ", emb_dim);
    auto input_conti = input.contiguous();
    auto seq_lengths_conti = seq_lengths.contiguous();
    return ReverseSequenceOp::apply(input_conti, seq_lengths_conti);
}
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("reverse_sequence("
        "                   Tensor input, "
        "                   Tensor seq_lengths) -> (Tensor)");

    m.impl("reverse_sequence",
       torch::dispatch(c10::DispatchKey::Autograd,
                       TORCH_FN(npu_reverse_sequence::npu_reverse_sequence_impl)));
    m.impl("reverse_sequence",
           torch::dispatch(c10::DispatchKey::PrivateUse1,
                           TORCH_FN(npu_reverse_sequence::npu_reverse_sequence_impl)));
}