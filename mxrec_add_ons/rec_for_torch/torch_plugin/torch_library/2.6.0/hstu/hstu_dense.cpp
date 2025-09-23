/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "hstu_common.h"

namespace hstu {
at::Tensor hstu_dense_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale)
{
    TORCH_CHECK(q.dim() == CONST_4, "The q should be 4D in dense layout");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _acSeqOffset = at::Tensor();
    const auto _acSeqOffsetK = at::Tensor();
    const auto _acSeqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();
    const auto _numContext = at::Tensor();
    const auto _numTarget = at::Tensor();
    const auto _maxSeqLenK = int();
    const auto _acTargetGroupSize = int();

    const char *layout = "normal";
    EXEC_NPU_CMD(aclnnHstuDenseForward,
                 denseQ,
                 denseK,
                 denseV,
                 maskNpu,
                 denseBias,
                 _acSeqOffset,
                 _acSeqOffsetK,
                 _acSeqOffsetT,
                 _kvCacheNpu,
                 _pageOffsets,
                 _pageIds,
                 _lastPageLen,
                 _numContext,
                 _numTarget,
                 maskType,
                 maxSeqLen,
                 _maxSeqLenK,
                 realSiluScale,
                 layout,
                 _acTargetGroupSize,
                 attnOutput);
    return attnOutput;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale)
{
    constexpr int dim = 4;
    TORCH_CHECK(grad.dim() == dim, "The grad should be 4D in normal layout");

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();

    uint32_t batchSize = denseGrad.size(0); // 0 means index 0
    uint32_t seqLen = denseGrad.size(1); // 1 means index 1
    uint32_t headNum = denseGrad.size(2); // 2 means index 2
    uint32_t headDim = denseGrad.size(3); // 3 means index 3

    TORCH_CHECK(seqLen >= MIN_SEQ_LEN && seqLen <= MAX_SEQ_LEN, "seqLen expect in [1, 20480], but value is ", seqLen);
    TORCH_CHECK(seqLen == maxSeqLen, "seqLen must be equal to maxSeqLen");

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);

    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::empty_like(denseAttnBias);
    } else {
        auto biasGradSeqLen = (seqLen + 256 - 1) / 256 * 256;  // get 256 bit aligned biasGrad space
        attnBiasGradOutput = at::empty({batchSize, headNum, biasGradSeqLen, biasGradSeqLen},
                                       at::device(denseGrad.device()).dtype(denseGrad.dtype()));
    }
    c10::optional<at::IntArrayRef> _acSeqOffset = c10::nullopt;
    auto _denseNumContext = at::Tensor();
    auto _denseNumTarget = at::Tensor();
    auto _acTargetGroupSize = int();

    const char *layout = "normal";
    EXEC_NPU_CMD(aclnnHstuDenseBackward,
                 denseGrad,
                 denseQ,
                 denseK,
                 denseV,
                 denseMask,
                 denseAttnBias,
                 _denseNumContext,
                 _denseNumTarget,
                 layout,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 _acSeqOffset,
                 _acTargetGroupSize,
                 qGradOutput,
                 kGradOutput,
                 vGradOutput,
                 attnBiasGradOutput);

    return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_dense(Tensor q, "
          "           Tensor k, "
          "           Tensor v, "
          "           Tensor? mask=None, "
          "           Tensor? attn_bias=None, "
          "           int mask_type=0, "
          "           int max_seq_len=0, "
          "           float silu_scale=0.0) -> Tensor");
    m.def("hstu_dense_backward(Tensor grad, "
          "                    Tensor q, "
          "                    Tensor k, "
          "                    Tensor v, "
          "                    Tensor? mask, "
          "                    Tensor? attn_bias, "
          "                    int mask_type=0, "
          "                    int max_seq_len=0, "
          "                    float silu_scale=0.0) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense", TORCH_FN(hstu_dense_forward_impl_npu));
    m.impl("hstu_dense_backward", TORCH_FN(hstu_dense_backward_impl_npu));
}

class HstuDenseNpuFusion : public torch::autograd::Function<HstuDenseNpuFusion> {
public:
    static at::Tensor forward(AutogradContext *ctx,
                              const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const c10::optional<at::Tensor>& mask,
                              const c10::optional<at::Tensor>& attnBias,
                              const int64_t maskType,
                              const int64_t maxSeqLen,
                              const double siluScale)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        ctx->save_for_backward({ q, k, v, mask.value_or(at::Tensor()), attnBias.value_or(at::Tensor())});
        ctx->saved_data["maskType"] = maskType;
        ctx->saved_data["maxSeqLen"] = maxSeqLen;
        ctx->saved_data["siluScale"] = siluScale;

        return hstu_dense_forward_impl_npu(q, k, v, mask, attnBias, maskType,
                                           maxSeqLen, siluScale);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto grad = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        auto q = saved[0];
        auto k = saved[1];
        auto v = saved[2];
        auto mask = saved[3];
        auto attnBias = saved[4];

        auto maskType = ctx->saved_data["maskType"].toInt();
        auto maxSeqLen = ctx->saved_data["maxSeqLen"].toInt();
        auto siluScale = ctx->saved_data["siluScale"].toDouble();

        auto resultTuple = hstu_dense_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType,
                                                        maxSeqLen, siluScale);

        if (attnBias.defined()) {
            // 返回q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale的梯度
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    std::get<3>(resultTuple), at::Tensor(), at::Tensor(), at::Tensor()};
        } else {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }
    }
};

at::Tensor hstu_dense_autograd(const at::Tensor& q,
                               const at::Tensor& k,
                               const at::Tensor& v,
                               const c10::optional<at::Tensor>& mask,
                               const c10::optional<at::Tensor>& attnBias,
                               const int64_t maskType,
                               const int64_t maxSeqLen,
                               const double siluScale)
{
    return HstuDenseNpuFusion::apply(q, k, v, mask, attnBias, maskType,
                                     maxSeqLen, siluScale);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense", TORCH_FN(hstu_dense_autograd));
}
}