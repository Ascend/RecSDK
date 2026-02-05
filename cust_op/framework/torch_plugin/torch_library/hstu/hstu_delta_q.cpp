/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
#include "torch/types.h"
#include "../common/common_utils.h"

namespace hstu {
at::Tensor hstu_deltaq_forward_impl_npu(const at::Tensor& q,
                                        const at::Tensor& k,
                                        const at::Tensor& v,
                                        const c10::optional<at::Tensor>& mask,
                                        const c10::optional<at::Tensor>& attnBias,
                                        const int64_t maskType,
                                        const int64_t maxSeqLen,
                                        const int64_t maxSeqLenK,
                                        const double siluScale,
                                        const at::Tensor& seqOffset,
                                        const at::Tensor& seqOffsetK,
                                        const c10::optional<at::Tensor>& numContext,
                                        const c10::optional<at::Tensor>& numTarget,
                                        const c10::optional<int64_t>& targetGroupSize,
                                        const c10::optional<double>& alpha,
                                        const bool deterministic = false)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK.to(acSeqOffset.scalar_type());
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] { return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] { return at::Tensor(); });

    auto batchsize = acSeqOffset.size(0) - 1;
    auto _zeros = at::zeros({batchsize}, acSeqOffset.options());
    auto acNumContext = numContext.has_value() ? numContext.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acNumTarget = numTarget.has_value() ? numTarget.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenK), "maxSeqLenK check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : denseQ.scalar_type();

    auto attnOutput =
        deterministic
            ? at::empty({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype))
            : at::zeros({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype));

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _acSeqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();

    const char *layout = "jagged";
    const int64_t isDeltaQK = 1;
    EXEC_NPU_CMD(aclnnHstuDenseForward,
                 denseQ,
                 denseK,
                 denseV,
                 maskNpu,
                 denseBias,
                 acSeqOffset,
                 acSeqOffsetK,
                 _acSeqOffsetT,
                 _kvCacheNpu,
                 _pageOffsets,
                 _pageIds,
                 _lastPageLen,
                 acNumContext,
                 acNumTarget,
                 maskType,
                 maxSeqLen,
                 maxSeqLenK,
                 realSiluScale,
                 layout,
                 acTargetGroupSize,
                 isDeltaQK,
                 realAlpha,
                 deterministic,
                 attnOutput);
    return attnOutput;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_deltaq_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t maskType,
    const int64_t maxSeqLenQ,
    const int64_t maxSeqLenK,
    const double siluScale,
    const at::Tensor& seqOffsetQ,
    const at::Tensor& seqOffsetK,
    const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget,
    const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha)
{
    if (CheckOptionalTensorIsNotNone(numContext) || CheckOptionalTensorIsNotNone(numTarget)) {
        uint32_t batchSize = seqOffsetQ.size(0) - 1;
        TORCH_CHECK(numContext.has_value(), "numContext is required when numTarget or targetGroupSize is not None");
        TORCH_CHECK(numTarget.has_value(), "numTarget is required when numContext or targetGroupSize is not None");
        TORCH_CHECK(numContext.value().dim() == CONST_1, "The numContext should be 1D in normal layout");
        TORCH_CHECK(numTarget.value().dim() == CONST_1, "The numTarget should be 1D in normal layout");
        TORCH_CHECK(numContext.value().size(0) == batchSize,
                    "The numContext batch size should be equal to the grad batch size");
        TORCH_CHECK(numTarget.value().size(0) == batchSize,
                    "The numTarget batch size should be equal to the grad batch size");
        TORCH_CHECK(CheckInList(targetGroupSize.value_or(0), {1, 3}), "The targetGroupSize should be in [1, 3]");
    }

    auto acSeqOffsetQ = seqOffsetQ;
    auto acSeqOffsetK = seqOffsetK;

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());
    auto acNumContext = numContext.value_or(at::Tensor());
    auto acNumTarget = numTarget.value_or(at::Tensor());

    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();
    auto denseNumContext = acNumContext.contiguous();
    auto denseNumTarget = acNumTarget.contiguous();

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLenQ : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);

    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::zeros_like(denseAttnBias);
    } else {
        attnBiasGradOutput = at::Tensor();
    }

    const char* layout = "jagged";

    EXEC_NPU_CMD(aclnnHstuJaggedBackward,
                 denseGrad,
                 denseQ,
                 denseK,
                 denseV,
                 denseMask,
                 denseAttnBias,
                 acSeqOffsetQ,
                 acSeqOffsetK,
                 denseNumContext,
                 denseNumTarget,
                 layout,
                 maskType,
                 maxSeqLenQ,
                 maxSeqLenK,
                 realSiluScale,
                 acTargetGroupSize,
                 realAlpha,
                 qGradOutput,
                 kGradOutput,
                 vGradOutput,
                 attnBiasGradOutput);

    if (denseAttnBias.defined()) {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
    } else {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, at::Tensor());
    }
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_jagged.delta(Tensor q, "
          "                  Tensor k, "
          "                  Tensor v, "
          "                  Tensor? mask=None, "
          "                  Tensor? attn_bias=None, "
          "                  int mask_type=0, "
          "                  int max_seq_len=0, "
          "                  int max_seq_len_k=0, "
          "                  float silu_scale=0.0, "
          "                  Tensor seq_offset=None, "
          "                  Tensor seq_offset_k=None, "
          "                  Tensor? num_context=None, "
          "                  Tensor? num_target=None, "
          "                  int? target_group_size=0, "
          "                  float? alpha=1.0, "
          "                  bool deterministic=False) -> Tensor");
    m.def("hstu_jagged_backward.delta(Tensor grad, "
          "                           Tensor q, "
          "                           Tensor k, "
          "                           Tensor v, "
          "                           Tensor? mask=None, "
          "                           Tensor? attn_bias=None, "
          "                           int mask_type=0, "
          "                           int max_seq_len=0, "
          "                           int max_seq_len_k=0, "
          "                           float silu_scale=0.0, "
          "                           Tensor seq_offset=None, "
          "                           Tensor seq_offset_k=None, "
          "                           Tensor? num_context=None, "
          "                           Tensor? num_target=None, "
          "                           int? target_group_size=0,"
          "                           float? alpha=1.0) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_jagged.delta", TORCH_FN(hstu_deltaq_forward_impl_npu));
    m.impl("hstu_jagged_backward.delta", TORCH_FN(hstu_deltaq_backward_impl_npu));
}

class HstuDeltaqNpuFusion : public torch::autograd::Function<HstuDeltaqNpuFusion> {
public:
    static at::Tensor forward(AutogradContext *ctx,
                              const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const c10::optional<at::Tensor>& mask,
                              const c10::optional<at::Tensor>& attnBias,
                              const int64_t maskType,
                              const int64_t maxSeqLenQ,
                              const int64_t maxSeqLenK,
                              const double siluScale,
                              const at::Tensor& seqOffsetQ,
                              const at::Tensor& seqOffsetK,
                              const c10::optional<at::Tensor>& numContext,
                              const c10::optional<at::Tensor>& numTarget,
                              const c10::optional<int64_t>& targetGroupSize,
                              const c10::optional<double>& alpha)
    {
        at::AutoDispatchBelowADInplaceOrView guard;

        ctx->save_for_backward({q, k, v, mask.value_or(at::Tensor()), attnBias.value_or(at::Tensor()),
                                numContext.value_or(at::Tensor()), numTarget.value_or(at::Tensor()), seqOffsetQ,
                                seqOffsetK});

        ctx->saved_data["maskType"] = maskType;
        ctx->saved_data["maxSeqLenQ"] = maxSeqLenQ;
        ctx->saved_data["maxSeqLenK"] = maxSeqLenK;
        ctx->saved_data["siluScale"] = siluScale;
        ctx->saved_data["targetGroupSize"] = targetGroupSize.value_or(0);
        ctx->saved_data["alpha"] = alpha.value_or(1.0);
        return hstu_deltaq_forward_impl_npu(q, k, v, mask, attnBias, maskType,
                                            maxSeqLenQ, maxSeqLenK, siluScale, seqOffsetQ, seqOffsetK,
                                            numContext, numTarget, targetGroupSize, alpha);
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
        auto numContext = saved[5];
        auto numTarget = saved[6];
        auto seqOffsetQ = saved[7];
        auto seqOffsetK = saved[8];

        auto maskType = ctx->saved_data["maskType"].toInt();
        auto maxSeqLenQ = ctx->saved_data["maxSeqLenQ"].toInt();
        auto maxSeqLenK = ctx->saved_data["maxSeqLenK"].toInt();
        auto siluScale = ctx->saved_data["siluScale"].toDouble();
        auto targetGroupSize = ctx->saved_data["targetGroupSize"].toInt();
        auto alpha = ctx->saved_data["alpha"].toDouble();
        auto resultTuple = hstu_deltaq_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType,
                                                         maxSeqLenQ, maxSeqLenK, siluScale, seqOffsetQ, seqOffsetK,
                                                         numContext, numTarget, targetGroupSize, alpha);

        // 返回梯度数量必须与前向输入参数数量一致
        return {
            std::get<0>(resultTuple), // q
            std::get<1>(resultTuple), // k
            std::get<2>(resultTuple), // v
            at::Tensor(),             // mask
            std::get<3>(resultTuple), // bias
            at::Tensor(),             // mask_type
            at::Tensor(),             // max_seqlen_q
            at::Tensor(),             // max_seqlen_k
            at::Tensor(),             // silu_scale
            at::Tensor(),             // offset_q
            at::Tensor(),             // offset_k
            at::Tensor(),             // num_context
            at::Tensor(),             // num_target
            at::Tensor(),             // target_group_size
            at::Tensor()              // alpha
        };
    }
};

at::Tensor hstu_deltaq_autograd(const at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const c10::optional<at::Tensor>& mask,
                                const c10::optional<at::Tensor>& attnBias,
                                const int64_t maskType,
                                const int64_t maxSeqLenQ,
                                const int64_t maxSeqLenK,
                                const double siluScale,
                                const at::Tensor& seqOffsetQ,
                                const at::Tensor& seqOffsetK,
                                const c10::optional<at::Tensor>& numContext,
                                const c10::optional<at::Tensor>& numTarget,
                                const c10::optional<int64_t>& targetGroupSize,
                                const c10::optional<double>& alpha,
                                const bool deterministic = false)
{
    return HstuDeltaqNpuFusion::apply(q, k, v, mask, attnBias, maskType, maxSeqLenQ, maxSeqLenK, siluScale, seqOffsetQ,
                                      seqOffsetK, numContext, numTarget, targetGroupSize, alpha);
}

TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("hstu_jagged.delta", TORCH_FN(hstu_deltaq_autograd));
}
}  // namespace hstu