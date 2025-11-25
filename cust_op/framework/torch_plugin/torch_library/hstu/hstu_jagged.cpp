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
#include <cstdint>
#include "c10/core/ScalarType.h"
#include "hstu_common.h"
#include "torch/types.h"
#include "../common/common_utils.h"

namespace hstu {

at::Tensor hstu_jagged_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const at::Tensor& seqOffset,
    const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget,
    const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset.to(torch::kInt64);
    auto batchsize = acSeqOffset.size(0) - 1;
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    auto _zeros = at::zeros({batchsize}, acSeqOffset.options());
    auto acNumContext = numContext.value_or(_zeros).to(torch::kInt64);
    auto acNumTarget = numTarget.value_or(_zeros).to(torch::kInt64);
    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "MaxSeqLen check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");
    if (static_cast<uint32_t>(maskType) == MASK_TYPE_CUSTOM) {
        // mask dim 2 must be equal to maxSeqLen
        TORCH_CHECK(maskNpu.size(2) == maxSeqLen, "mask size 2 should be equal to maxSeqLen\n");
    }

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto acSeqOffsetK = acSeqOffset;
    const auto maxSeqLenK = maxSeqLen;

    const auto _acSeqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();

    const char *layout = "jagged";
    const int64_t isDeltaQK = 0;
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
                 attnOutput);
    return attnOutput;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_jagged_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const at::Tensor& seqOffset,
    const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget,
    const c10::optional<int64_t>& targetGroupSize,
    const c10::optional<double>& alpha)
{
    TORCH_CHECK(grad.dim() == CONST_3, "The grad should be 3D in jagged layout");
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");
    TORCH_CHECK(k.dim() == CONST_3, "The k should be 3D in jagged layout");
    TORCH_CHECK(v.dim() == CONST_3, "The v should be 3D in jagged layout");

    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K batch size check failed");
    TORCH_CHECK(q.sizes() == v.sizes(), "Q and V batch size check failed");
    TORCH_CHECK(q.sizes() == grad.sizes(), "Q and grad batch size check failed");

    TORCH_CHECK(
        grad.scalar_type() == at::kHalf || grad.scalar_type() == at::kFloat || grad.scalar_type() == at::kBFloat16,
        "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ", grad.scalar_type());
    TORCH_CHECK(grad.scalar_type() == q.scalar_type(),
                "grad dtype should be the same as q dtype, but got grad: ", grad.scalar_type(),
                " and q: ", q.scalar_type());
    TORCH_CHECK(grad.scalar_type() == k.scalar_type(),
                "grad dtype should be the same as k dtype, but got grad: ", grad.scalar_type(),
                " and k: ", k.scalar_type());
    TORCH_CHECK(grad.scalar_type() == v.scalar_type(),
                "grad dtype should be the same as v dtype, but got grad: ", grad.scalar_type(),
                " and v: ", v.scalar_type());

    uint32_t batchSize = seqOffset.size(0) - 1;
    uint32_t seqLen = maxSeqLen;
    uint32_t headNum = grad.size(1);
    uint32_t headDim = grad.size(2);

    ShapeRange batchSizeRange(MIN_BATCH_SIZE, MAX_BATCH_SIZE, MULTIPLE_BATCH_SIZE_TIMES, "batchSize");
    ShapeRange seqLenRange(MIN_SEQ_LEN, MAX_SEQ_LEN, MULTIPLE_SEQ_LEN_TIMES, "seqLen");
    ShapeRange headNumRange(MIN_HEAD_NUM, MAX_HEAD_NUM, MULTIPLE_HEAD_NUM_TIMES, "headNum");
    ShapeRange headDimRange(MIN_HEAD_DIM, MAX_HEAD_DIM, MULTIPLE_HEAD_DIM_TIMES, "headDim");

    TORCH_CHECK(batchSizeRange.Check(batchSize), "batchSize expect in [1, 2048], but value is ", batchSize);
    TORCH_CHECK(seqLenRange.Check(seqLen), "seqLen expect in [1, 20480], but value is ", seqLen);
    TORCH_CHECK(headNumRange.Check(headNum), "headNum expect in [1, 16], but value is ", headNum);
    TORCH_CHECK(headDimRange.Check(headDim), "headDim expect in [16, 512], but value is ", headDim);

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaskCheck(maskType, CheckOptionalTensorIsNotNone(mask)), "maskType check failed");

    if (static_cast<uint32_t>(maskType) == MASK_TYPE_CUSTOM) {
        TORCH_CHECK(CheckOptionalTensorIsNotNone(mask), "mask is required when maskType is MASK_TYPE_CUSTOM");
        TORCH_CHECK(mask.value().dim() == CONST_4, "The mask should be 4D in normal layout");
        TORCH_CHECK(mask.value().size(0) == batchSize, "The mask batch size should be equal to the grad batch size");
        TORCH_CHECK(mask.value().size(1) == headNum, "The mask seqLen should be equal to the grad seqLen");
        TORCH_CHECK(mask.value().size(2) == seqLen, "The mask headNum should be equal to the grad headNum");
        TORCH_CHECK(mask.value().size(3) == seqLen, "The mask seqLen should be equal to the grad seqLen");
        TORCH_CHECK(mask.value().scalar_type() == grad.scalar_type(),
                    "mask dtype should be the same as grad dtype, but got mask: ", mask.value().scalar_type(),
                    " and grad: ", grad.scalar_type());
    }

    if (CheckOptionalTensorIsNotNone(attnBias)) {
        TORCH_CHECK(attnBias.value().dim() == CONST_4, "The attnBias should be 4D in normal layout");
        TORCH_CHECK(attnBias.value().size(0) == batchSize,
                    "The attnBias batch size should be equal to the grad batch size");
        TORCH_CHECK(attnBias.value().size(1) == headNum, "The attnBias seqLen should be equal to the grad seqLen");
        TORCH_CHECK(attnBias.value().size(2) == seqLen, "The attnBias headNum should be equal to the grad headNum");
        TORCH_CHECK(attnBias.value().size(3) == seqLen, "The attnBias seqLen should be equal to the grad seqLen");
        TORCH_CHECK(attnBias.value().scalar_type() == grad.scalar_type(),
                    "attnBias dtype should be the same as grad dtype, but got attnBias: ",
                    attnBias.value().scalar_type(), " and grad: ", grad.scalar_type());
    }

    if (CheckOptionalTensorIsNotNone(numContext) || CheckOptionalTensorIsNotNone(numTarget)) {
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

    auto acSeqOffset = seqOffset.to(torch::kInt64);
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());
    auto acNumContext = numContext.value_or(at::Tensor());
    if (acNumContext.defined()) {
        acNumContext = acNumContext.to(torch::kInt64);
    }
    auto acNumTarget = numTarget.value_or(at::Tensor());
    if (acNumTarget.defined()) {
        acNumTarget = acNumTarget.to(torch::kInt64);
    }

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

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");

    if (static_cast<uint32_t>(maskType) == MASK_TYPE_CUSTOM) {
        TORCH_CHECK(denseMask.defined(), "use maskType:MASK_CUSTOM, but no mask given\n");
        // mask dim 2 must be equal to maxSeqLen
        TORCH_CHECK(denseMask.size(2) == maxSeqLen, "mask size 2 should be equal to maxSeqLen\n");
    }

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

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

    EXEC_NPU_CMD(aclnnHstuDenseBackward,
                 denseGrad,
                 denseQ,
                 denseK,
                 denseV,
                 denseMask,
                 denseAttnBias,
                 acSeqOffset,
                 denseNumContext,
                 denseNumTarget,
                 layout,
                 maskType,
                 maxSeqLen,
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
    m.def("hstu_jagged.equal(Tensor q, "
          "                  Tensor k, "
          "                  Tensor v, "
          "                  Tensor? mask=None, "
          "                  Tensor? attn_bias=None, "
          "                  int mask_type=0, "
          "                  int max_seq_len=0, "
          "                  float silu_scale=0.0, "
          "                  Tensor seq_offset=None, "
          "                  Tensor? num_context=None, "
          "                  Tensor? num_target=None, "
          "                  int? target_group_size=0,"
          "                  float? alpha=1.0) -> Tensor");
    m.def("hstu_jagged_backward.equal(Tensor grad, "
          "                           Tensor q, "
          "                           Tensor k, "
          "                           Tensor v, "
          "                           Tensor? mask=None, "
          "                           Tensor? attn_bias=None, "
          "                           int mask_type=0, "
          "                           int max_seq_len=0, "
          "                           float silu_scale=0.0, "
          "                           Tensor seq_offset=None, "
          "                           Tensor? num_context=None, "
          "                           Tensor? num_target=None, "
          "                           int? target_group_size=0,"
          "                           float? alpha=1.0) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_jagged.equal", TORCH_FN(hstu_jagged_forward_impl_npu));
    m.impl("hstu_jagged_backward.equal", TORCH_FN(hstu_jagged_backward_impl_npu));
}

class HstuJaggedNpuFusion : public torch::autograd::Function<HstuJaggedNpuFusion> {
public:
    static at::Tensor forward(AutogradContext *ctx,
                              const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const c10::optional<at::Tensor>& mask,
                              const c10::optional<at::Tensor>& attnBias,
                              const int64_t maskType,
                              const int64_t maxSeqLen,
                              const double siluScale,
                              const at::Tensor& seqOffset,
                              const c10::optional<at::Tensor>& numContext,
                              const c10::optional<at::Tensor>& numTarget,
                              const c10::optional<int64_t>& targetGroupSize,
                              const c10::optional<double>& alpha)
    {
        at::AutoDispatchBelowADInplaceOrView guard;

        ctx->save_for_backward({ q, k, v, mask.value_or(at::Tensor()), attnBias.value_or(at::Tensor()),
                                numContext.value_or(at::Tensor()), numTarget.value_or(at::Tensor()), seqOffset});

        ctx->saved_data["maskType"] = maskType;
        ctx->saved_data["maxSeqLen"] = maxSeqLen;
        ctx->saved_data["siluScale"] = siluScale;
        ctx->saved_data["targetGroupSize"] = targetGroupSize.value_or(0);
        ctx->saved_data["alpha"] = alpha.value_or(1.0);
        return hstu_jagged_forward_impl_npu(q, k, v, mask, attnBias, maskType,
                                            maxSeqLen, siluScale, seqOffset,
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
        auto seqOffset = saved[7];

        auto maskType = ctx->saved_data["maskType"].toInt();
        auto maxSeqLen = ctx->saved_data["maxSeqLen"].toInt();
        auto siluScale = ctx->saved_data["siluScale"].toDouble();
        auto targetGroupSize = ctx->saved_data["targetGroupSize"].toInt();
        auto alpha = ctx->saved_data["alpha"].toDouble();
        auto resultTuple = hstu_jagged_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType,
                                                         maxSeqLen, siluScale, seqOffset,
                                                         numContext, numTarget, targetGroupSize, alpha);

        // 返回梯度数量必须与前向输入参数数量一致
        if (attnBias.defined()) {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    std::get<3>(resultTuple), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
        } else {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
        }
    }
};

at::Tensor hstu_jagged_autograd(const at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const c10::optional<at::Tensor>& mask,
                                const c10::optional<at::Tensor>& attnBias,
                                const int64_t maskType,
                                const int64_t maxSeqLen,
                                const double siluScale,
                                const at::Tensor& seqOffset,
                                const c10::optional<at::Tensor>& numContext,
                                const c10::optional<at::Tensor>& numTarget,
                                const c10::optional<int64_t>& targetGroupSize,
                                const c10::optional<double>& alpha)
{
    return HstuJaggedNpuFusion::apply(q, k, v, mask, attnBias, maskType,
                                      maxSeqLen, siluScale, seqOffset,
                                      numContext, numTarget, targetGroupSize, alpha);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_jagged.equal", TORCH_FN(hstu_jagged_autograd));
}
}