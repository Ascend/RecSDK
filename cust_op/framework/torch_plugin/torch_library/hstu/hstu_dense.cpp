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
#include "../common/common_utils.h"

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

    const auto _acSeqOffset = at::empty(1, at::kInt);
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
    const int64_t isDeltaQK = 0;
    double realAlpha = 1.0;
    bool deterministic = false;
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
                 isDeltaQK,
                 realAlpha,
                 deterministic,
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
    TORCH_CHECK(q.dim() == dim, "The q should be 4D in normal layout");
    TORCH_CHECK(k.dim() == dim, "The k should be 4D in normal layout");
    TORCH_CHECK(v.dim() == dim, "The v should be 4D in normal layout");

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

    uint32_t batchSize = grad.size(0);  // 0 means index 0
    uint32_t seqLen = grad.size(1);     // 1 means index 1
    uint32_t headNum = grad.size(2);    // 2 means index 2
    uint32_t headDim = grad.size(3);    // 3 means index 3

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

    if (maskType == MASK_TYPE_CUSTOM) {
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

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();

    TORCH_CHECK(seqLen >= MIN_SEQ_LEN && seqLen <= MAX_SEQ_LEN, "seqLen expect in [1, 20480], but value is ", seqLen);
    TORCH_CHECK(seqLen == maxSeqLen, "seqLen must be equal to maxSeqLen");

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);

    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::zeros_like(denseAttnBias);
    } else {
        auto biasGradSeqLen = (seqLen + 256 - 1) / 256 * 256;  // get 256 bit aligned biasGrad space
        attnBiasGradOutput = at::zeros({batchSize, headNum, biasGradSeqLen, biasGradSeqLen},
                                       at::device(denseGrad.device()).dtype(denseGrad.dtype()));
    }
    auto _acSeqOffset = at::empty(1, at::kInt);
    auto _denseNumContext = at::Tensor();
    auto _denseNumTarget = at::Tensor();
    auto _acTargetGroupSize = int();
    double realAlpha = 1.0;

    const char *layout = "normal";
    EXEC_NPU_CMD(aclnnHstuDenseBackward,
                 denseGrad,
                 denseQ,
                 denseK,
                 denseV,
                 denseMask,
                 denseAttnBias,
                 _acSeqOffset,
                 _denseNumContext,
                 _denseNumTarget,
                 layout,
                 maskType,
                 maxSeqLen,
                 realSiluScale,
                 _acTargetGroupSize,
                 realAlpha,
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

TORCH_LIBRARY_IMPL(mxrec, AutogradPrivateUse1, m)
{
    m.impl("hstu_dense", TORCH_FN(hstu_dense_autograd));
}
}