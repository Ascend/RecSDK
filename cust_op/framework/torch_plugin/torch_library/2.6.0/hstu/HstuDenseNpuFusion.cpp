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

#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr size_t MIN_SEQ_LEN = 1;
constexpr size_t MAX_SEQ_LEN = 20480;
constexpr uint32_t MASK_TYPE_TRIL = 0;
constexpr uint32_t MASK_TYPE_TRIU = 1;
constexpr uint32_t MASK_TYPE_CUSTOM = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_2 = 2;

namespace hstu {
bool MaskCheck(int64_t maskType, uint32_t maskIsDefine)
{
    if (maskType < MASK_TYPE_TRIL || maskType > MASK_TYPE_CUSTOM) {
        printf("maskType expect in [0, 3], but value is %d\n", maskType);
        return false;
    }

    if (maskType == MASK_TYPE_TRIU) {
        printf("maskType current not support triu now, pls use custome mask\n");
        return false;
    }

    if (maskType == MASK_TYPE_CUSTOM && !maskIsDefine) {
        printf("use custome mask must have valide mask tensor \n");
        return false;
    }
    return true;
}

at::Tensor hstu_dense_normal_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset
)
{
    TORCH_CHECK(q.dim() == CONST_4, "The q should be 4D in normal layout");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    uint32_t batchSize = denseQ.size(0); // 0 means index 0
    uint32_t seqLen = denseQ.size(1); // 1 means index 1
    uint32_t headNum = denseQ.size(2); // 2 means index 2
    uint32_t headDim = denseQ.size(3); // 3 means index 3

    TORCH_CHECK(seqLen >= MIN_SEQ_LEN && seqLen <= MAX_SEQ_LEN,
        "maxSeqLen expect in [1, 20480], but value is ", seqLen);
    TORCH_CHECK(maxSeqLen == seqLen, "maxSeqLen should equal to q dim 1");

    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const char *layout = "normal";
    
    EXEC_NPU_CMD(aclnnHstuDenseForward,
        denseQ,
        denseK,
        denseV,
        maskNpu,
        denseBias,
        maskType,
        maxSeqLen,
        realSiluScale,
        layout,
        seqOffset,
        attnOutput);

    return attnOutput;
}

at::Tensor hstu_dense_jagged_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset.value_or(at::IntArrayRef{});
    TORCH_CHECK(acSeqOffset.size() >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
        "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);

    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const char *layout = "jagged";
    EXEC_NPU_CMD(aclnnHstuDenseForward,
        denseQ,
        denseK,
        denseV,
        maskNpu,
        denseBias,
        maskType,
        maxSeqLen,
        realSiluScale,
        layout,
        acSeqOffset,
        attnOutput);

    return attnOutput;
}

at::Tensor hstu_dense_forward_impl_npu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const std::string layout,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(layout == "normal" || layout == "jagged", "The layout should be normal/jagged");

    check_tensor_non_empty(q, "q");
    check_tensor_non_empty(k, "k");
    check_tensor_non_empty(v, "v");

    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat || q.scalar_type() == at::kBFloat16,
                "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ", q.scalar_type());
    
    // NPU设备校验
    std::vector<at::Tensor> tensors = {q, k, v};
    std::vector<std::string> names = {"q", "k", "v"};
    
    if (mask.has_value()) {
        tensors.push_back(mask.value());
        names.push_back("mask");
    }
    if (attnBias.has_value()) {
        tensors.push_back(attnBias.value());
        names.push_back("attnBias");
    }
    check_tensor_npu_device(tensors, names);

    if (layout == "normal") {
        return hstu_dense_normal_forward_impl_npu(q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale, seqOffset);
    } else {
        return hstu_dense_jagged_forward_impl_npu(q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale, seqOffset);
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_normal_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
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
        auto biasGradSeqLen = (seqLen + 256 - 1) / 256 * 256; // get 256 bit aligned biasGrad space
        attnBiasGradOutput = at::empty({batchSize, headNum, biasGradSeqLen, biasGradSeqLen},
                                       at::device(denseGrad.device()).dtype(denseGrad.dtype()));
    }
    check_tensor_non_empty(attnBiasGradOutput, "attnBiasGradOutput");

    const char *layout = "normal";
    EXEC_NPU_CMD(aclnnHstuDenseBackward,
        denseGrad,
        denseQ,
        denseK,
        denseV,
        denseMask,
        denseAttnBias,
        layout,
        maskType,
        maxSeqLen,
        realSiluScale,
        seqOffset,
        qGradOutput,
        kGradOutput,
        vGradOutput,
        attnBiasGradOutput);

    return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_jagged_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
{
    constexpr int dim = 3;
    TORCH_CHECK(grad.dim() == dim, "The grad should be 3D in jagged layout");

    auto acSeqOffset = seqOffset.value_or(at::IntArrayRef{});
    TORCH_CHECK(acSeqOffset.size() >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto acAttnBias = attnBias.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());

    auto denseGrad = grad.contiguous();
    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseAttnBias = acAttnBias.contiguous();
    auto denseMask = acMask.contiguous();

    uint32_t batchSize = acSeqOffset.size() - 1;
    uint32_t headNum = denseGrad.size(1); // 1 means index 1
    uint32_t headDim = denseGrad.size(2); // 2 means index 2

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);

    if (static_cast<uint32_t>(maskType) == MASK_TYPE_CUSTOM) {
        TORCH_CHECK(denseMask.defined(), "use maskType:MASK_CUSTOM, but no mask given\n");
        // mask dim 2 must be equalto maxSeqLen
        TORCH_CHECK(denseMask.size(2) == maxSeqLen, "mask size 2 should be equal to maxSeqLen\n");
    }

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);
    
    at::Tensor attnBiasGradOutput;
    if (denseAttnBias.defined()) {
        attnBiasGradOutput = at::empty_like(denseAttnBias);
        check_tensor_non_empty(denseAttnBias, "denseAttnBias");
    } else {
        attnBiasGradOutput = at::Tensor();
    }

    const char *layout = "jagged";
    EXEC_NPU_CMD(aclnnHstuDenseBackward, denseGrad, denseQ, denseK, denseV, denseMask, denseAttnBias, layout, maskType,
        maxSeqLen, realSiluScale, acSeqOffset, qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);

    if (denseAttnBias.defined()) {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, attnBiasGradOutput);
    } else {
        return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, at::Tensor());
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> attnBias,
    const std::string layout,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(layout == "normal" || layout == "jagged", "The layout should be normal/jagged");
    
    check_tensor_non_empty(grad, "grad");
    check_tensor_non_empty(q, "q");
    check_tensor_non_empty(k, "k");
    check_tensor_non_empty(v, "v");

    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat || q.scalar_type() == at::kBFloat16,
                "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ", q.scalar_type());
    
    // NPU设备校验
    std::vector<at::Tensor> tensors = {q, k, v};
    std::vector<std::string> names = {"q", "k", "v"};
    
    if (mask.has_value()) {
        tensors.push_back(mask.value());
        names.push_back("mask");
    }
    if (attnBias.has_value()) {
        tensors.push_back(attnBias.value());
        names.push_back("attnBias");
    }
    
    check_tensor_npu_device(tensors, names);

    if (layout == "normal") {
        return hstu_dense_normal_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale,
            seqOffset);
    } else {
        return hstu_dense_jagged_backward_impl_npu(grad, q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale,
            seqOffset);
    }
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_dense(Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, \
        int maskType=0, int maxSeqLen=0, float siluScale=0.0, str layout=\"normal\", int[]? seqOffset=None) -> Tensor");
    m.def("hstu_dense_backward(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? mask, Tensor? attnBias, \
        str layout, int maskType, int maxSeqLen, float siluScale=0.0, int[]? seqOffset=None) -> (Tensor, Tensor, \
        Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense", &hstu_dense_forward_impl_npu);
    m.impl("hstu_dense_backward", &hstu_dense_backward_impl_npu);
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
                              const double siluScale,
                              const std::string layout,
                              c10::optional<at::IntArrayRef> seqOffset)
    {
        at::AutoDispatchBelowADInplaceOrView guard;

        ctx->save_for_backward({ q, k, v, mask.value_or(at::Tensor()), attnBias.value_or(at::Tensor()) });
        ctx->saved_data["maskType"] = maskType;
        ctx->saved_data["maxSeqLen"] = maxSeqLen;
        ctx->saved_data["siluScale"] = siluScale;
        ctx->saved_data["layout"] = layout;

        if (seqOffset.has_value()) {
            auto seqOffsetVec = seqOffset->vec();
            ctx->saved_data["seqOffset"] = seqOffsetVec;
            ctx->saved_data["hasSeqOffset"] = true;
        } else {
            ctx->saved_data["hasSeqOffset"] = false;
        }

        return hstu_dense_forward_impl_npu(q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale, layout, seqOffset);
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
        auto layout = ctx->saved_data["layout"].toStringRef();

        bool hasSeqOffset = ctx->saved_data["hasSeqOffset"].toBool();
        std::vector<int64_t> seqOffsetVec;
        c10::optional<at::IntArrayRef> seqOffset;
        if (hasSeqOffset) {
            seqOffsetVec = ctx->saved_data["seqOffset"].toIntVector();
            seqOffset = at::IntArrayRef(seqOffsetVec);
        }

        auto resultTuple = hstu_dense_backward_impl_npu(grad, q, k, v, mask, attnBias, layout, maskType,
            maxSeqLen, siluScale, seqOffset);
        
        if (attnBias.defined()) {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                std::get<3>(resultTuple), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
        } else {
            return { std::get<0>(resultTuple), std::get<1>(resultTuple), std::get<2>(resultTuple), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor() };
        }
    }
};

at::Tensor hstu_dense_forward_impl_autograd(const at::Tensor& q,
                                            const at::Tensor& k,
                                            const at::Tensor& v,
                                            const c10::optional<at::Tensor>& mask,
                                            const c10::optional<at::Tensor>& attnBias,
                                            const int64_t maskType,
                                            const int64_t maxSeqLen,
                                            const double siluScale,
                                            const std::string layout,
                                            c10::optional<at::IntArrayRef> seqOffset)
{
    return HstuDenseNpuFusion::apply(q, k, v, mask, attnBias, maskType, maxSeqLen, siluScale, layout, seqOffset);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense", &hstu_dense_forward_impl_autograd);
}
}
