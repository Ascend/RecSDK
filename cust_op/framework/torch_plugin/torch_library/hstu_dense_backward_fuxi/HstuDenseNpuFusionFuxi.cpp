/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
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
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_2 = 2;

bool HstuBackMaskCheck(int64_t maskType)
{
    if (maskType < MASK_TYPE_TRIL || maskType > MASK_TYPE_CUSTOM) {
        printf("maskType expect in [0, 3], but value is %d\n", maskType);
        return false;
    }

    if (maskType == MASK_TYPE_TRIU) {
        printf("maskType current not support triu now, pls use custome mask\n");
        return false;
    }

    return true;
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_jagged_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> biasPosition,
    const c10::optional<at::Tensor> biasTimestamp,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(grad.dim() == CONST_3, "The grad should be 3D in jagged layout");

    auto acSeqOffset = seqOffset.value_or(at::IntArrayRef{});
    TORCH_CHECK(acSeqOffset.size() >= CONST_2, "acSeqOffset params error should have at least two element.");

    auto acBiasPosition = biasPosition.value_or(at::Tensor());
    auto acBiasTimestamp = biasTimestamp.value_or(at::Tensor());
    auto acMask = mask.value_or(at::Tensor());

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBiasPosition = acBiasPosition.contiguous();
    auto denseBiasTimestamp = acBiasTimestamp.contiguous();

    auto denseMask = acMask.contiguous();
    bool enableBias = denseBiasPosition.defined();
    if (enableBias) {
        TORCH_CHECK(grad.size(CONST_2) % CONST_3 == 0, "grad size 2 should be divisible by 3");
    }
    auto grads = enableBias ? at::chunk(grad, CONST_3, CONST_2) : std::vector<at::Tensor>{};
    auto denseGrad = enableBias ? grads[0].contiguous() : grad.contiguous();
    auto denseGradBiasTimestamp = enableBias ? grads[1].contiguous() : at::Tensor();
    auto denseGradBiasPosition = enableBias ? grads[2].contiguous() : at::Tensor();

    uint32_t batchSize = acSeqOffset.size() - 1;
    uint32_t headNum = denseGrad.size(1); // 1 means index 1
    uint32_t headDim = denseGrad.size(2); // 2 means index 2

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);

    TORCH_CHECK(HstuBackMaskCheck(maskType), "maskType check failed");
    if (static_cast<uint32_t>(maskType) == MASK_TYPE_CUSTOM) {
        TORCH_CHECK(denseMask.defined(), "use maskType:MASK_CUSTOM, but no mask given\n");
        // mask dim 2 must be equalto maxSeqLen
        TORCH_CHECK(denseMask.size(2) == maxSeqLen, "mask size 2 should be equal to maxSeqLen\n");
    }

    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    auto qGradOutput = at::empty_like(denseQ);
    auto kGradOutput = at::empty_like(denseK);
    auto vGradOutput = at::empty_like(denseV);
    
    at::Tensor biasPositionGradOutput;
    at::Tensor biasTimestampGradOutput;
    at::Tensor vBtsGradOutput = at::empty_like(denseV);
    at::Tensor vBposGradOutput = at::empty_like(denseV);
    if (enableBias) {
        biasPositionGradOutput = at::zeros({batchSize, headNum, maxSeqLen, maxSeqLen}, denseBiasPosition.options());
        biasTimestampGradOutput = at::zeros({batchSize, headNum, maxSeqLen, maxSeqLen}, denseBiasTimestamp.options());
    } else {
        biasPositionGradOutput = at::zeros({batchSize, headNum, maxSeqLen, maxSeqLen},
                                           at::device(denseGrad.device()).dtype(denseGrad.dtype()));
        biasTimestampGradOutput = at::zeros({batchSize, headNum, maxSeqLen, maxSeqLen},
                                            at::device(denseGrad.device()).dtype(denseGrad.dtype()));
    }

    const char *layout = "jagged";
    EXEC_NPU_CMD(aclnnHstuDenseBackwardFuxi,
        denseGrad,
        denseQ,
        denseK,
        denseV,
        denseMask,
        denseBiasPosition,
        denseBiasTimestamp,
        denseGradBiasPosition,
        denseGradBiasTimestamp,
        layout,
        maskType,
        maxSeqLen,
        realSiluScale,
        acSeqOffset,
        qGradOutput,
        kGradOutput,
        vGradOutput,
        biasPositionGradOutput,
        biasTimestampGradOutput,
        vBposGradOutput,
        vBtsGradOutput);

    if (enableBias) {
        vGradOutput = vGradOutput + vBposGradOutput + vBtsGradOutput;
    }
    auto bposGradOutput = enableBias ? at::sum(at::sum(biasPositionGradOutput, 1), 0, true) : at::Tensor();
    auto btsGradOutput = enableBias ? at::sum(biasTimestampGradOutput, 1) : at::Tensor();
    return std::make_tuple(qGradOutput, kGradOutput, vGradOutput, bposGradOutput, btsGradOutput);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_dense_backward_impl_npu(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor> mask,
    const c10::optional<at::Tensor> biasPosition,
    const c10::optional<at::Tensor> biasTimestamp,
    const std::string layout,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(layout == "jagged", "The layout should be jagged");
    check_tensor_non_empty(grad, "grad");
    check_tensor_non_empty(q, "q");
    check_tensor_non_empty(k, "k");
    check_tensor_non_empty(v, "v");

    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat || q.scalar_type() == at::kBFloat16,
                "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ", q.scalar_type());
    
    // NPU设备校验
    std::vector<at::Tensor> tensors = {grad, q, k, v};
    std::vector<std::string> names = {"grad", "q", "k", "v"};
    
    if (mask.has_value()) {
        tensors.push_back(mask.value());
        names.push_back("mask");
    }
    if (biasPosition.has_value()) {
        tensors.push_back(biasPosition.value());
        names.push_back("biasPosition");
    }
    if (biasTimestamp.has_value()) {
        tensors.push_back(biasTimestamp.value());
        names.push_back("biasTimestamp");
    }
     
    check_tensor_npu_device(tensors, names);

    return hstu_dense_jagged_backward_impl_npu(
        grad, q, k, v, mask, biasPosition, biasTimestamp,
        maskType, maxSeqLen, siluScale, seqOffset);
}


TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_dense_backward_fuxi(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? mask=None, \
        Tensor? biasPosition=None, Tensor? biasTimestamp=None, \
        str layout=\"jagged\", int maskType=0, int maxSeqLen=0, float siluScale=0.0, \
        int[]? seqOffset=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_dense_backward_fuxi", &hstu_dense_backward_impl_npu);
}
