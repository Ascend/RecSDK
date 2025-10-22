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
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_2 = 2;

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
    const c10::optional<at::Tensor>& timestampBias,
    const c10::optional<at::Tensor>& positionBias,
    const c10::optional<at::Tensor>& mask,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(q.dim() == CONST_4, "The q should be 4D in normal layout");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseTsBias = c10::value_or_else(timestampBias, [] {return at::Tensor(); });
    auto densePosBias = c10::value_or_else(positionBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    uint32_t batchSize = denseQ.size(0); // 0 means index 0
    uint32_t seqLen = denseQ.size(1); // 1 means index 1
    uint32_t headNum = denseQ.size(2); // 2 means index 2
    uint32_t headDim = denseQ.size(3); // 3 means index 3

    TORCH_CHECK(seqLen >= MIN_SEQ_LEN && seqLen <= MAX_SEQ_LEN,
        "maxSeqLen expect in [1, 20480], but value is ", seqLen);
    TORCH_CHECK(maxSeqLen == seqLen, "maxSeqLen should equal to q dim 1");
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    bool useRab = (timestampBias.has_value() && positionBias.has_value());
    uint64_t tempDim = static_cast<uint64_t>(headNum) * static_cast<uint64_t>(headDim);
    if (useRab) {
        tempDim = static_cast<uint64_t>(CONST_3) * tempDim;
    }
    TORCH_CHECK(tempDim <= std::numeric_limits<uint32_t>::max(),
        "tempDim limit (0, %u], but get %llu\n", std::numeric_limits<uint32_t>::max(), tempDim);
    uint32_t outDim2 = static_cast<uint32_t>(tempDim);
    auto attnOutput = at::empty({batchSize, seqLen, outDim2}, q.options());

    const char *layout = "normal";
    EXEC_NPU_CMD(aclnnHstuDenseForwardFuxi,
        denseQ,
        denseK,
        denseV,
        denseTsBias,
        densePosBias,
        maskNpu,
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
    const c10::optional<at::Tensor>& timestampBias,
    const c10::optional<at::Tensor>& positionBias,
    const c10::optional<at::Tensor>& mask,
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
    auto denseTsBias = c10::value_or_else(timestampBias, [] {return at::Tensor(); });
    auto densePosBias = c10::value_or_else(positionBias, [] {return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] {return at::Tensor(); });

    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
        "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    bool useRab = (timestampBias.has_value() && positionBias.has_value());
    uint32_t outDim1 = useRab ? (CONST_3 * denseQ.size(1) * denseQ.size(2)) : (denseQ.size(1) * denseQ.size(2));
    auto attnOutput = at::empty({denseQ.size(0), outDim1}, q.options());

    const char *layout = "jagged";
    EXEC_NPU_CMD(aclnnHstuDenseForwardFuxi,
        denseQ,
        denseK,
        denseV,
        denseTsBias,
        densePosBias,
        maskNpu,
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
    const c10::optional<at::Tensor>& timestampBias,
    const c10::optional<at::Tensor>& positionBias,
    const c10::optional<at::Tensor>& mask,
    const int64_t maskType,
    const int64_t maxSeqLen,
    const double siluScale,
    const std::string layout,
    c10::optional<at::IntArrayRef> seqOffset)
{
    TORCH_CHECK(layout == "normal" || layout == "jagged", "The layout should be normal or jagged");

    check_tensor_non_empty(q, "q");
    check_tensor_non_empty(k, "k");
    check_tensor_non_empty(v, "v");

    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat || q.scalar_type() == at::kBFloat16,
        "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ", q.scalar_type());

    // NPU设备校验
    std::vector<at::Tensor> tensors = {q, k, v};
    std::vector<std::string> names = {"q", "k", "v"};
    
    if (timestampBias.has_value()) {
        tensors.push_back(timestampBias.value());
        names.push_back("timestampBias");
    }
    if (positionBias.has_value()) {
        tensors.push_back(positionBias.value());
        names.push_back("positionBias");
    }
    if (mask.has_value()) {
        tensors.push_back(mask.value());
        names.push_back("mask");
    }
    
    check_tensor_npu_device(tensors, names);
  
    if (layout == "normal") {
        return hstu_dense_normal_forward_impl_npu(q, k, v, timestampBias, positionBias, mask, maskType, maxSeqLen,
            siluScale, seqOffset);
    } else {
        return hstu_dense_jagged_forward_impl_npu(q, k, v, timestampBias, positionBias, mask, maskType, maxSeqLen,
            siluScale, seqOffset);
    }
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_fuxi(Tensor q, Tensor k, Tensor v, Tensor? timestampBias=None, Tensor? positionBias=None,\
        Tensor? mask=None, int maskType=0, int maxSeqLen=0, float siluScale=0.0, str layout=\"normal\", \
        int[]? seqOffset=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_fuxi", &hstu_dense_forward_impl_npu);
}

