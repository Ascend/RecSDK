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
============================================================================== */
#include <cstdint>
#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "hstu_v2_param_check.h"
#include "c10/core/ScalarType.h"
#include "torch/types.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hstu_backward_v2_impl_npu(
    const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const int64_t maxSeqLenQ,
    const int64_t maxSeqLenK, const at::Tensor& seqOffsetQ, const at::Tensor& seqOffsetK,
    const c10::optional<at::Tensor>& rab, const c10::optional<at::Tensor>& numContext,
    const c10::optional<at::Tensor>& numTarget, const c10::optional<double>& scale,
    const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha, const int64_t windowSizeLeft,
    const int64_t windowSizeRight, const c10::optional<at::Tensor>& metadata,
    const c10::optional<at::Tensor>& arbitraryFunc, const c10::optional<tensor_list>& sparseInfo)
{
    hstu_v2::HstuV2ParamChecker(grad, q, k, v, seqOffsetQ, seqOffsetK, maxSeqLenQ, maxSeqLenK, rab, numContext,
                                numTarget, targetGroupSize, windowSizeLeft, windowSizeRight);

    auto _empty = at::Tensor();
    auto acRab = rab.value_or(at::Tensor());
    auto acNumContext =
        CheckOptionalTensorIsNotNone(numContext) ? numContext.value().to(seqOffsetQ.scalar_type()) : _empty;
    auto acNumTarget =
        CheckOptionalTensorIsNotNone(numTarget) ? numTarget.value().to(seqOffsetQ.scalar_type()) : _empty;
    // metadata 为可选: 未传/None → 空 tensor → aclnn 视作 null → kernel nullptr → 旧设备现算分核(零回归)
    auto acMetadata = CheckOptionalTensorIsNotNone(metadata) ? metadata.value().to(at::kInt) : _empty;
    // arbitrary_func 为可选: 未传/None → 空 tensor → kernel nullptr → 非 arbitrary 路径(IS_ARBITRARY=0)
    auto acArbitraryFunc =
        CheckOptionalTensorIsNotNone(arbitraryFunc) ? arbitraryFunc.value().to(at::kInt).contiguous() : _empty;

    // op input
    auto inGrad = grad.contiguous();
    auto inQ = q.contiguous();
    auto inK = k.contiguous();
    auto inV = v.contiguous();
    auto inRab = acRab.contiguous();
    auto inNumContext = acNumContext.contiguous();
    auto inNumTarget = acNumTarget.contiguous();
    auto inqShare = at::zeros_like(inQ, at::TensorOptions().dtype(at::kFloat));
    auto inMetadata = acMetadata.defined() ? acMetadata.contiguous() : _empty;
    auto inArbitraryFunc = acArbitraryFunc;

    tensor_list _empty_tensor_ls = tensor_list{at::empty({0}, at::kInt)};
    tensor_list inSparseInfoVec = sparseInfo.value_or(_empty_tensor_ls);
    at::TensorList inSparseInfo = at::TensorList(inSparseInfoVec);

    // op attr
    auto attrTargetGroupSize = targetGroupSize.value_or(1);
    double attrScale = scale.value_or(1.0f / maxSeqLenQ);
    double attrAlpha = alpha.value_or(1.0);
    int64_t attrWinLeft = windowSizeLeft;
    int64_t attrWinRight = windowSizeRight;

    // op output
    auto outQGrad = at::empty_like(inQ);
    auto outKGrad = at::empty_like(inK);
    auto outVGrad = at::empty_like(inV);
    auto outRabGrad = inRab.defined() ? at::zeros_like(inRab) : at::Tensor();

    // op exec —— metadata 紧跟 inqShare(与 OpDef 输入顺序一致: ...q_share, metadata),置于所有 attr 之前
    EXEC_NPU_CMD(aclnnHstuBackwardV2, inGrad, inQ, inK, inV, inRab, seqOffsetQ, seqOffsetK, inNumContext, inNumTarget,
                 inqShare, inMetadata, inArbitraryFunc, inSparseInfo, maxSeqLenQ, maxSeqLenK, attrScale,
                 attrTargetGroupSize, attrAlpha, attrWinLeft, attrWinRight, outQGrad, outKGrad, outVGrad, outRabGrad);
    // op return
    return std::make_tuple(outQGrad, outKGrad, outVGrad, outRabGrad);
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_backward_v2(Tensor grad, "
          "                   Tensor q, "
          "                   Tensor k, "
          "                   Tensor v, "
          "                   int max_seqlen_q, "
          "                   int max_seqlen_k, "
          "                   Tensor seq_offset_q, "
          "                   Tensor seq_offset_k, "
          "                   Tensor? rab=None, "
          "                   Tensor? num_context=None, "
          "                   Tensor? num_target=None, "
          "                   float? scale=0.0, "
          "                   int? target_group_size=1, "
          "                   float? alpha=1.0, "
          "                   int window_size_left=-1, "
          "                   int window_size_right=-1, "
          "                   Tensor? metadata=None, "
          "                   Tensor? arbitrary_func=None, "
          "                   Tensor[]? sparse_info=None) "
          " -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_backward_v2", TORCH_FN(hstu_backward_v2_impl_npu));
}
