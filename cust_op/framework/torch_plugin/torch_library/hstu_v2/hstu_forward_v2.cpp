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
#include "c10/core/ScalarType.h"
#include "torch/types.h"
#include "hstu_common.h"
#include "../common/common_utils.h"

at::Tensor hstu_forward_v2_impl_npu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                    const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& rab,
                                    const int64_t maskType, const int64_t maxSeqLenQ,
                                    const c10::optional<int64_t> maxSeqLenK, const c10::optional<double> siluScale_,
                                    const at::Tensor& seqOffset, const c10::optional<at::Tensor>& seqOffsetK,
                                    const c10::optional<at::Tensor>& numContext,
                                    const c10::optional<at::Tensor>& numTarget,
                                    const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha,
                                    const c10::optional<at::Tensor>& metadata)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK.has_value() ? seqOffsetK.value().to(acSeqOffset.scalar_type())
                                               : seqOffset.to(acSeqOffset.scalar_type());
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = rab.value_or(at::Tensor());
    auto maskNpu = mask.value_or(at::Tensor());

    auto batchsize = acSeqOffset.size(0) - 1;
    auto _zeros = at::zeros({batchsize}, acSeqOffset.options());
    auto acNumContext = numContext.has_value() ? numContext.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acNumTarget = numTarget.has_value() ? numTarget.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);
    // metadata 为可选: 未传/None → 空 tensor → aclnn 视作 null → kernel nullptr → 旧设备现算分核(零回归)
    auto acMetadata = CheckOptionalTensorIsNotNone(metadata) ? metadata.value().to(at::kInt) : at::Tensor();

    bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : denseQ.scalar_type();

    auto attnOutput = at::zeros({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype));

    TORCH_CHECK(attnOutput.defined(), "attnOutput is not defined!");
    TORCH_CHECK(attnOutput.storage().data_ptr() != nullptr, "attnOutput storage is null!");
    TORCH_CHECK(attnOutput.is_contiguous(), "attnOutput must be contiguous!");

    auto siluScale = siluScale_.value_or(0.0);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLenQ) : siluScale;
    auto realmaxSeqLenK = maxSeqLenK.value_or(maxSeqLenQ);
    auto inMetadata = acMetadata.defined() ? acMetadata.contiguous() : at::Tensor();

    // op exec —— metadata 紧跟 num_target(与 OpDef 输入顺序一致: ...num_target, metadata),置于所有 attr 之前
    EXEC_NPU_CMD(aclnnHstuForwardV2, denseQ, denseK, denseV, maskNpu, denseBias, acSeqOffset, acSeqOffsetK,
                 acNumContext, acNumTarget, inMetadata, maxSeqLenQ, realmaxSeqLenK, realSiluScale, acTargetGroupSize,
                 realAlpha, attnOutput);
    return attnOutput;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_forward_v2(Tensor q, "
          "            Tensor k, "
          "            Tensor v, "
          "            Tensor? mask, "
          "            Tensor? rab, "
          "            int mask_type, "
          "            int max_seq_len, "
          "            int? max_seq_len_k, "
          "            float? silu_scale, "
          "            Tensor seq_offset, "
          "            Tensor? seq_offset_k, "
          "            Tensor? num_context, "
          "            Tensor? num_target, "
          "            int? target_group_size, "
          "            float? alpha, "
          "            Tensor? metadata=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_forward_v2", TORCH_FN(hstu_forward_v2_impl_npu));
}
