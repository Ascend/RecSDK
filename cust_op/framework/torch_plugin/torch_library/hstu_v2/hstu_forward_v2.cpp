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
                                    const c10::optional<int64_t>& targetGroupSize, const c10::optional<double>& alpha)
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
    auto denseBias = c10::value_or_else(rab, [] { return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] { return at::Tensor(); });

    if (denseBias.defined()) {
        TORCH_CHECK((denseBias.dim() == 4 && (denseBias.size(1) == denseQ.size(1))), "Error: rab shape mismatch. ");
        TORCH_CHECK((denseBias.size(0) * denseBias.size(2) == denseQ.size(0)), "Error: rab shape mismatch. ");
        TORCH_CHECK((denseBias.size(0) * denseBias.size(3) == denseK.size(0)), "Error: rab shape mismatch. ");
    }

    auto batchsize = acSeqOffset.size(0) - 1;
    auto _zeros = at::zeros({batchsize}, acSeqOffset.options());
    auto acNumContext = numContext.has_value() ? numContext.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acNumTarget = numTarget.has_value() ? numTarget.value().to(acSeqOffset.scalar_type()) : _zeros;
    auto acTargetGroupSize = targetGroupSize.value_or(0);
    double realAlpha = alpha.value_or(1.0);

    bool use_fp8 = (q.scalar_type() == at::kFloat8_e4m3fn);
    at::ScalarType output_dtype = use_fp8 ? at::kHalf : denseQ.scalar_type();

    auto attnOutput = at::zeros({denseQ.size(0), denseQ.size(1), denseV.size(2)}, denseQ.options().dtype(output_dtype));

    TORCH_CHECK(attnOutput.defined(), "attnOutput is not defined!");
    TORCH_CHECK(attnOutput.storage().data_ptr() != nullptr, "attnOutput storage is null!");
    TORCH_CHECK(attnOutput.is_contiguous(), "attnOutput must be contiguous!");

    auto siluScale = siluScale_.value_or(0.0);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLenQ) : siluScale;
    auto realmaxSeqLenK = maxSeqLenK.value_or(maxSeqLenQ);

    EXEC_NPU_CMD(aclnnHstuForwardV2, denseQ, denseK, denseV, maskNpu, denseBias, acSeqOffset, acSeqOffsetK,
                 acNumContext, acNumTarget, maxSeqLenQ, realmaxSeqLenK, realSiluScale, acTargetGroupSize, realAlpha,
                 attnOutput);
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
          "            float? alpha) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_forward_v2", TORCH_FN(hstu_forward_v2_impl_npu));
}
