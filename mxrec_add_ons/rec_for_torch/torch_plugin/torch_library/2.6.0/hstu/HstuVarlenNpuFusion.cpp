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
at::Tensor hstu_varlen_forward_impl_npu(const at::Tensor& q,
                                        const at::Tensor& k,
                                        const at::Tensor& v,
                                        const c10::optional<at::Tensor>& mask,
                                        const c10::optional<at::Tensor>& attnBias,
                                        const int64_t maskType,
                                        const int64_t maxSeqLen,
                                        const int64_t maxSeqLenK,
                                        const double siluScale,
                                        const at::Tensor& seqOffset,
                                        const at::Tensor& seqOffsetK)
{
    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK;
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffset params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    auto denseBias = c10::value_or_else(attnBias, [] { return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] { return at::Tensor(); });

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenK), "maxSeqLenK check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");

    auto attnOutput = at::empty_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / maxSeqLen : siluScale;

    const auto _acSeqOffsetT = at::Tensor();
    const auto _kvCacheNpu = at::Tensor();
    const auto _pageOffsets = at::Tensor();
    const auto _pageIds = at::Tensor();
    const auto _lastPageLen = at::Tensor();
    const auto _numContext = at::Tensor();
    const auto _numTarget = at::Tensor();
    const auto _actTargetGroupSize = int();

    const char *layout = "jagged";
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
                 _numContext,
                 _numTarget,
                 maskType,
                 maxSeqLen,
                 maxSeqLenK,
                 realSiluScale,
                 layout,
                 _actTargetGroupSize,
                 attnOutput);
    return attnOutput;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_varlen(Tensor q, Tensor k, Tensor v, Tensor? mask=None, Tensor? attnBias=None, \
           int maskType=0, int maxSeqLen=0, int maxSeqLenK=0, float siluScale=0.0, Tensor seqOffset=None, \
           Tensor seqOffsetK=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("hstu_varlen", &hstu_varlen_forward_impl_npu);
}
}