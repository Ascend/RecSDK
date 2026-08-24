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
#include "torch/types.h"
#include "../common/common_utils.h"

namespace hstu {
at::Tensor hstu_paged_forward_impl_npu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const c10::optional<at::Tensor>& kvCache,
    const c10::optional<at::Tensor>& kCache, const c10::optional<at::Tensor>& vCache,
    const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& attnBias, const int64_t maskType,
    const int64_t maxSeqLen, const int64_t maxSeqLenK, const double siluScale, const at::Tensor& seqOffset,
    const at::Tensor& seqOffsetK, const at::Tensor& seqOffsetT, const at::Tensor& pageOffsets,
    const at::Tensor& pageIds, const at::Tensor& lastPageLen, const at::Tensor& numTarget,
    const int64_t targetGroupSize, const c10::optional<double>& alpha, const bool deterministic = false)
{
    const bool hasSplitCache = kCache.has_value() && kCache->defined() && vCache.has_value() && vCache->defined();
    const bool hasCombinedCache = kvCache.has_value() && kvCache->defined();
    TORCH_CHECK(!(hasSplitCache && hasCombinedCache),
                "hstu_paged: kv_cache and k_cache/v_cache cannot be provided at the same time.");
    TORCH_CHECK(hasSplitCache || hasCombinedCache, "hstu_paged: must provide either kv_cache or k_cache/v_cache.");
    check_tensor_non_empty(pageIds, "page_ids");
    check_tensor_non_empty(lastPageLen, "last_page_len");
    auto acPageOffsets = pageOffsets.to(seqOffset.scalar_type());
    auto acPageIds = pageIds.to(seqOffset.scalar_type());
    auto acLastPageLen = lastPageLen.to(seqOffset.scalar_type());

    TORCH_CHECK(q.dim() == CONST_3, "The q should be 3D in jagged layout");

    auto acSeqOffset = seqOffset;
    auto acSeqOffsetK = seqOffsetK.to(seqOffset.scalar_type());
    auto acSeqOffsetT = seqOffsetT.to(seqOffset.scalar_type());
    TORCH_CHECK(acSeqOffset.size(0) >= CONST_2, "acSeqOffsetq params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetK.size(0) >= CONST_2, "acSeqOffsetK params error should have at least two element.");
    TORCH_CHECK(acSeqOffsetT.size(0) >= CONST_2, "acSeqOffsetT params error should have at least two element.");

    auto denseQ = q.contiguous();
    auto denseK = k.contiguous();
    auto denseV = v.contiguous();
    at::Tensor denseKvCache;
    at::Tensor denseKCache;
    at::Tensor denseVCache;
    if (hasCombinedCache) {
        check_tensor_non_empty(kvCache.value(), "kv_cache");
        denseKvCache = kvCache->contiguous();
    } else {
        check_tensor_non_empty(kCache.value(), "k_cache");
        check_tensor_non_empty(vCache.value(), "v_cache");
        denseKCache = kCache->contiguous();
        denseVCache = vCache->contiguous();
    }
    auto denseBias = c10::value_or_else(attnBias, [] { return at::Tensor(); });
    auto maskNpu = c10::value_or_else(mask, [] { return at::Tensor(); });

    TORCH_CHECK(MaxSeqLenCheck(maxSeqLen), "maxSeqLen check failed");
    TORCH_CHECK(MaxSeqLenCheck(maxSeqLenK), "maxSeqLenK check failed");
    TORCH_CHECK(MaskCheck(maskType, maskNpu.defined()), "maskType check failed");
    TORCH_CHECK(!numTarget.defined() || targetGroupSize > 0,
                "targetGroupSize must be greater than 0 when numTarget is defined");

    auto attnOutput = deterministic ? at::empty_like(denseQ) : at::zeros_like(denseQ);
    double realSiluScale = (siluScale == 0.0) ? 1.0f / static_cast<double>(maxSeqLen) : siluScale;
    double realAlpha = alpha.value_or(1.0);

    const auto _numContext = at::zeros_like(numTarget);
    const auto acNumTarget = numTarget.to(seqOffset.scalar_type());

    EXEC_NPU_CMD(aclnnHstuPagedForward, denseQ, denseK, denseV, maskNpu, denseBias, acSeqOffset, acSeqOffsetK,
                 acSeqOffsetT, denseKvCache, acPageOffsets, acPageIds, acLastPageLen, _numContext, acNumTarget,
                 denseKCache, denseVCache, maskType, maxSeqLen, maxSeqLenK, realSiluScale, targetGroupSize, realAlpha,
                 deterministic, attnOutput);
    return attnOutput;
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("hstu_paged(Tensor q, "
          "           Tensor k, "
          "           Tensor v, "
          "           Tensor? kv_cache=None, "
          "           Tensor? k_cache=None, "
          "           Tensor? v_cache=None, "
          "           Tensor? mask=None, "
          "           Tensor? attn_bias=None,"
          "           int mask_type=0, "
          "           int max_seq_len=0, "
          "           int max_seq_len_k=0, "
          "           float silu_scale=0.0, "
          "           Tensor seq_offset=None, "
          "           Tensor seq_offset_k=None, "
          "           Tensor seq_offset_t=None, "
          "           Tensor page_offsets=None, "
          "           Tensor page_ids=None, "
          "           Tensor last_page_len=None, "
          "           Tensor num_target=None, "
          "           int target_group_size=0, "
          "           float? alpha=1.0, "
          "           bool deterministic=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)

{
    m.impl("hstu_paged", TORCH_FN(hstu_paged_forward_impl_npu));
}
}  // namespace hstu
