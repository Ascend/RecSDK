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
==============================================================================*/

#include "kernel_operator.h"

#include "hstu_paged_forward_kernel.h"
#include "hstu_kernel_tiling_key.h"

using namespace HstuForward;
using namespace HstuPagedForward;

template <typename FastConfig, typename MatmulMgmtType, typename VectorScoreType>
__aicore__ inline void InvokeHstuOpImpl(const PagedArgs& args)
{
    MatmulMgmtType mmMgmt;
    VectorScoreType vecScore;

    HstuPagedForwardKernel<FastConfig, HstuPagedForwardTilingData, MatmulMgmtType, VectorScoreType> op;
    
    REGIST_MATMUL_OBJ(&op.pipe_, GetSysWorkSpacePtr(), mmMgmt.qkMatmul_, (TCubeTiling*)nullptr,
        mmMgmt.svMatmul_, (TCubeTiling*)nullptr);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    mmMgmt.qkMatmul_.SetUserDefInfo(tilingPtr);
    mmMgmt.svMatmul_.SetUserDefInfo(tilingPtr);

    op.Compute(args, &mmMgmt, &vecScore);
}

template<int maskedType, bool enableBias, int typeTilingKey, bool deterministic,
         int blockM, int blockN, int blockK>
__global__ __aicore__ void hstu_paged_forward(GM_ADDR q,
                                              GM_ADDR k,
                                              GM_ADDR v,
                                              GM_ADDR mask,
                                              GM_ADDR attnBias,
                                              GM_ADDR seq_offset_q,
                                              GM_ADDR seq_offset_k,
                                              GM_ADDR seq_offset_t,
                                              GM_ADDR kv_cache,
                                              GM_ADDR page_offsets,
                                              GM_ADDR page_ids,
                                              GM_ADDR last_page_len,
                                              GM_ADDR num_context,
                                              GM_ADDR num_target,
                                              GM_ADDR attnOutput,
                                              GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const HstuPagedForwardTilingData* __restrict tilingDataPtr = &tiling_data;

    PagedArgs args{q, k, v, mask, attnBias, seq_offset_q, seq_offset_k,
                   seq_offset_t, kv_cache, page_offsets, page_ids, last_page_len,
                   num_context, num_target, attnOutput, workspace, tiling, tilingDataPtr};

    using FastConfig = TraitParams<DTYPE_Q, DTYPE_SEQ_OFFSET_Q, enableBias,
        deterministic, static_cast<CausalMaskT>(maskedType), blockM, blockN, blockK>;
    
    using mmMgmtType = MatmulMgmt<DTYPE_Q, HstuPagedForwardTilingData>;

    if constexpr (enableBias) {
        using vecScoreType = VectorScoreWithRab<DTYPE_Q, static_cast<CausalMaskT>(maskedType), BlockMaskParams>;
        InvokeHstuOpImpl<FastConfig, mmMgmtType, vecScoreType>(args);
    } else {
        using vecScoreType = VectorScoreWithoutRab<DTYPE_Q, static_cast<CausalMaskT>(maskedType), BlockMaskParams>;
        InvokeHstuOpImpl<FastConfig, mmMgmtType, vecScoreType>(args);
    }
}
