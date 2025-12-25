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


#ifdef SUPPORT_V200
#include "hstu_dense_forward_kernel_v200.h"

template <typename T>
__aicore__ inline void InvokeHstuOpImpl(const HstuDenseForward::Args& args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData* __restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

#else
#include "hstu_dense_forward_jagged_kernel.h"
#include "hstu_paged_forward_kernel.h"
#include "hstu_dense_forward_kernel.h"

template <typename T>
__aicore__ inline void InvokeHstuOpImpl(const HstuDenseForward::Args& args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData* __restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, (TCubeTiling*)nullptr, op.svMatmul,
        (TCubeTiling*)nullptr);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    op.qkMatmul.SetUserDefInfo(tilingPtr);
    op.svMatmul.SetUserDefInfo(tilingPtr);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

#endif

#include "kernel_operator.h"
#include "hstu_kernel_tiling_key.h"

template<int maskedType, bool enableBias, bool isQkUseUb, int typeTilingKey, bool deterministic>
__global__ __aicore__ void hstu_dense_forward(GM_ADDR q,
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
    HstuDenseForward::Args args{q, k, v, mask, attnBias, seq_offset_q, seq_offset_k,
                                seq_offset_t, kv_cache, page_offsets, page_ids, last_page_len,
                                num_context, num_target, attnOutput, workspace, tiling};
#ifdef SUPPORT_V200
    if constexpr (typeTilingKey == HstuDenseForward::NORMAL_TILING_KEY) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernelv200<half>>(args);
    }
#else
    if constexpr (typeTilingKey == HstuDenseForward::NORMAL_TILING_KEY) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<
            DTYPE_Q, enableBias, isQkUseUb, static_cast<HstuDenseForward::CausalMaskT>(maskedType)>>(args);
    } else if constexpr (typeTilingKey == HstuDenseForward::JAGGED_TILING_KEY) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<
            DTYPE_Q, DTYPE_SEQ_OFFSET_Q, enableBias, isQkUseUb, deterministic,
            static_cast<HstuDenseForward::CausalMaskT>(maskedType)>>(args);
    } else if constexpr (typeTilingKey == HstuDenseForward::PAGED_TILING_KEY) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardPagedKernel<
            DTYPE_Q, DTYPE_SEQ_OFFSET_Q, enableBias, isQkUseUb, deterministic,
            static_cast<HstuDenseForward::CausalMaskT>(maskedType)>>(args);
    }
#endif
}
