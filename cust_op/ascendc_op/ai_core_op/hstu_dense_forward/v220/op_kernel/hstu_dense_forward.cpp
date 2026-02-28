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
__aicore__ inline void InvokeHstuOpImpl(const HstuDenseForward::DenseArgs& args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData* __restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute();
}

#else

#include "hstu_dense_forward_kernel.h"

template <typename T>
__aicore__ inline void InvokeHstuOpImpl(const HstuDenseForward::DenseArgs& args)
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
    op.Compute();
}

#endif

#include "kernel_operator.h"
#include "hstu_kernel_tiling_key.h"

template<int maskedType, bool enableBias, int typeTilingKey, bool deterministic,
         int blockM, int blockN, int blockK>
__global__ __aicore__ void hstu_dense_forward(GM_ADDR q,
                                              GM_ADDR k,
                                              GM_ADDR v,
                                              GM_ADDR mask,
                                              GM_ADDR attnBias,
                                              GM_ADDR attnOutput,
                                              GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    HstuDenseForward::DenseArgs args{q, k, v, mask, attnBias, attnOutput, workspace, tiling};

#ifdef SUPPORT_V200
    if constexpr (typeTilingKey == HstuDenseForward::NORMAL_TILING_KEY) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernelv200<half, HstuDenseForwardTilingData>>(args);
    }
#else
    using FastConfig = HstuForward::TraitParams<DTYPE_Q, DTYPE_Q, enableBias,
        deterministic, static_cast<HstuForward::CausalMaskT>(maskedType), blockM, blockN, blockK>;

    InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<FastConfig, HstuDenseForwardTilingData>>(args);
#endif
}
