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
__aicore__ inline void InvokeHstuOpImpl(const HstuDenseForward::Args &args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData *__restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

#else
#include "hstu_dense_forward_jagged_kernel.h"
#include "hstu_dense_forward_kernel.h"

template <typename T>
__aicore__ inline void InvokeHstuOpImpl(const HstuDenseForward::Args &args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardTilingData *__restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkMatmul, &tilingDataPtr->qkMatmul, op.svMatmul,
                      &tilingDataPtr->svMatmul);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    op.qkMatmul.SetUserDefInfo(tilingPtr);
    op.svMatmul.SetUserDefInfo(tilingPtr);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

#endif

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hstu_dense_forward(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                         GM_ADDR attnBias, GM_ADDR attnOutput, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    HstuDenseForward::Args args{q, k, v, attnBias, mask, attnOutput, workspace, tiling};
#ifdef SUPPORT_V200
    if (TILING_KEY_IS(0)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernelv200<half>>(args);
    }
#else
    if (TILING_KEY_IS(0)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<half>>(args);
    } else if (TILING_KEY_IS(1)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<bfloat16_t>>(args);
    } else if (TILING_KEY_IS(2)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardKernel<float>>(args);
    } else if (TILING_KEY_IS(3)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<half>>(args);
    } else if (TILING_KEY_IS(4)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<bfloat16_t>>(args);
    } else if (TILING_KEY_IS(5)) {
        InvokeHstuOpImpl<HstuDenseForward::HstuDenseForwardJaggedKernel<float>>(args);
    }
#endif
}