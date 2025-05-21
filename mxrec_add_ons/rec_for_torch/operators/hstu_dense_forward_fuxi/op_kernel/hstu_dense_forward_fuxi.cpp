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

#include "hstu_dense_forward_normal_kernel_v200_fuxi.h"

#include "kernel_operator.h"

#ifndef INVOKE_HSTU_NORMAL_V200_FUXI_OP_IMPL
#define INVOKE_HSTU_NORMAL_V200_FUXI_OP_IMPL(...)    \
    do {                                   \
        
    } while (0)
#endif

template <typename T>
__aicore__ void run_op(HstuDenseForwardFuxi::Args& args)
{
    TPipe tPipe;
    HstuDenseForwardFuxi::HstuDenseForwardNormalKernelv200Fuxi<T> op;
    GET_TILING_DATA(tilingData, args.tiling);
    const HstuDenseForwardFuxiTilingData *__restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(),
        op.qkMatmul, &tilingDataPtr->qkMatmul,
        op.svMatmul, &tilingDataPtr->svMatmul,
        op.tvMatmul, &tilingDataPtr->tvMatmul,
        op.pvMatmul, &tilingDataPtr->pvMatmul);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute(tilingDataPtr);
}

extern "C" __global__ __aicore__ void hstu_dense_forward_fuxi(GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR timestampBias, GM_ADDR positionBias, GM_ADDR mask, GM_ADDR attnOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    HstuDenseForwardFuxi::Args args{q, k, v, timestampBias, positionBias, mask, attnOutput, workspace, tiling};
    if (TILING_KEY_IS(0)) {
        INVOKE_HSTU_NORMAL_V200_FUXI_OP_IMPL(half);
    }
}