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
    #include "hstu_dense_forward_normal_kernel_v200_fuxi.h"
#else
    #include "hstu_dense_forward_jagged_kernel_fuxi.h"
#endif
#include "kernel_operator.h"

#ifdef SUPPORT_V200
    template <typename T>
    __aicore__ inline void run_op(HstuDenseForwardFuxi::Args& args)
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
#else
    template <typename T>
    __aicore__ inline void run_op(HstuDenseForwardFuxi::Args& args)
    {
        TPipe tPipe;
        HstuDenseForwardFuxi::HstuDenseForwardJaggedKernelFuxi<T> op;
        GET_TILING_DATA(tilingData, args.tiling);
        const HstuDenseForwardFuxiTilingData *__restrict tilingDataPtr = &tilingData;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(),
            op.qkMatmul, &tilingDataPtr->qkMatmul,
            op.svMatmul, &tilingDataPtr->svMatmul,
            op.tvMatmul, &tilingDataPtr->tvMatmul,
            op.pvMatmul, &tilingDataPtr->pvMatmul);
        uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
        op.qkMatmul.SetUserDefInfo(tilingPtr);
        op.svMatmul.SetUserDefInfo(tilingPtr);
        op.tvMatmul.SetUserDefInfo(tilingPtr);
        op.pvMatmul.SetUserDefInfo(tilingPtr);
        op.Init(args, tilingDataPtr, &tPipe);
        op.Compute(tilingDataPtr);
    }
#endif

extern "C" __global__ __aicore__ void hstu_dense_forward_fuxi(GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR timestampBias, GM_ADDR positionBias, GM_ADDR mask, GM_ADDR attnOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    HstuDenseForwardFuxi::Args args{q, k, v, timestampBias, positionBias, mask, attnOutput, workspace, tiling};
#ifdef SUPPORT_V200
    if (TILING_KEY_IS(0)) {
        run_op<half>(args);
    }
#else
    run_op<DTYPE_Q>(args);
#endif
}