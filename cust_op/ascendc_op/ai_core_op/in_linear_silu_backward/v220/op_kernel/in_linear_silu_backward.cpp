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
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"
#include "in_linear_silu_backward_tilingKey.h"
#include "in_linear_sliu_backward.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void InvokeLinearSiluBackwardOpImpl(const InLinearSiluBackward::Args& args)
{
    TPipe tPipe;
    T op;
    GET_TILING_DATA(tilingData, args.tiling);
    const InLinearSiluBackwardTilingData* __restrict tilingDataPtr = &tilingData;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.lxMatmul, &tilingData.LX_MM, op.lwMatmul,
        &tilingData.LW_MM);
    uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
    op.lxMatmul.SetUserDefInfo(tilingPtr);
    op.lwMatmul.SetUserDefInfo(tilingPtr);
    op.Init(args, tilingDataPtr, &tPipe);
    op.Compute();
}

template<bool enableBias, bool isTrans, bool isVardim>
__global__ __aicore__ void in_linear_silu_backward(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR user_grad,
                                                   GM_ADDR value_grad, GM_ADDR query_grad, GM_ADDR key_grad,
                                                   GM_ADDR linear_output, GM_ADDR x_grad, GM_ADDR weight_grad,
                                                   GM_ADDR bias_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    InLinearSiluBackward::Args args{x, weight, bias, user_grad, value_grad, query_grad, key_grad,
                                   linear_output, x_grad, weight_grad, bias_grad, workspace, tiling};
    InvokeLinearSiluBackwardOpImpl<InLinearSiluBackward::InLinearSiluBackward<
        DTYPE_X, DTYPE_WEIGHT, enableBias, isTrans, isVardim>>(args);
}
