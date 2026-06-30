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

#include "hstu_dense_forward_tiling.h"
#include "hstu_dense_forward_normal_kernel.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void hstu_dense_forward(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                         GM_ADDR attnBias, GM_ADDR attnOutput, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HstuDenseForwardTilingData);
    HstuDenseForward::Args args{q, k, v, attnBias, mask, attnOutput, workspace, tiling};

    if (TILING_KEY_IS(0)) {
        InvokeHstuNormalOpImpl<half>(args);
    } else if (TILING_KEY_IS(1)) {
        InvokeHstuNormalOpImpl<bfloat16_t>(args);
    } else if (TILING_KEY_IS(2)) {
        InvokeHstuNormalOpImpl<float>(args);
    }
}
