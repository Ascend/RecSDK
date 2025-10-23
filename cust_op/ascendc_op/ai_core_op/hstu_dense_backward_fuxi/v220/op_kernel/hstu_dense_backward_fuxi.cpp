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
#include <cstdint>

#include "kernel_operator.h"

#include "hstu_dense_backward_jagged_kernel.h"
#include "hstu_dense_backward_kernel.h"

extern "C" __global__ __aicore__ void hstu_dense_backward_fuxi(
    GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
    GM_ADDR biasPosition, GM_ADDR biasTimestamp, GM_ADDR gradBposIn, GM_ADDR gradBtsIn,
    GM_ADDR qGrad, GM_ADDR kGrad, GM_ADDR vGrad, GM_ADDR bposGrad, GM_ADDR btsGrad,
    GM_ADDR vbposGrad, GM_ADDR vbtsGrad,
    GM_ADDR workspace, GM_ADDR tiling)
{
    HstuDenseBackwardFuxi::Args args{
        grad, q, k, v, mask,
        biasPosition, biasTimestamp, gradBposIn, gradBtsIn,
        qGrad, kGrad, vGrad, bposGrad, btsGrad,
        vbposGrad, vbtsGrad,
        workspace, tiling
    };

    if (TILING_KEY_IS(5)) {
        HstuDenseBackwardFuxi::HstuDenseBackwardJaggedKernelFuxi<float> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(4)) {
        HstuDenseBackwardFuxi::HstuDenseBackwardJaggedKernelFuxi<bfloat16_t> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(3)) {
        HstuDenseBackwardFuxi::HstuDenseBackwardJaggedKernelFuxi<half> kernel;
        kernel.Compute(args);
    }
}