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

extern "C" __global__ __aicore__ void hstu_dense_backward(GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                          GM_ADDR attnBias, GM_ADDR seqOffset, GM_ADDR numContext,
                                                          GM_ADDR numTarget, GM_ADDR qGrad, GM_ADDR kGrad,
                                                          GM_ADDR vGrad, GM_ADDR attnBiasGrad, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    HstuDenseBackward::Args args{grad,      q,     k,     v,     mask,         attnBias,  seqOffset, numContext,
                                 numTarget, qGrad, kGrad, vGrad, attnBiasGrad, workspace, tiling};

    if (TILING_KEY_IS(5)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<float, DTYPE_SEQ_OFFSET_Q> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(4)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<bfloat16_t, DTYPE_SEQ_OFFSET_Q> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(3)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<half, DTYPE_SEQ_OFFSET_Q> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(2)) {
        HstuDenseBackward::HstuDenseBackwardKernel<float> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(1)) {
        HstuDenseBackward::HstuDenseBackwardKernel<bfloat16_t> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(0)) {
        HstuDenseBackward::HstuDenseBackwardKernel<half> kernel;
        kernel.Compute(args);
    }
}