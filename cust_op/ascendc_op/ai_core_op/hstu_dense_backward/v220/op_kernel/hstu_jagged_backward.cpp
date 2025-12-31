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
#include "hstu_jagged_f16_r0_kernel.h"
#include "static_switch.h"
constexpr static int64_t HEAD_NUM_4 = 4;
constexpr static int64_t HEAD_NUM_8 = 8;
constexpr static int64_t HEAD_DIM_64 = 64;
constexpr static int64_t HEAD_DIM_128 = 128;
constexpr static int64_t BLOCK_HEIGHT_256 = 256;
constexpr static int64_t MASK_TYPE_TRIL = 0;

extern "C" __global__ __aicore__ void hstu_jagged_backward(GM_ADDR grad, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask,
                                                           GM_ADDR attnBias, GM_ADDR seqOffset, GM_ADDR numContext,
                                                           GM_ADDR numTarget, GM_ADDR qGrad, GM_ADDR kGrad,
                                                           GM_ADDR vGrad, GM_ADDR attnBiasGrad, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const HstuDenseBackwardTilingData* __restrict tilingDataPtr = &tilingData;

    HstuDenseBackward::Args args{grad,      q,     k,     v,     mask,         attnBias,  seqOffset, numContext,
                                 numTarget, qGrad, kGrad, vGrad, attnBiasGrad, workspace, tiling,    tilingDataPtr};

    bool isConstHeadNum = (tilingDataPtr->headNum == HEAD_NUM_4 || tilingDataPtr->headNum == HEAD_NUM_8);
    bool isConstHeadDim = (tilingDataPtr->headDim == HEAD_DIM_64 || tilingDataPtr->headDim == HEAD_DIM_128);
    bool isConstMaskType = (tilingDataPtr->maskType == MASK_TYPE_TRIL);
    bool isConstEnableBias = (tilingDataPtr->enableBias == 0);

    if (TILING_KEY_IS(5)) {
        HstuDenseBackward::HstuDenseBackwardJaggedKernel<float, DTYPE_SEQ_OFFSET_Q> kernel;
        kernel.Compute(args);
    } else if (TILING_KEY_IS(4)) {
        if (isConstHeadNum && isConstHeadDim && isConstMaskType && isConstEnableBias) {
            HEAD_NUM_SWITCH(tilingDataPtr->headNum, HEAD_NUM, {
                HEAD_DIM_SWITCH(tilingDataPtr->headDim, HEAD_DIM, {
                    HstuDenseBackward::HstuJaggedF16R0Kernel<bfloat16_t, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_256,
                                                             BLOCK_HEIGHT_256, HEAD_NUM, HEAD_DIM>
                        kernel;
                    kernel.Compute(args);
                });
            });
        } else {
            HstuDenseBackward::HstuDenseBackwardJaggedKernel<bfloat16_t, DTYPE_SEQ_OFFSET_Q> kernel;
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(3)) {
        if (isConstHeadNum && isConstHeadDim && isConstMaskType && isConstEnableBias) {
            HEAD_NUM_SWITCH(tilingDataPtr->headNum, HEAD_NUM, {
                HEAD_DIM_SWITCH(tilingDataPtr->headDim, HEAD_DIM, {
                    HstuDenseBackward::HstuJaggedF16R0Kernel<half, DTYPE_SEQ_OFFSET_Q, BLOCK_HEIGHT_256,
                                                             BLOCK_HEIGHT_256, HEAD_NUM, HEAD_DIM>
                        kernel;
                    kernel.Compute(args);
                });
            });
        } else {
            HstuDenseBackward::HstuDenseBackwardJaggedKernel<half, DTYPE_SEQ_OFFSET_Q> kernel;
            kernel.Compute(args);
        }
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