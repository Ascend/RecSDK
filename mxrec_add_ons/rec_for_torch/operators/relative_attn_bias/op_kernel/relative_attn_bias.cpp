/**
* @file relative_attn_bias.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/

#include "rab_common.h"
#include "relative_attn_bias_kernel.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void relative_attn_bias(GM_ADDR positionBias,
                                                         GM_ADDR identity,
                                                         GM_ADDR timestamps,
                                                         GM_ADDR timestampsWeights,
                                                         GM_ADDR rabPosOut,
                                                         GM_ADDR rabTimeOut,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    Args args{
        positionBias, identity, timestamps, timestampsWeights, rabPosOut, rabTimeOut, workspace, tiling
    };
    if (tilingData.floatType == TYPE_FP32) {
        RelativeAttnBiasKernel<float> kernel;
        kernel.Compute(args);
    } else if (tilingData.floatType == TYPE_FP16) {
        RelativeAttnBiasKernel<half> kernel;
        kernel.Compute(args);
    }
}
