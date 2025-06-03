/**
* @file relative_attn_bias_backward.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/

#include "kernel_operator.h"
#include "rab_common.h"
#include "relative_attn_bias_backward.h"

extern "C" __global__ __aicore__ void relative_attn_bias_backward(GM_ADDR rabTimeGrad,
                                                                  GM_ADDR bucketTimestamps,
                                                                  GM_ADDR timestampsWeightsGrad,
                                                                  GM_ADDR workspace,
                                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    Args args{
        rabTimeGrad, bucketTimestamps, timestampsWeightsGrad, workspace, tiling
    };
    if (tilingData.gradDataType == TYPE_FP32) {
        RelativeAttnBiasBackward<float> kernel;
        kernel.Compute(args);
    } else if (tilingData.gradDataType == TYPE_FP16) {
        RelativeAttnBiasBackward<half> kernel;
        kernel.Compute(args);
    }
}
