/**
* @file relative_attn_bias_time.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/

#include "kernel_operator.h"
#include "rab_common.h"
#include "relative_attn_bias_time.h"

extern "C" __global__ __aicore__ void relative_attn_bias_time(GM_ADDR timestamps,
                                                              GM_ADDR timestampsWeights,
                                                              GM_ADDR rabTimeOut,
                                                              GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    Args args{
        timestamps, timestampsWeights, rabTimeOut, workspace, tiling
    };
    if (tilingData.tswType == static_cast<int>(DataType::FP32)) {
        RelativeAttnBiasTime<float> kernel;
        kernel.Compute(args);
    } else if (tilingData.tswType == static_cast<int>(DataType::FP16)) {
        RelativeAttnBiasTime<half> kernel;
        kernel.Compute(args);
    }
}
