/**
* @file relative_attn_bias_pos.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/

#include "kernel_operator.h"
#include "rab_common.h"
#include "relative_attn_bias_pos.h"

extern "C" __global__ __aicore__ void relative_attn_bias_pos(GM_ADDR positionBias,
                                                             GM_ADDR identity,
                                                             GM_ADDR rabPosOut,
                                                             GM_ADDR workspace,
                                                             GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    Args args{
        positionBias, identity, rabPosOut, workspace, tiling
    };
    if (tilingData.dataType == static_cast<int>(DataType::FP32)) {
        RelativeAttnBiasPos<float> kernel;
        kernel.Compute(args);
    } else if (tilingData.dataType == static_cast<int>(DataType::FP16)) {
        RelativeAttnBiasPos<half> kernel;
        kernel.Compute(args);
    }
}
