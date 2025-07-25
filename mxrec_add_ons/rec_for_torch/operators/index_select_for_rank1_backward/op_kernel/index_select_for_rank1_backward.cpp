/**
 * @file index_select_for_rank1_backward.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#include "kernel_operator.h"
#include "index_select_for_rank1_backward_kernel.h"

enum class DataType {
    FP32 = 0,
    FP16 = 1,
    INT32 = 3,
    INT64 = 9
};

using namespace AscendC;

extern "C" __global__ __aicore__ void index_select_for_rank1_backward(GM_ADDR gradY,
                                                                      GM_ADDR x,
                                                                      GM_ADDR index,
                                                                      GM_ADDR gradX,
                                                                      GM_ADDR workspace,
                                                                      GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    IndexSelectForRank1BackwardKernel<DTYPE_INDEX> op(tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    op.Init(gradY, index, gradX);
    op.Process();
}
