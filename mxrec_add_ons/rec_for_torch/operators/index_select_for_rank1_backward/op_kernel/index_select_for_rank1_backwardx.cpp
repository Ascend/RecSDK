/**
* @file index_select_for_rank1_backward.cpp
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
*
*/
#include "index_select_for_rank1_backward.h"
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void index_select_for_rank1_backward(
        GM_ADDR gradY,
        GM_ADDR x,
        GM_ADDR index,
        GM_ADDR gradX,
        GM_ADDR gradIndex,
        GM_ADDR workspace,
        GM_ADDR tiling)
{
    Args args {
            gradY,
            x,
            index,
            gradX,
            gradIndex,
            workspace,
            tiling};
    IndexSelectForRank1BackwardOpKernel op;
    op.Process(args);
}
