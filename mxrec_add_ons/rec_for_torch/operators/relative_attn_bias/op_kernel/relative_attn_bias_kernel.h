/**
 * @file relative_attn_bias_kernel.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_KERNEL_H
#define MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_KERNEL_H

#include "rab_common.h"
#include "relative_attn_bias_pos.h"
#include "relative_attn_bias_time.h"
#include "kernel_operator.h"
using namespace AscendC;

template <typename FloatType>
class RelativeAttnBiasKernel {
public:
    __aicore__ inline RelativeAttnBiasKernel() {}

    __aicore__ inline void Compute(Args args)
    {
#ifdef SUPPORT_V200
        RelativeAttnBiasPos<FloatType> rabPos;
        rabPos.Compute(args);
#endif
        RelativeAttnBiasTime<FloatType> rabTime;
        rabTime.Compute(args);
    }
};

#endif  // MXREC_ADD_ONS_RELATIVE_ATTN_BIAS_KERNEL_H
