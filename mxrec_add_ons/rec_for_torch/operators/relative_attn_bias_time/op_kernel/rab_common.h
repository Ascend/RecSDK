/**
 * @file rab_common.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_ADD_ONS_RAB_COMMON_H
#define MXREC_ADD_ONS_RAB_COMMON_H

#include "kernel_operator.h"
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_SEQ_CNT = 128;
constexpr int GATHER_PROCESS_WINDOW = 4096;

enum class DataType {
    FP32 = 0,
    FP16 = 1,
    INT32 = 3,
    INT64 = 9
};

struct Args {
    // ts_bias
    GM_ADDR timestamps;
    GM_ADDR timestampsWeights;
    // out
    GM_ADDR rabTimeOut;

    GM_ADDR workspace;
    GM_ADDR tiling;
};
#endif  // MXREC_ADD_ONS_RAB_COMMON_H
