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
constexpr int MAX_BATCH_SIZE = 512;
constexpr int NUM_BUFFER = 2;
constexpr int MAX_SEQ_CNT = 128;
constexpr int GATHER_PROCESS_WINDOW = 4096;

constexpr int8_t TYPE_FP32 = 0;
constexpr int8_t TYPE_FP16 = 1;
constexpr int8_t TYPE_INT32 = 3;
constexpr int8_t TYPE_INT64 = 9;

using namespace AscendC;

struct Args {
    // pos_bias
    GM_ADDR positionBias;
    GM_ADDR identity;
    // ts_bias
    GM_ADDR timestamps;
    GM_ADDR timestampsWeights;
    // out
    GM_ADDR rabPosOut;
    GM_ADDR rabTimeOut;

    GM_ADDR workspace;
    GM_ADDR tiling;
};
#endif  // MXREC_ADD_ONS_RAB_COMMON_H
