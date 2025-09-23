/**
 * @file hstu_common_const.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef MXREC_HSTU_COMMON_CONST_H
#define MXREC_HSTU_COMMON_CONST_H

namespace HstuDenseForward {

constexpr uint32_t MAX_BATCH_SIZE = 2048;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;

constexpr int INVALID_TASK_ID = -1;

}  // namespace HstuDenseForward

#endif  // MXREC_HSTU_COMMON_CONST_H
