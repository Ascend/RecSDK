/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LCCL_COMM_ARGS_H
#define LCCL_COMM_ARGS_H

#include <climits>
#include "kernel_operator.h"

constexpr int32_t MAX_BLOCK_NUM = 48;
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
constexpr int64_t TIME_OUT = 375000000;     // timeout. about 5 minits

constexpr int32_t RMA_UB_B4_BUFF_OFFSET = 64;
constexpr int32_t RMA_UB_B8_BUFF_OFFSET = 128;
constexpr int32_t RMA_UB_DATA_BUFF_OFFSET = 256;
constexpr int32_t RMA_SHM_HEAD_LEN = 128;      // length of queue head
constexpr int32_t RMA_SHM_DATA_HEAD = 56;      // length of data's head
constexpr int32_t UNIT_COPY_SIZE = 190 * 1024;
constexpr int32_t RMA_SHAPE_DIM_MAX = 2;

constexpr uint64_t RMA_WORK_SPACE_SIZE = 202 * 1024 * 1024;
constexpr uint64_t SWAP_CACHE_SIZE = 100 * 1024 * 1024;        // cache size
constexpr uint64_t SWAP_IN_FLAG_OFFSET = 0;                    // offset of flags
constexpr uint64_t SWAP_IN_CACHE_OFFSET = 1 * 1024 * 1024;     // data cache offset
constexpr uint64_t SWAP_OUT_FLAG_OFFSET = 101 * 1024 * 1024;   // offset of flags
constexpr uint64_t SWAP_OUT_CACHE_OFFSET = 102 * 1024 * 1024;  // data cache offset

// queue's head
struct RmaShmHeader {
    uint64_t queueCapacity;     // queue's capacity or depth
    uint64_t totalMemSize;      // total memory size(B)
    uint64_t seqIn;             // last enqueue sequence
    uint64_t seqOut;            // last dequeue sequence
    uint64_t frontOffset;       // front offset
    uint64_t tailOffset;        // tail offset
    uint64_t buffLimit;
    uint64_t seqOutPre;         // last pre-dequeue sequence
    uint64_t frontOffsetPre;    // last pre-dequeue front offset
};

struct RmaQueueOffset {
    static constexpr int32_t RMA_CAPACITY_OFFSET = 0;
    static constexpr int32_t RMA_TOTAL_SIZE_OFFSET = 1;
    static constexpr int32_t RMA_SEQ_IN_OFFSET = 2;
    static constexpr int32_t RMA_SEQ_OUT_OFFSET = 3;
    static constexpr int32_t RMA_QUEUE_FRONT_OFFSET = 4;
    static constexpr int32_t RMA_QUEUE_TAIL_OFFSET = 5;
    static constexpr int32_t RMA_BUFF_LIMIT_OFFSET = 6;
    static constexpr int32_t RMA_SEQ_OUT_PRE_OFFSET = 7;
    static constexpr int32_t RMA_QUEUE_FRONT_PRE_OFFSET = 8;
};

// data's head
struct RmaShmDataHead {
    uint64_t totalLen;                  // queue item's total length(B) = dataLen + RMA_SHM_DATA_HEAD
    uint64_t sequence;                  // item's sequence
    int32_t dataType;                   // data type: {0:float32}
    int32_t dimNum;
    int64_t dims[RMA_SHAPE_DIM_MAX];
    uint64_t dataLen;                   // data size(B)
    uint64_t readyLen;                  // data size has been written to queue(use to pipe)
};
constexpr int32_t RMA_READY_LEN_OFFSET = 6;

// error type
constexpr uint64_t RMA_QUEUE_TIME_OUT = 10000;
// sync flag
constexpr uint64_t RMA_PRE_SYNC = 10086;
constexpr uint64_t RMA_POST_SYNC = 10087;

#endif // LCCL_COMM_ARGS_H
