/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef RMA_SHM_SVM_H
#define RMA_SHM_SVM_H

constexpr int32_t RMA_SHM_HEAD_LEN = 128; // queue's head length
constexpr int32_t RMA_SHM_DATA_HEAD = 56; // queue item's head length
constexpr int32_t RMA_SHM_READY_LEN = 48; // offset of readyLen in item's head
constexpr int32_t RMA_DIM_MAX = 2;

enum class RmaDevModel {
    SVM_MAP_DEV,    // 910_93
    PCIE_TH_DEV     // 910B
};

// Queue header definition
struct RmaShmHeader {
    uint64_t queueCapacity;    // depth of queue
    uint64_t totalMemSize;     // total mem size
    uint64_t seqIn;            // last enqueue sequence
    uint64_t seqOut;           // last dequeue sequence
    uint64_t frontOffset;      // front offset
    uint64_t tailOffset;       // tail offset
    uint64_t buffLimit;        // An offset where data cannot be written to the end of the queue
    uint64_t seqOutPre;        // prefetched sequence
    uint64_t frontOffsetPre;   // prefetched front offset
};

// item header definition
struct RmaShmData {
    uint64_t totalLen;         // item total length(B) = dataLen + RMA_SHM_DATA_HEAD
    uint64_t sequence;         // item sequence
    int32_t dataType;          // data type {0:float32}
    int32_t dimNum;            // data's dim num
    int64_t dims[RMA_DIM_MAX]; // shape value
    uint64_t dataLen;          // data length(B)
    uint64_t readyLen;         // data length(B) has been written to queue
};

bool Full(RmaShmHeader* queHeader, uint64_t dataSize);
int64_t GetShmAddr(std::string name, int rankId, int capacity);
void *GetHostAddr(std::string name);
void FreeShmAddr(int deviceId);
RmaShmData *ShmDequeuePre(RmaShmHeader* queHeader);
RmaShmData *ShmDequeue(RmaShmHeader* queHeader);
int64_t GetShmElemNum(RmaShmHeader* header);
uint64_t GetShmSeq(RmaShmHeader* queueHeader);
void ClearShmQueue();
RmaShmData *MallocFromShm(std::string channelName, int64_t dims[]);
uint8_t *GetDataAddr(RmaShmData* dataHeader);
void SetReadyLen(RmaShmData* dataHeader, uint64_t value);
uint8_t *ShmEnqueueHeadRaw(RmaShmHeader* header, int64_t dims[], uint64_t sequence);

#endif  // RMA_SHM_SVM_H