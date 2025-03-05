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

#include <string>
#include <array>

constexpr int32_t RMA_SHM_HEAD_LEN = 128; // queue's head length
constexpr int32_t RMA_SHM_DATA_HEAD = 56; // queue item's head length
constexpr int32_t RMA_SHM_READY_LEN = 48; // offset of readyLen in item's head
constexpr int32_t RMA_DIM_MAX = 2;
constexpr int MAX_WAIT_LOOP = 1800;
constexpr int SHM_WR_OWN = 0600;
constexpr int SHM_WR_ALL = 0666;

namespace MxRec {
enum class RmaDevModel {
    SVM_MAP_DEV,    // 910_93
    PCIE_TH_DEV     // 910B
};

// Queue header definition
struct RmaShmHeader {
    uint64_t queueCapacity = 0;    // depth of queue
    uint64_t totalMemSize = 0;     // total mem size
    uint64_t seqIn = 0;            // last enqueue sequence
    uint64_t seqOut = 0;           // last dequeue sequence
    uint64_t frontOffset = 0;      // front offset
    uint64_t tailOffset = 0;       // tail offset
    uint64_t buffLimit = 0;        // An offset where data cannot be written to the end of the queue
    uint64_t seqOutPre = 0;        // prefetched sequence
    uint64_t frontOffsetPre = 0;   // prefetched front offset
};
// item header definition
struct RmaShmData {
    uint64_t totalLen = 0;         // item total length(B) = dataLen + RMA_SHM_DATA_HEAD
    uint64_t sequence = 0;         // item sequence
    int32_t dataType = 0;          // data type {0:float32}
    int32_t dimNum = 0;            // data's dim num
    std::array<int64_t, RMA_DIM_MAX> dims = {0, 0};
    uint64_t dataLen = 0;          // data length(B)
    uint64_t readyLen = 0;         // data length(B) has been written to queue
};

int64_t GetShmAddr(std::string& name, int rankId, int capacity);
void *GetHostAddr(std::string& name);
void FreeShmAddr(uint32_t deviceId);
RmaShmData *ShmDequeuePre(RmaShmHeader* queHeader);
RmaShmData *ShmDequeue(RmaShmHeader* queHeader);
uint64_t GetShmSeq(RmaShmHeader* queueHeader);
void ClearShmQueue();
RmaShmData *MallocFromShm(std::string& channelName, std::array<int64_t, RMA_DIM_MAX>& dims);
uint8_t *GetDataAddr(RmaShmData* dataHeader);
void SetReadyLen(RmaShmData* dataHeader, uint64_t value);
uint8_t *ShmEnqueueHeadRaw(RmaShmHeader* header, std::array<int64_t, RMA_DIM_MAX>& dims, uint64_t sequence);
}  // namespace MxRec
#endif  // RMA_SHM_SVM_H