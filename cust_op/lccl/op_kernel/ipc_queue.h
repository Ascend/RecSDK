/**
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

#ifndef LCCL_IPC_QUEUE_H
#define LCCL_IPC_QUEUE_H

#include "sync_collectives.h"
using namespace AscendC;

template <typename T>
class IpcQueue {
public:
    FORCE_INLINE_AICORE IpcQueue() {}

    FORCE_INLINE_AICORE void Init(SyncCollectives *sync, int64_t magic, GM_ADDR workSpace, uint64_t bufferNum,
                                  uint64_t blockNum)
    {
        this->sync = sync;
        this->magic = magic;
        depth = bufferNum / blockNum;
        front = 0;
        rear = 0;
        count = 0;
        this->blockNum = blockNum;
        buff.SetGlobalBuffer((__gm__ T*)workSpace, bufferNum);
        blockIdx = GetBlockIdx();
    }

    FORCE_INLINE_AICORE bool Full()
    {
        if ((rear + 1) % depth == front) {
            return true;
        }
        return false;
    }

    /**
     * @brief 从队列尾申请一块内存，返回地址
     */
    FORCE_INLINE_AICORE GlobalTensor<T> EnQue()
    {
        uint64_t rearOld = rear;
        rear = (rear + 1) % depth;
        return buff[rearOld * blockNum];
    }

    /**
     * @brief 校验单个读端是否完成队列头的读取
     * @param checkRank     校验的读端rank号
     * @param checkBlock    校验的读端blockIdx，缺省时使用和当前相同的blockIdx
     */
    FORCE_INLINE_AICORE void DeQue(int checkRank, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        // 校验其他卡都已完成队列头的数据搬运
        sync->WaitInnerFlag(magic, count, checkRank, checkBlock);  // 后续改成IPC单独的标志位
        pipe_barrier(PIPE_ALL);

        int64_t val = sync->GetInnerFlag(checkRank, checkBlock) & EVENT_ID_MASK;
        count = val + 1;
        front = (val + 1) % depth;
    }

    /**
     * @brief 校验多个读端是否完成队列头的数据搬运
     * @param checkRank     校验的读端rank号
     * @param checkBlock    校验的读端blockIdx，缺省时使用和当前相同的blockIdx
     */
    FORCE_INLINE_AICORE void DeQue(int *rankList, int checkCount, int checkBlock = -1)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        // 校验其他卡都已完成队列头的数据搬运
        int64_t minIndex = LLONG_MAX;
        for (int i = 0; i < checkCount; ++i) {
            sync->WaitInnerFlag(magic, count, rankList[i], checkBlock);  // 后续改成IPC单独的标志位
            pipe_barrier(PIPE_ALL);
            int64_t val = sync->GetInnerFlag(rankList[i], checkBlock) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        front = (val + 1) % depth;
    }

    /**
     * @brief 校验多个读端是否完成队列头的读取
     * @param rankList      校验多个读端的rank号数组首地址
     * @param checkCount    数组长度
     * @param checkBlock    校验的读端blockIdx，缺省时使用和当前相同的blockIdx
     */
    FORCE_INLINE_AICORE void DeQue(int *rankList, int checkCount, int checkBlock, int sliceIdx)
    {
        if (!Full()) {
            return;
        }
        if (checkBlock == -1) {
            checkBlock = blockIdx;
        }
        // 校验其他卡都已完成队列头的数据搬运
        int64_t minIndex = LLONG_MAX;

        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, sliceIdx-1, rankList[i], checkBlock);  // 后续改成IPC单独的标志位
            pipe_barrier(PIPE_ALL);

            int64_t val = sync->GetInnerFlag(rankList[i], checkBlock) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        front = (minIndex + 1) % depth;
    }

    /**
     * @brief 校验多个读端是否完成队列头的读取
     * @param rankList      校验多个读端的rank号数组首地址
     * @param blockIdxList  校验的读端blockIdx数组首地址
     * @param checkCount    数组长度
     */
    FORCE_INLINE_AICORE void DeQue(int *rankList, int *blockIdxList, int checkCount)
    {
        if (!Full()) {
            return;
        }
        // 校验其他卡都已完成队列头的数据搬运
        int64_t minIndex = LLONG_MAX;

        for (int i = 0; i < checkCount; i++) {
            sync->WaitInnerFlag(magic, count, rankList[i], blockIdxList[i]);  // 后续改成IPC单独的标志位
            pipe_barrier(PIPE_ALL);

            int64_t val = sync->GetInnerFlag(rankList[i], blockIdxList[i]) & EVENT_ID_MASK;
            if (minIndex > val) {
                minIndex = val;
            }
        }
        count = minIndex + 1;
        front = (minIndex + 1) % depth;
    }

    /**
     * @brief 仅读端使用，返回队列头的地址，记录了读端自己历史处理到的状态，没有别的含义。
     */
    FORCE_INLINE_AICORE GlobalTensor<T> ReadFront()
    {
        uint64_t frontOld = front;
        front = (front + 1) % depth;
        return buff[frontOld * blockNum];
    }

private:
    int64_t magic;
    uint64_t depth;     // 队列深度，以blockSize为单位
    uint64_t front;     // 头指针
    uint64_t rear;      // 尾指针
    uint64_t count;     // 头指针对应的数据块偏移
    uint64_t blockNum; // 搬运粒度
    GlobalTensor<T> buff;
    SyncCollectives *sync;
    int blockIdx;
};

#endif  // LCCL_IPC_QUEUE_H
