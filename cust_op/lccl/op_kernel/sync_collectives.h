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

#ifndef LCCL_SYNC_H
#define LCCL_SYNC_H

#include "comm_args.h"

using namespace AscendC;

// 同步标志位占用长度
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
// 每个同步单位占用内存大小（Bytes）
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
// magic作为比较值时高位偏移量
constexpr int64_t MAGIC_OFFSET = 32;

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs, int blockIdx, int blockNum)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = blockIdx;
        this->blockNum = blockNum;
        // 单个标志段长度
        segmentCount = blockNum * FLAG_UNIT_INT_NUM;
        // 初始化当前核对应的卡内/卡间同步地址
        blockInnerSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + blockIdx * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + blockIdx * FLAG_UNIT_INT_NUM;
        // 初始化标志位数据搬运队列，一次最多可搬运blockNum个标志
        pipe.InitBuffer(syncSetQue, PING_PONG_SIZE, blockNum * SYNC_UNIT_SIZE);
        pipe.InitBuffer(syncWaitQue, PING_PONG_SIZE, blockNum * SYNC_UNIT_SIZE);
    }

    // 设置单个卡内同步标志（内存A）
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = GetFlagValue(magic, eventID);
        SetFlag(blockInnerSyncAddr, value);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = GetFlagValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    // 等待单个卡内同步标志（内存A）
    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = GetFlagValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    // 等待整个rank内所有卡内同步标志（内存A）
    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = GetFlagValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // 检验整个rank内所有卡内同步标志（内存A）
    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = GetFlagValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // 设置单个卡间同步标志（内存B）
    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = GetFlagValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = GetFlagValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    // 等待单个卡间同步标志（内存B）
    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = GetFlagValue(magic, eventID);
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    // 等待整个rank内所有卡间同步标志（内存B）
    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = GetFlagValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }

    // 等待所有rank从startBlock开始的flagNum个卡间同步标志（内存B）
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = GetFlagValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // 错峰读取rank标志，防止多核并发拷贝影响性能
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    // 检验所有rank从startBlock开始的flagNum个卡间同步标志（内存B）
    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = GetFlagValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // 错峰读取rank标志，防止多核并发拷贝影响性能
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    // 等待所有rank的所有卡间同步标志，全rank同步（内存B）
    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // 检验所有rank的所有卡间同步标志，全rank同步（内存B）
    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // 低级接口，设置同步标志
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = syncSetQue.AllocTensor<int64_t>();
        localSet.SetValue(0, setValue);

        // 将global同步标识拷贝至local
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);  // 等待SetValue完成
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);  // 等待UB->GM完成

        syncSetQue.FreeTensor(localSet);
    }

    // 低级接口，等待同步标志
    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    // 读取一个标志位，返回立即数
    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = syncWaitQue.AllocTensor<int64_t>();
        // 将global拷贝至local
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);  // 等待GM->UB

        int64_t res = localWait.GetValue(0);
        syncWaitQue.FreeTensor(localWait);
        return res;
    }

    // 获取单个卡内多个连续的同步标志
    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = GetFlagValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    // 获取单个卡内同步标志（内存A）
    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + segmentCount + waitBlock * FLAG_UNIT_INT_NUM);
    }

public:
    __aicore__ inline int64_t GetFlagValue(int32_t magic, int32_t eventID)
    {
        // magic作为高位，eventID作为低位，组成一个value值用于比较
        return (static_cast<int64_t>(magic) << MAGIC_OFFSET) + static_cast<int64_t>(eventID);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    // 等待一个rank内部分同步标志
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = syncWaitQue.AllocTensor<int64_t>();
        bool isSync = true;
        do {
            // 将global同步标识拷贝至local
            DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);  // 等待GM->UB

            // 检验同步标识是否为checkValue
            isSync = true;
            for (auto i = 0; i < flagNum; ++i) {
                // 当有core未达到checkValue的阶段时，继续等待
                if (localWait.GetValue(i * FLAG_UNIT_INT_NUM) < checkValue) {
                    isSync = false;
                    break;
                }
            }
        } while (!isSync);
        syncWaitQue.FreeTensor(localWait);
    }

    // 等待一个rank内所有同步标志
    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    // 检验一个rank内部分同步标志，仅拷贝一次
    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = syncWaitQue.AllocTensor<int64_t>();
        // 将global同步标识拷贝至local
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);  // 等待GM->UB
        // 检验同步标识是否为checkValue
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            // 当有core未达到checkValue的阶段时，继续等待
            if (localWait.GetValue(i * FLAG_UNIT_INT_NUM) < checkValue) {
                isSync = false;
                break;
            }
        }
        syncWaitQue.FreeTensor(localWait);
        return isSync;
    }

    // 检验一个rank内所有同步标志，仅拷贝一次
    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }
    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR *shareAddrs;
    int64_t segmentCount;  // 一组同步标志段的长度（int64_t类型计数）
    __gm__ int64_t* blockInnerSyncAddr;  // 当前block卡内同步标志地址
    __gm__ int64_t* blockOuterSyncAddr;  // 当前block卡间同步标志地址
    TQue<QuePosition::VECOUT, PING_PONG_SIZE> syncSetQue;  // 从local拷贝同步标志至global的队列
    TQue<QuePosition::VECIN, PING_PONG_SIZE> syncWaitQue;  // 从global拷贝同步标志至local的队列
    TPipe pipe;
};

#endif //LCCL_SYNC_H
