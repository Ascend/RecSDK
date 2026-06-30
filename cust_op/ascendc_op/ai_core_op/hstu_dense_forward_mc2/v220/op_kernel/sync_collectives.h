/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/
#ifndef LCCL_SYNC_H
#define LCCL_SYNC_H

#include "comm_args.h"

using namespace AscendC;
using namespace Lcal;

// Sync flag segment size (in int64_t units)
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
// Bit shift for embedding magic number as upper 32 bits of comparison value
constexpr int64_t MAGIC_OFFSET = 32;

constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR* shareAddrs, TBuf<QuePosition::VECCALC>& tBuf)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = block_idx;
        this->blockNum = block_num;
        // Single segment length (in int64_t)
        segmentCount = block_num * FLAG_UNIT_INT_NUM;
        // Initialize inner-block and inter-block sync addresses for current core
        localSyncAddr = reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]);
        basicSyncAddr = reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + block_idx * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr =
            reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + segmentCount + block_idx * FLAG_UNIT_INT_NUM;
        this->tBuf = tBuf;
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    /**
     * @brief 设置指定卡的指定eventID的flag，设置的值为 magic 和 value 组合而成的值。
     * @param magic 算子批次，最终会组合到要set的flag的数值中高32位去
     * @param value 具体的最终要set的flag的数值中低32位的值
     * @param eventID 实际上从物理地址来看，是以共享内存首地址起往后的偏移量（要进行缩放，不是偏移量绝对值）。
     * @param rank 这个rank是在CommArgs结构体内peerMems数组内对应的rankId，并非global或local的id。
     *              （91093场景local不适用，910B多机场景global不适用。）
     */
    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    /**
     * @brief 等待指定卡的指定eventID的flag变为 magic 和 value 组合而成的值。
     * @param magic 算子批次，最终会组合到要wait的flag的数值中高32位去
     * @param value 具体的最终要wait的flag的数值中低32位的值
     * @param eventID 实际上从物理地址来看，是以共享内存首地址起往后的偏移量。
     * @param rank 这个rank是在CommArgs结构体内peerMems数组内对应的rankId，并非global或local的id。
     *              （91093场景local不适用，910B多机场景global不适用。）
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    /**
     * @brief 相比起WaitSyncFlag函数，额外允许 远端Flag > 期望要check的FlagValue的值 通过校验。
     */
    __aicore__ inline void WaitSyncGreaterFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v,
                            false);
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1,
                            v);
    }

    __aicore__ inline void WaitSyncGreaterFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1,
                            v, false);
    }

    /**
     * @brief 等待指定卡的指定eventID往后的flagNum个flag变为 magic 和 value 组合而成的值。<br>
     *          注：[eventID, eventID + flagNum)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank, int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum,
                            v);
    }

    __aicore__ inline void WaitSyncGreaterFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank,
                                               int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum,
                            v, false);
    }

    // Set single-card inner sync flag (memory region A)
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    // Wait for single-card inner sync flag (memory region A)
    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1,
                            value);
    }

    // Wait for all inner sync flags across the entire rank (memory region A)
    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[waitRank]), value);
    }

    // Check all inner sync flags across the entire rank (memory region A)
    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[waitRank]), value);
    }

    // Set single-block inter-card sync flag (memory region B)
    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    // Wait for single-block inter-card sync flag (memory region B)
    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    // Wait for all inter-card sync flags within one rank (memory region B)
    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }

    // Wait for partial outer flags across all ranks starting from startBlock (memory region B)
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // Staggered rank read to avoid multi-core concurrent copy contention
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    // Check partial outer flags across all ranks starting from startBlock (memory region B)
    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock,
                                                     int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // Staggered rank read to avoid multi-core concurrent copy contention
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    // Wait for all inter-card sync flags across all ranks (memory region B)
    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // Check all inter-card sync flags across all ranks (memory region B)
    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // Low-level interface: set sync flag value at given address
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);

        // Copy local flag value to global address
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // Wait for SetValue to complete
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // Wait for UB->GM copy to complete
    }

    // Low-level interface: wait for sync flag at given address
    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    // Read one flag and return as immediate value
    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        // Copy flag from global to local
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB copy to complete

        int64_t res = localWait.GetValue(0);
        return res;
    }

    // Wait for multiple consecutive inner sync flags for one card
    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    // Read inner flag (memory region A) for a specific rank and block
    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag(reinterpret_cast<__gm__ int64_t*>(shareAddrs[waitRank]) + segmentCount +
                       waitBlock * FLAG_UNIT_INT_NUM);
    }

private:
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        // Pack magic as upper 32 bits and value as lower 32 bits for comparison
        return (static_cast<int64_t>(magic) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return reinterpret_cast<__gm__ int64_t*>(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return reinterpret_cast<__gm__ int64_t*>(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    /**
     * @brief Wait for a subset of sync flags within one rank.
     * @param waitAddr    Address of the first flag to wait on (inclusive)
     * @param flagNum     Number of flags to wait on
     * @param checkValue  Expected flag value
     * @param mustEqual   When remote flag >= checkValue, controls further comparison logic.<br>
     *                    true: MAGIC_MASK must be strictly equal;
     *                    false: remote MAGIC_MASK >= checkValue's MAGIC_MASK is accepted.
     */
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue,
                                               bool mustEqual = true)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        int64_t checkedFlagNum = 0;
        do {
            int64_t remainToCheck = flagNum - checkedFlagNum;
            // Copy sync flags from global to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM], remainToCheck * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB copy

            // Check if all sync flags match checkValue
            isSync = true;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core hasn't reached checkValue yet
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((mustEqual && (v < checkValue || ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK)))) ||
                    ((!mustEqual) && (v < checkValue || ((v & MAGIC_MASK) < (checkValue & MAGIC_MASK))))) {
                    isSync = false;
                    break;
                }
                checkedFlagNum++;
            }
        } while (!isSync);
    }

    // Wait for all sync flags within one rank
    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    // Check a subset of sync flags within one rank (single copy only)
    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        // Copy sync flags from global to local
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB copy
        // Check if all sync flags match checkValue
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            // Continue if any core hasn't reached checkValue yet
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        return isSync;
    }

    // Check all sync flags within one rank (single copy only)
    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR* shareAddrs;
    int64_t segmentCount;  // Length of one sync flag segment (in int64_t units)
    __gm__ int64_t* localSyncAddr;
    __gm__ int64_t* basicSyncAddr;       // Inner-card sync flag address for current block
    __gm__ int64_t* blockOuterSyncAddr;  // Inter-card sync flag address for current block
    TBuf<QuePosition::VECCALC> tBuf;
};

#endif  // LCCL_SYNC_H
