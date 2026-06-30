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

#ifndef LCCL_COLLECTIVES_H
#define LCCL_COLLECTIVES_H

#include <climits>

#include "datacopy_gm2gm.h"
#include "datacopy_gm2gm_delay.h"
#include "sync_collectives.h"
using namespace AscendC;
using namespace Lcal;

class Collectives {
    // UB header reserved space: accounts for sync flag header (64B) + padding to 32B alignment
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE;

public:
    // UB flag area size: 2KB reserved for intra/inter-card sync flags per block
    constexpr static int64_t UB_FLAG_SIZE = 2 * 1024;

    __aicore__ inline Collectives(int rank, int rankSize, uint32_t extraFlag, TPipe* pipePtr)
        : rank(rank),
          rankSize(rankSize),
          extraFlag(extraFlag),
          pipe(pipePtr)
    {
    }

    __aicore__ inline ~Collectives() {}

    template <typename T>
    __aicore__ inline void DataCopyWrap(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                        const uint32_t size)
    {
        if (size % UB_ALIGN_SIZE == 0) {
            DataCopy(dstGlobal, srcLocal, size / sizeof(T));
        } else {
            DataCopyExtParams copyParams{1, size, 0, 0, 0};
            DataCopyPad(dstGlobal, srcLocal, copyParams);
        }
    }

    template <typename T>
    __aicore__ inline void DataCopyWrap(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                        const uint32_t size)
    {
        if (size % UB_ALIGN_SIZE == 0) {
            DataCopy(dstLocal, srcGlobal, size / sizeof(T));
        } else {
            DataCopyExtParams copyParams{1, size, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
            DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
        }
    }

    /**
     * @brief 使用该接口的前提条件是，tbuf必须init了UB_SINGLE_DMA_SIZE_MAX大小，且这块大小的区域必须是连贯的，
     *          不能是超过了最大UB大小后折返回来。
     */
    template <typename T>
    __aicore__ inline void DataCopyWrap(const GlobalTensor<T>& inputGT, const GlobalTensor<T>& outputGT,
                                        int64_t dataSizeRemain, int op)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        LocalTensor<T> localUB = tBuf.GetWithOffset<T>(UB_SINGLE_DMA_SIZE_MAX / sizeof(T), UB_FLAG_SIZE);

        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        PipeBarrier<PIPE_ALL>();
        if (op != -1) {
            SetAtomicOpType<T>(op);
        }
        PipeBarrier<PIPE_ALL>();

        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_DMA_SIZE_MAX ? UB_SINGLE_DMA_SIZE_MAX : dataSizeRemain;
            DataCopyWrap(localUB, inputGT[inputOffsetNum], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);  // MTE3等MTE2
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyWrap(outputGT[outputOffsetNum], localUB, size);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);   // MTE2等MTE3
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);  // Scalar等MTE3
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        if (op != -1) {
            SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }

    /**
     * @tparam dataSizeRemain output size
     * @tparam T output type
     * @tparam U input type
     */
    template <typename T, typename U = T>
    __aicore__ inline void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& inputGT,
                                           const GlobalTensor<T>& outputGT, int op)
    {
        // 一般情况(U = T)，input/output相同，共用一块UB
        // 只有在需要转换(U->T)时，UB会按比例(sizeof(U):sizeof(T))分成input/output两块，并32对齐
        constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
        constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(T) + sizeof(U)) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
        constexpr int32_t inputUbBlockSize = std::is_same_v<T, U> ? ubBlockSize : ubAlignNum * sizeof(U);
        constexpr int32_t outputUbBlockSize = std::is_same_v<T, U> ? ubBlockSize : ubAlignNum * sizeof(T);

        __gm__ U* input = const_cast<__gm__ U*>(inputGT.GetPhyAddr());
        __gm__ T* output = const_cast<__gm__ T*>(outputGT.GetPhyAddr());
        __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET), (__ubuf__ U*)(UB_MID_OFFSET)};
        __ubuf__ T* outputUB[2] = {(__ubuf__ T*)inputUB[0], (__ubuf__ T*)inputUB[1]};
        if constexpr (!std::is_same_v<T, U>) {
            outputUB[0] = (__ubuf__ T*)(inputUB[0] + inputUbBlockSize / sizeof(U));
            outputUB[1] = (__ubuf__ T*)(inputUB[1] + inputUbBlockSize / sizeof(U));
        }
        int inputOffsetNum = 0;
        int outputOffsetNum = 0;
        if (dataSizeRemain <= 0) {
            return;
        }

        SetAtomic<T>(op);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2等MTE3
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > ubBlockSize ? ubBlockSize : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID1 : EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CpGM2UB(inputUB[1 - (i & 1)], input + inputOffsetNum, (uint32_t)(size));  // size是以sizeof(T)为单位的
            AscendC::SetFlag<HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_V>(eventId);
            CpUB2GM(output + outputOffsetNum, outputUB[i & 1], size);
            AscendC::SetFlag<HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::V_MTE3>(eventId);
            CpUB2GM(output + outputOffsetNum, outputUB[1 - (i & 1)], size);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(U));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2等MTE3

        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);  // Scalar等MTE3
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);

        UnsetAtomic(op);
        return;
    }

public:
    int rank;
    int rankSize;
    int localRank = 0;
    int localRankSize = 0;  // On 910A5: max cards on one board; on 910B: cards within a single machine
    int xRankSize = 0;
    int yRankSize = 0;
    int xRankIdx = 0;
    int yRankIdx = 0;
    uint32_t extraFlag;
    int root;
    int64_t len;
    int64_t magic;
    GM_ADDR scale;
    int64_t blockIdx;                        // 当前aicore序号
    int64_t blockNum;                        // 当前rank的总aicore数
    GM_ADDR shareAddrs[LCAL_MAX_RANK_SIZE];  // 共享内存地址列表
    TPipe* pipe;                             // pipe工具类
    TBuf<QuePosition::VECCALC> tBuf;
    SyncCollectives sync;

    template <typename T>
    __aicore__ inline void SetAtomic(int op)
    {
        PipeBarrier<PIPE_ALL>();
        if (op != -1) {
            SetAtomicOpType<T>(op);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void UnsetAtomic(int op)
    {
        if (op != -1) {
            AscendC::SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }
};

#endif  // LCCL_COLLECTIVES_H
