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

#ifndef LCCL_COLLECTIVES_H
#define LCCL_COLLECTIVES_H

#include <climits>

#include "sync_collectives.h"
#include "datacopy_gm2gm.h"
using namespace AscendC;


class Collectives {
public:
    FORCE_INLINE_AICORE Collectives(int rank, int rankSize, uint32_t extraFlag)
        : rank(rank), rankSize(rankSize), extraFlag(extraFlag) {}

    FORCE_INLINE_AICORE void Init(GM_ADDR input, GM_ADDR output, GM_ADDR commArgs,
                                  int64_t len, int64_t magic, int op, int root, int cycleCount)
    {
        GlobalTensor<GM_ADDR> peerMemsAddrGm;
        peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0],
                LCAL_MAX_RANK_SIZE);
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = peerMemsAddrGm.GetValue(i) +
                            (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }

        this->root = root;
        this->len = len;
        this->magic = magic;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum();

        sync.Init(rank, rankSize, shareAddrs, blockIdx, blockNum);
    }

public:
    template <typename T>
    FORCE_INLINE_AICORE void CpGM2GM(const GlobalTensor<T>& outputGT, const GlobalTensor<T>& inputGT,
                                     const uint32_t calCount, int op)
    {
        DataCopyGM2GM<T> cpKernel;
        cpKernel.Init(outputGT, inputGT, calCount, op);
        cpKernel.Process();
    }
    template <typename T>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<T>& inputGT,
                                             const GlobalTensor<T>& outputGT, int op)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        __gm__ T *input = const_cast<__gm__ T *>(inputGT.GetPhyAddr());
        __gm__ T *output = const_cast<__gm__ T *>(outputGT.GetPhyAddr());
        // each UB's size is 95 KB
        const int64_t firstUBStart = 96;
        const int64_t secondUBStart = 97440;
        __ubuf__ T* inputUB[2] = {(__ubuf__ T*)get_imm(firstUBStart), (__ubuf__ T*)get_imm(secondUBStart)};
        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        pipe_barrier(PIPE_ALL);
        if (op != COPYONLY) {
            SetAtomicDataType<T>();
#ifdef __DAV_C220_VEC__
            SetAtomicOpType(op);
#endif
        }
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); // MTE2等MTE3
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); // MTE2等MTE3
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_PING_PONG_ADD_SIZE_MAX ?
                            UB_SINGLE_PING_PONG_ADD_SIZE_MAX : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size);
            set_flag(PIPE_MTE2, PIPE_MTE3, eventId);  // MTE3等MTE2
            wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
            CpUB2GM(output + outputOffsetNum, (i & 1) ? inputUB[0] : inputUB[1], size);
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId);

            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); // MTE2等MTE3
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); // MTE2等MTE3

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID3); // Scalar等MTE3
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID3);
        if (op != COPYONLY) {
            set_atomic_none();
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    template <typename T>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, GM_ADDR inputGT,
                                             GM_ADDR outputGT, int op)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        __gm__ T *input = (__gm__ T *)inputGT;
        __gm__ T *output = (__gm__ T *)outputGT;
        // each UB's size is 95 KB
        const int64_t firstUBStart = 96;
        const int64_t secondUBStart = 97440;
        __ubuf__ T* inputUB[2] = {(__ubuf__ T*)get_imm(firstUBStart), (__ubuf__ T*)get_imm(secondUBStart)};
        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        pipe_barrier(PIPE_ALL);
        if (op != COPYONLY) {
            SetAtomicDataType<T>();
#ifdef __DAV_C220_VEC__
            SetAtomicOpType(op);
#endif
        }
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); // MTE2等MTE3
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); // MTE2等MTE3
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_PING_PONG_ADD_SIZE_MAX ?
                            UB_SINGLE_PING_PONG_ADD_SIZE_MAX : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size);
            set_flag(PIPE_MTE2, PIPE_MTE3, eventId);  // MTE3等MTE2
            wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
            CpUB2GM(output + outputOffsetNum, (i & 1) ? inputUB[0] : inputUB[1], size);
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId);

            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); // MTE2等MTE3
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); // MTE2等MTE3

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID3); // Scalar等MTE3
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID3);
        if (op != COPYONLY) {
            set_atomic_none();
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    template<typename T>
    FORCE_INLINE_AICORE void SetSingleValue2Gm(GM_ADDR gm, T value)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        // each UB's size is 95 KB
        const int64_t UBStart = 96;
        __ubuf__ T *inputUB = (__ubuf__ T*)(UBStart);
        *inputUB = value;
        AscendC::PipeBarrier<PIPE_ALL>();
        CpUB2GM((__gm__ T *)gm, inputUB, sizeof(T));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

protected:
    int rank;
    int rankSize;
    uint32_t extraFlag;
    int root;
    int64_t len;
    int64_t magic;
    int64_t blockIdx;  // 当前aicore序号
    int64_t blockNum;  // 当前rank的总aicore数
    GM_ADDR shareAddrs[LCAL_MAX_RANK_SIZE];  // 共享内存地址列表
    TPipe pipe;  // pipe工具类
    SyncCollectives sync;

private:
    template <typename T>
    FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
    {
#ifndef __GET_CODE_CHANNEL__
        copy_ubuf_to_gm_align_b8(gmAddr, ubAddr, 0, 1, size, 0, 0, 0, 0);
#endif
    }

    template <typename T>
    FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
    {
#ifndef __GET_CODE_CHANNEL__
        copy_gm_to_ubuf_align_b8(ubAddr, gmAddr, 0, 1, size, 0, 0, 0, 0);
#endif
    }
};

// CeilDiv
template <typename T1, typename T2>
FORCE_INLINE_AICORE T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

// 32字节对齐
FORCE_INLINE_AICORE int64_t Align(int64_t len)
{
    return CeilDiv(len, ALIGN_SIZE) * ALIGN_SIZE;
}

#endif  // LCCL_COLLECTIVES_H