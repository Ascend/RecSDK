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

#ifndef LCCL_DATACOPY_GM2GM_DELAY_H
#define LCCL_DATACOPY_GM2GM_DELAY_H
#include "datacopy_gm2gm.h"

using namespace AscendC;
using namespace Lcal;
/**
 * @brief Delay Scaling scheme overview:
 * Scale values are pre-loaded to UB and kept resident.
 * Execution pipeline: int8 cast to fp16 => fp16 / scale => fp16 add => fp16 * min_scale => fp16 cast to int8
 * @tparam T add type
 * @tparam U input type
 * @tparam V output type
 *
 */
template <typename V, typename T, typename U = T>
class DataCopyGM2GMDelay {
    constexpr static int64_t THREE_NUM = 3;
    constexpr static int64_t FOUR_NUM = 4;
    constexpr static int32_t WORK_OFFSET = 8192;  // Compute granularity in bytes. 2KB outperforms 1KB in practice.
    constexpr static int32_t WORK_BLOCK_NUM = WORK_OFFSET / sizeof(T);  // Compute granularity in elements (1KB = 1024B)
    constexpr static int32_t UB_HEAD_OFFSET = WORK_OFFSET * 2;
    constexpr static int32_t SCALE_SIZE = 32;                     // Scale buffer size, 32B
    constexpr static int32_t SCALE_NUM = SCALE_SIZE / sizeof(T);  // Number of scale elements (including FP8 data)
    constexpr static int32_t BLOCK_NUM = (UB_SINGLE_DMA_SIZE_MAX - WORK_OFFSET * 2 - SCALE_SIZE * 4) / 2 /
                                         (sizeof(U) + sizeof(T)) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;  // UB_ALIGN_SIZE=32
    constexpr static int32_t IN_BLOCKSIZE = BLOCK_NUM * sizeof(U);  // Input block size (including FP8 data)

    // Delay scaling implementation: incoming FP8 data carries only one scale value
public:
    __aicore__ inline DataCopyGM2GMDelay() {}

    __aicore__ inline void Init(GlobalTensor<V>& outputGt, GlobalTensor<U> (&inputGt)[8],
                                GlobalTensor<U> (&inputScaleGt)[8], const uint32_t calNum, int rankCount,
                                GlobalTensor<U>& outScaleGt, TBuf<QuePosition::VECCALC> tbuf)
    {
        for (int index = 0; index < rankCount; index++) {
            this->inputGt[index] = inputGt[index];
            this->inputScaleGt[index] = inputScaleGt[index];
        }
        this->outputGt = outputGt;
        this->outScaleGt = outScaleGt;
        inTensor[0] = tbuf.GetWithOffset<U>(BLOCK_NUM, 0);  // FP8 ping/pong input and scale values
        inTensor[1] = tbuf.GetWithOffset<U>(BLOCK_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM + IN_BLOCKSIZE * THREE_NUM);
        singleScaleUBTensor[0] = tbuf.GetWithOffset<T>(SCALE_NUM, IN_BLOCKSIZE);  // Computed single scale value (ping)
        singleScaleUBTensor[1] =
            tbuf.GetWithOffset<T>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM + IN_BLOCKSIZE * FOUR_NUM);
        singleScaleUUBTensor[0] =
            tbuf.GetWithOffset<U>(SCALE_NUM, IN_BLOCKSIZE);  // Computed single scale value (ping), U type
        singleScaleUUBTensor[1] =
            tbuf.GetWithOffset<U>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM + IN_BLOCKSIZE * FOUR_NUM);
        scaleUBTensor[0] = tbuf.GetWithOffset<T>(SCALE_NUM, IN_BLOCKSIZE + SCALE_SIZE);  // All scale values (ping)
        scaleUBTensor[1] =
            tbuf.GetWithOffset<T>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * THREE_NUM + IN_BLOCKSIZE * FOUR_NUM);
        scaleUUBTensor[0] =
            tbuf.GetWithOffset<U>(SCALE_NUM, IN_BLOCKSIZE + SCALE_SIZE);  // All scale values (ping), U type
        scaleUUBTensor[1] =
            tbuf.GetWithOffset<U>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * THREE_NUM + IN_BLOCKSIZE * FOUR_NUM);
        workUBTensor[0] =
            tbuf.GetWithOffset<T>(WORK_BLOCK_NUM, IN_BLOCKSIZE + SCALE_SIZE * HALF_NUM);  // Work buffer (ping)
        workUBTensor[1] =
            tbuf.GetWithOffset<T>(WORK_BLOCK_NUM, WORK_OFFSET + SCALE_SIZE * FOUR_NUM + IN_BLOCKSIZE * FOUR_NUM);
        outputUBTensor[0] = tbuf.GetWithOffset<T>(BLOCK_NUM, IN_BLOCKSIZE + SCALE_SIZE * HALF_NUM + WORK_OFFSET);
        outputUBTensor[1] =
            tbuf.GetWithOffset<T>(BLOCK_NUM, WORK_OFFSET * HALF_NUM + SCALE_SIZE * FOUR_NUM + IN_BLOCKSIZE * FOUR_NUM);

        this->rankCount = rankCount;
        totalDataSize = calNum * sizeof(U);
        this->calNum = calNum;
    }

    __aicore__ inline void PreProcess()
    {
        for (int index = 0; index < rankCount; index++) {
            DataCopyWrap(scaleUUBTensor[0][index * SCALE_SIZE / sizeof(U)], inputScaleGt[index],
                         SCALE_SIZE);  // MTE2 pipe
            pipe_barrier(PIPE_ALL);
            DataCopyWrap(scaleUUBTensor[1][index * SCALE_SIZE / sizeof(U)], inputScaleGt[index],
                         SCALE_SIZE);  // MTE2 pipe
            pipe_barrier(PIPE_ALL);
        }
        for (int index = 0; index < rankCount; index++) {
            scaleUBTensor[0][index].SetValue(0, scaleUBTensor[0].GetValue(index * SCALE_SIZE / sizeof(T)));
            pipe_barrier(PIPE_ALL);
            scaleUBTensor[1][index].SetValue(0, scaleUBTensor[1].GetValue(index * SCALE_SIZE / sizeof(T)));
            pipe_barrier(PIPE_ALL);
            outputUBTensor[0][index].SetValue(0, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        Div(scaleUBTensor[1], outputUBTensor[0], scaleUBTensor[1], rankCount);  // Compute reciprocal of scale
        AscendC::PipeBarrier<PIPE_ALL>();
        // Compute min scale across all 8 cards' original scale values
        ReduceMin<T>(singleScaleUBTensor[0], scaleUBTensor[0], workUBTensor[1][WORK_BLOCK_NUM / HALF_NUM], rankCount,
                     false);
        pipe_barrier(PIPE_ALL);
        // Write min scale to local card's IPC buffer, overwriting original scale value
        DataCopyWrap(outScaleGt, singleScaleUUBTensor[0], sizeof(T));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoopUncastAndMul(int idx, int index, event_t eventId)
    {
        PipeBarrier<PIPE_V>();
        T scalarValue = scaleUBTensor[1].GetValue(index);  // Per-rank scale value; dequant = divide by scale
        PipeBarrier<PIPE_V>();
        int32_t perRankNum;  // vloop instruction translation
        PipeBarrier<PIPE_V>();
        for (int j = 0; perRankNumRemain > 0; j++) {
            PipeBarrier<PIPE_V>();
            perRankNum = perRankNumRemain >= WORK_BLOCK_NUM ? WORK_BLOCK_NUM : perRankNumRemain;
            PipeBarrier<PIPE_V>();
            perRankNumRemain -= perRankNum;
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
            PipeBarrier<PIPE_V>();
            Cast((idx & 1) ? workUBTensor[0] : workUBTensor[1],
                 (idx & 1) ? inTensor[0][j * WORK_BLOCK_NUM] : inTensor[1][j * WORK_BLOCK_NUM], RoundMode::CAST_NONE,
                 perRankNum);
            PipeBarrier<PIPE_V>();
            if (index == 0) {
                Muls<T>((idx & 1) ? outputUBTensor[0][j * WORK_BLOCK_NUM] : outputUBTensor[1][j * WORK_BLOCK_NUM],
                        (idx & 1) ? workUBTensor[0] : workUBTensor[1], scalarValue, perRankNum);
            } else {
                Axpy<T, T>((idx & 1) ? outputUBTensor[0][j * WORK_BLOCK_NUM] : outputUBTensor[1][j * WORK_BLOCK_NUM],
                           (idx & 1) ? workUBTensor[0] : workUBTensor[1], scalarValue, perRankNum);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Mte3Process(int idx, int index, int calCount, event_t eventId)
    {
        if (index == (rankCount - 1)) {  // Last layer: output in high precision directly
            if constexpr (std::is_same_v<V, T>) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
                DataCopyWrap(outputGt[idx * BLOCK_NUM], (idx & 1) ? outputUBTensor[0] : outputUBTensor[1],
                             calCount * sizeof(V));  // MTE3 pipe
            }
            if constexpr (std::is_same_v<V, U>) {
                // Non-last layer: convert back to low precision
                PipeBarrier<PIPE_V>();
                T scaleValue = singleScaleUBTensor[0].GetValue(0);
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
                PipeBarrier<PIPE_V>();
                Muls<T>((idx & 1) ? outputUBTensor[0] : outputUBTensor[1],
                        (idx & 1) ? outputUBTensor[0] : outputUBTensor[1], scaleValue, calCount);
                PipeBarrier<PIPE_V>();
                Cast((idx & 1) ? inTensor[0] : inTensor[1], (idx & 1) ? outputUBTensor[0] : outputUBTensor[1],
                     RoundMode::CAST_NONE, calCount);
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
                DataCopyWrap(outputGt[idx * BLOCK_NUM], (idx & 1) ? inTensor[0] : inTensor[1],
                             calCount * sizeof(V));  // MTE3 pipe
            }
        }
    }

    __aicore__ inline int GetSize(int idx, int numOfPiece)
    {
        int size;
        if (idx < (numOfPiece - 1)) {
            size = IN_BLOCKSIZE;
        } else if (idx == (numOfPiece - 1)) {
            size = totalDataSize - (numOfPiece - 1) * IN_BLOCKSIZE;
        } else {
            size = 0;
        }
        return size;
    }

    __aicore__ inline void Process()
    {
        PreProcess();
        int numOfPiece = CeilDiv<int32_t, int32_t>(calNum, BLOCK_NUM);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
        for (int64_t i = 0; i < numOfPiece; i += HALF_NUM) {   // 数据分片数目
            for (int index = 0; index < rankCount; index++) {  // 轮流拉每个卡
                for (int k = 0; k < HALF_NUM; k++) {
                    int idx = i + k;
                    int size = GetSize(idx, numOfPiece);
                    int32_t calCount = size / sizeof(U);
                    perRankNumRemain = calCount;
                    event_t eventId = (idx & 1) ? EVENT_ID0 : EVENT_ID1;
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventId);
                    DataCopyWrap((idx & 1) ? inTensor[0] : inTensor[1], inputGt[index][BLOCK_NUM * idx], size);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                    LoopUncastAndMul(idx, index, eventId);
                    Mte3Process(idx, index, calCount, eventId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventId);
                }
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    }

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

private:
    int64_t totalDataSize = 0;
    int rankCount = 0;
    int perRankNumRemain = 0;
    int calNum = 0;
    int rankId = 0;
    int numLayer = 0;

    LocalTensor<U> inTensor[2];
    LocalTensor<U> singleScaleUUBTensor[2];
    LocalTensor<T> singleScaleUBTensor[2];
    LocalTensor<U> scaleUUBTensor[2];
    LocalTensor<T> scaleUBTensor[2];
    LocalTensor<T> workUBTensor[2];
    LocalTensor<T> outputUBTensor[2];

    GlobalTensor<V> outputGt;
    GlobalTensor<U> inputGt[8];
    GlobalTensor<U> inputScaleGt[8];
    GlobalTensor<U> outScaleGt;
};

#endif  // LCCL_DATACOPY_GM2GM_DELAY_H
