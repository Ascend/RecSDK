/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef MXREC_HSTU_COMMON_CONST_H
#define MXREC_HSTU_COMMON_CONST_H

#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

namespace HstuDenseBackward {
enum class MaskType {
    MASK_TRIL = 0,
    MASK_TRIU = 1,
    MASK_NONE = 2,
    MASK_CUSTOM = 3
};
constexpr uint32_t MAX_BATCH_SIZE = 2048;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;
constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int INVALID_TASK_ID = -1;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int ALIGN_16 = 16;
constexpr int UB_SIZE = 170 * 1024;

constexpr int MID_USE_TIMES = 2;
constexpr int USE_BUFFER_NUM = 2;
constexpr int TWO = 2;
constexpr int AIV_NUM_IN_ONE_CORE = 2;

using TNDShape = Shape<uint32_t, uint32_t, uint32_t>;
using TNDStride = AscendC::Stride<uint32_t, uint32_t, uint32_t>;
using TNDLayout = Layout<TNDShape, TNDStride>;
using PipeBlockShape = Shape<uint32_t, uint32_t, uint32_t>;
using PipeBlockStride = AscendC::Stride<uint32_t, uint32_t, uint32_t>;
using PipeBlockLayout = Layout<PipeBlockShape, PipeBlockStride>;

using BNSSShape = Shape<uint32_t, uint32_t, uint32_t, uint32_t>;
using BNSSStride = AscendC::Stride<uint32_t, uint32_t, uint32_t, uint32_t>;
using BNSSLayout = Layout<BNSSShape, BNSSStride>;

struct AddrArgs {
    GM_ADDR grad;
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR qGrad;
    GM_ADDR kGrad;
    GM_ADDR vGrad;
    GM_ADDR workspace;
};

struct BaseShapeArgs {
    uint32_t totalBatchSize;
    uint32_t batchSize;
    uint32_t headNum;
    uint32_t headDim;
    uint32_t maxSeqLen;
};

__aicore__ inline void DoVWhenMte2Finish(TPipe* pipePtr)
{
    int32_t eventMTE2ToV = static_cast<int32_t>(pipePtr->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
}

__aicore__ inline void DoVWhenMte3Finish(TPipe* pipePtr)
{
    int32_t eventMTE3ToV = static_cast<int32_t>(pipePtr->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV);
}

__aicore__ inline void DoSWhenMte3Finish(TPipe* pipePtr)
{
    int32_t eventMTE3ToS = static_cast<int32_t>(pipePtr->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
}

__aicore__ inline void DoMte2WhenVFinish(TPipe* pipePtr)
{
    int32_t eventVToMTE2 = static_cast<int32_t>(pipePtr->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventVToMTE2);
}

__aicore__ inline void DoMte3WhenVFinish(TPipe* pipePtr)
{
    int32_t eventVToMTE3 = static_cast<int32_t>(pipePtr->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
}
}  // namespace HstuDenseBackward

#endif  // MXREC_HSTU_COMMON_CONST_H
