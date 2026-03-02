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

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

namespace HstuForward {

enum class CausalMaskT {
    MASK_TRIL = 0,  // 下三角
    MASK_TRIU,      // 上三角
    MASK_NONE,      // 不使能mask
    MASK_CUSTOM,   // 用户自定义mask
};

constexpr uint32_t MAX_BATCH_SIZE = 2048;
constexpr uint32_t MAX_HEAD_NUM = 8;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;

constexpr int INVALID_TASK_ID = -1;
constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_MN = 256 * 256;

constexpr int DEFAULT_SPLIT = 0;
constexpr int FAST_SPLIT = 1;
constexpr int FAST_SPLIT_SINGLE = 2;
constexpr int STREAM_K = 3;

constexpr int NORMAL_TILING_KEY = 0;
constexpr int JAGGED_TILING_KEY = 1;
constexpr int PAGED_TILING_KEY = 2;

constexpr int VEC_PER_PROCESS = 32;
constexpr int UB_SIZE = 170 * 1024;  // 170KB
constexpr int QUEUE_IN_NUM = 2;
constexpr int SPLIT_CORE = 2;
constexpr int ALIGN_16 = 16;
constexpr int ALIGN_32 = 32;

constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;
constexpr int INT_ALIGN_NUM = 8;

constexpr int BLOCK_HEIGHT_256 = 256;
constexpr int BLOCK_HEIGHT_128 = 128;
constexpr int BASIC_K_32 = 32;
constexpr int BASIC_K_64 = 64;
constexpr int MAX_BLOCK_DIM = 512;
constexpr int MATMUL_L1_SIZE = 524288;  // 512KB

constexpr int VECTOR_SCORE_UB_BLOCK_ELEM = (VEC_PER_PROCESS * BLOCK_HEIGHT_256) / USE_QUEUE_NUM;

template <typename T>
__aicore__ inline T CeilDiv(T dividend, T divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename qTypeTemplate, typename oTypeTemplate, bool bias,
    bool determin, CausalMaskT maskedType, int tilingM, int tilingN, int tilingK>
struct TraitParams {
    using qType = qTypeTemplate;
    using oType = oTypeTemplate;
    static constexpr bool enableBias = bias;
    static constexpr bool deterministic = determin;
    static constexpr CausalMaskT maskType = maskedType;
    static constexpr int blockM = tilingM;
    static constexpr int blockN = tilingN;
    static constexpr int blockK = tilingK;
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

template <typename oType>
__aicore__ inline int64_t GetBatchSizeFromJaggedOffset(GlobalTensor<oType>& seqOffsetData, int32_t seqOffsetLens)
{
    if (seqOffsetLens <= 0) {
        return 0;
    }

    // 二分法找出有效batch
    int64_t maxValue = seqOffsetData.GetValue(seqOffsetLens - 1);
    int32_t left = 0;
    int32_t right = seqOffsetLens - 1;
    int32_t firstMaxIdx = seqOffsetLens - 1;
    while (left <= right) {
        int32_t mid = left + (right - left) / 2;  // 二分法除以2找到剩余中间位置
        if (seqOffsetData.GetValue(mid) == maxValue) {
            firstMaxIdx = mid;
            right = mid - 1;
        } else if (seqOffsetData.GetValue(mid) < maxValue) {
            left = mid + 1;
        }
    }

    int64_t batchSize_ = static_cast<int64_t>(firstMaxIdx);
    return batchSize_;
}

struct JaggedTaskArgs {
    uint32_t batchId = 0;           // 该基本块所属的batch
    uint32_t headId = 0;            // 该基本块所属的head
    uint32_t qSeqId = 0;            // 该基本块所属Query 输入的第几个seq block 一个block是256条seq
    uint32_t kSeqId = 0;            // 该基本块所属Key 输入的第几个seq block 一个block是256条seq
    uint32_t actualSeqLen = 0;      // Q序列的基本块实际的序列长度
    uint32_t actualSeqLenK = 0;     // K序列的基本块实际的序列长度
    uint32_t actualHistLen = 0;     // KV序列的历史序列长度
    uint32_t actualNewHistLen = 0;  // Q序列的历史序列长度
    uint32_t qSeqNum = 0;           // 该Batch下Qblock数
    uint32_t kSeqNum = 0;           // 该基本块在K轴需要乘多少次
    uint32_t transTaskId = 0;       // 该基本块转置任务的id
    uint32_t computeASeqLen = 0;    // 该基本块matmul计算左矩阵的序列长度
    uint32_t computeBSeqLen = 0;    // 该基本块matmul计算右矩阵的序列长度
    float scale = 0.0f;             // 该基本块的siluScale
    int64_t numContext = 0;         // 该基本块所属序列的numContext
    int64_t numTarget = 0;          // 该基本块所属序列的numTarget
    int64_t seqGlobalOffset = 0;    // 该基本块的全局序列偏移
    int64_t batchOffset = 0;        // 该基本块的batch偏移
    int64_t batchOffsetK = 0;       // K序列的基本块的batch偏移
    int64_t headSeqLimit = 0;       // 该基本块的head offset最大长度, 超过则需要考虑切换head_id
    int64_t kvOffset = 0;           // 该基本块的key value计算偏移
    int64_t kOffset = 0;            // 该基本块的key计算偏移
    int64_t vOffset = 0;            // 该基本块的value计算偏移
    int64_t ioOffset = 0;           // 该基本块的query attnOutput计算偏移
    int64_t iOffset = 0;            // 该基本块的query计算偏移
    int64_t oOffset = 0;            // 该基本块的attnOutput计算偏移
    int64_t pageNum = 0;            // 该基本块存在kvcache中的page个数
    uint32_t isFirstSeqBlk = 1;     // 该基本块的sv matmul是否需要清空流水对应空间
    uint32_t isStartFromZero = 1;   // 该基本块所在q行在本核心中的是否从0开始
    uint32_t isEndToTail = 1;       // 该基本块所在q行在本核心中的是否到最后结束
};

}  // namespace HstuForward

#endif  // MXREC_HSTU_COMMON_CONST_H

