/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef HSTU_SPLIT_CORE_POLICY_H
#define HSTU_SPLIT_CORE_POLICY_H

#include <unistd.h>
#include <cstdint>

#include "kernel_log.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hstu_common_const.h"

using namespace AscendC;

namespace HstuDenseForward {
template <typename T>
__aicore__ inline T CeilDiv(T dividend, T divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

class SeqTask {
public:
    __aicore__ inline SeqTask(int64_t seqLen, int64_t numCtx, int64_t numTg, int64_t blockLen)
    {
        this->seqLen_ = seqLen;
        this->numBlk_ = CeilDiv(seqLen, blockLen);
        this->numCtx_ = numCtx;
        this->numTg_ = numTg;
        this->blockLen_ = blockLen;
    }

    __aicore__ inline uint32_t Get(CausalMaskT maskType)
    {
        this->taskNum_ = (blockQCount_ == INIT_BLOCK_Q_COUNT) ? Get1st(maskType) : GetNext(maskType);
        blockQCount_++;
        return this->taskNum_;
    }

    template <typename oType>
    __aicore__ inline static SeqTask Create(int64_t seqId,
                                            GlobalTensor<oType>& seqOffsetsGt,
                                            GlobalTensor<oType>& numContextGt,
                                            GlobalTensor<oType>& numTargetGt,
                                            int64_t blockLen)
    {
        int64_t seqLen = seqOffsetsGt.GetValue(seqId + 1) - seqOffsetsGt.GetValue(seqId);
        int64_t numCtx = numContextGt.GetValue(seqId);
        int64_t numTg = numTargetGt.GetValue(seqId);

        return SeqTask(seqLen, numCtx, numTg, blockLen);
    }

private:
    int64_t seqLen_;
    int64_t numBlk_;
    int64_t numCtx_;
    int64_t numTg_;
    int64_t blockLen_;

    constexpr static uint32_t INIT_BLOCK_Q_COUNT = 0;
    uint32_t taskNum_ = INIT_BLOCK_Q_COUNT;
    uint32_t blockQCount_ = INIT_BLOCK_Q_COUNT;

    __aicore__ inline uint32_t Get1st(CausalMaskT maskType)
    {
        if (maskType != CausalMaskT::MASK_TRIL) {
            uint32_t taskNum = CeilDiv(seqLen_, blockLen_);
            return taskNum;
        }

        uint32_t taskNum = 1;
        if (numCtx_ > 0) {
            // context mask特殊处理
            taskNum = CeilDiv(seqLen_ - numTg_, blockLen_);
        }
        return taskNum;
    }

    __aicore__ inline uint32_t GetNext(CausalMaskT maskType)
    {
        if (maskType != CausalMaskT::MASK_TRIL) {
            return this->taskNum_;
        }
        if (blockQCount_ == INIT_BLOCK_Q_COUNT + 1) {
            // 去除context mask的特殊处理，回到causal mask的正常逻辑
            this->taskNum_ = 1;
        }
        this->taskNum_++;
        return this->taskNum_;
    }
};

template <typename oType>
class BlockTaskAssign {
public:
    __aicore__ inline BlockTaskAssign(CausalMaskT maskType,
                                      uint32_t coreNum,
                                      int64_t blockLen,
                                      int64_t batchSize,
                                      int64_t headNum,
                                      int64_t tgsize,
                                      GlobalTensor<oType>& seqOffsetsGt,
                                      GlobalTensor<oType>& numContextGt,
                                      GlobalTensor<oType>& numTargetGt)
    {
        this->maskType = maskType;
        this->coreNum = coreNum;
        this->blockLen = blockLen;
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->tgsize = tgsize;
        this->seqOffsetsGt = seqOffsetsGt;
        this->numContextGt = numContextGt;
        this->numTargetGt = numTargetGt;

        this->bxn = batchSize * headNum;
    }

    __aicore__ inline void Compute(int (&result)[2], int coreId)
    {
        // 计算总任务量
        uint32_t totalTaskNum = 0;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            int64_t seqlen = seqOffsetsGt.GetValue(batchId + 1) - seqOffsetsGt.GetValue(batchId);
            int64_t numBlk = CeilDiv(seqlen, blockLen);
            int64_t numCtx = numContextGt.GetValue(batchId);
            int64_t numTg = numTargetGt.GetValue(batchId);

            uint32_t seqTaskNum = ComputeSeqTaskNum(seqlen, numBlk, numCtx, numTg);
            totalTaskNum += headNum * seqTaskNum;

            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber_[batchId * headNum + headId] = numBlk;
            }
        }
        int64_t eachCoreTaskNumLimit = CeilDiv(totalTaskNum, this->coreNum);

        // 为指定核心分配任务
        uint32_t batchId = 0;
        uint32_t processedBlocks = 0;
        
        SeqTask seqTask = SeqTask::Create(batchId / headNum, seqOffsetsGt, numContextGt, numTargetGt, blockLen);
        for (int core = 0; core < coreId && batchId < bxn; core++) {
            processedBlocks += AssignQBlocksToCore(batchId, seqTask, eachCoreTaskNumLimit);
        }
        
        // 记录当前core的Q block范围
        result[0] = processedBlocks;
        
        // 计算当前core的Q block范围
        processedBlocks += AssignQBlocksToCore(batchId, seqTask, eachCoreTaskNumLimit);
        
        result[1] = processedBlocks;
    }

private:
    __aicore__ inline uint32_t AssignQBlocksToCore(uint32_t& batchId, SeqTask& seqTask, uint32_t taskLimit)
    {
        // 如果没有更多Q block可分配，直接返回0
        if (blockNumber_[batchId] == 0) {
            return 0;
        }
        
        uint32_t assignedQBlocks = 0;
        uint32_t workload = 0;
        
        while (workload < taskLimit && batchId < bxn) {
            workload += seqTask.Get(maskType);
            assignedQBlocks++;
            blockNumber_[batchId]--;
            
            if (blockNumber_[batchId] == 0) {
                // 移动到下一个batch
                batchId++;
                seqTask = SeqTask::Create(batchId / headNum, seqOffsetsGt, numContextGt, numTargetGt, blockLen);
            }
        }
        return assignedQBlocks;
    }

    __aicore__ inline uint32_t ComputeSeqTaskNum(int64_t seqLen, int64_t numBlk, int64_t numCtx, int64_t numTg)
    {
        uint32_t seqTaskNum;
        if (maskType == CausalMaskT::MASK_TRIL) {
            seqTaskNum = numBlk * (numBlk + 1) / 2;
            if (numCtx > 0) {
                seqTaskNum += CeilDiv(seqLen - numTg, blockLen) - 1;  // -1避免重复计算
            }
        } else {
            seqTaskNum = numBlk * numBlk;
        }
        return seqTaskNum;
    }

    CausalMaskT maskType;
    uint32_t coreNum;
    int64_t blockLen;
    int64_t batchSize;
    int64_t headNum;
    int64_t tgsize;
    uint32_t bxn;  // 总序列数

    GlobalTensor<oType> seqOffsetsGt;
    GlobalTensor<oType> numContextGt;
    GlobalTensor<oType> numTargetGt;
    uint8_t blockNumber_[MAX_BATCH_SIZE * MAX_HEAD_NUM] = {0};
};
}  // namespace HstuDenseForward
#endif