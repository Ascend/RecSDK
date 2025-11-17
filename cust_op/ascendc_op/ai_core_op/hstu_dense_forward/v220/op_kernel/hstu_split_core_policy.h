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

class SeqTask {
public:
    __aicore__ inline SeqTask(int64_t seqLenQ, int64_t seqLenK, int64_t numCtx, int64_t numTg, int64_t blockLen)
    {
        this->seqLenQ = seqLenQ;
        this->seqLenK = seqLenK;
        this->numBlk_ = CeilDiv(seqLenQ, blockLen);
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
                                            GlobalTensor<oType>& seqOffsetsQGt,
                                            GlobalTensor<oType>& seqOffsetsKGt,
                                            GlobalTensor<oType>& numContextGt,
                                            GlobalTensor<oType>& numTargetGt,
                                            int64_t blockLen)
    {
        int64_t seqLenQ = seqOffsetsQGt.GetValue(seqId + 1) - seqOffsetsQGt.GetValue(seqId);
        int64_t seqLenK = seqOffsetsKGt.GetValue(seqId + 1) - seqOffsetsKGt.GetValue(seqId);
        int64_t numCtx = numContextGt.GetValue(seqId);
        int64_t numTg = numTargetGt.GetValue(seqId);

        return SeqTask(seqLenQ, seqLenK, numCtx, numTg, blockLen);
    }

private:
    int64_t seqLenQ;
    int64_t seqLenK;
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
            uint32_t taskNum = CeilDiv(seqLenK, blockLen_);
            return taskNum;
        }

        uint32_t taskNum = CeilDiv(seqLenK - seqLenQ, blockLen_);
        if (numCtx_ > 0) {
            // context mask特殊处理
            taskNum = CeilDiv(seqLenQ - numTg_, blockLen_);
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
            this->taskNum_ = CeilDiv(seqLenK - seqLenQ, blockLen_) + 1;
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
                                      GlobalTensor<oType>& seqOffsetsQGt,
                                      GlobalTensor<oType>& seqOffsetsKGt,
                                      GlobalTensor<oType>& numContextGt,
                                      GlobalTensor<oType>& numTargetGt,
                                      bool fastMode)
    {
        this->maskType = maskType;
        this->coreNum = coreNum;
        this->blockLen = blockLen;
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->tgsize = tgsize;
        this->seqOffsetsQGt = seqOffsetsQGt;
        this->seqOffsetsKGt = seqOffsetsKGt;
        this->numContextGt = numContextGt;
        this->numTargetGt = numTargetGt;

        this->bxn = batchSize * headNum;
        this->fastMode = fastMode;
    }
    
    __aicore__ inline void SplitCoreFast(int (&result)[2], int coreId)
    {
        uint32_t totalTaskNum = this->bxn;
        uint32_t usedCoreNum = (this->coreNum > totalTaskNum) ? totalTaskNum : this->coreNum;

        uint32_t splitNextCoreProcNum = totalTaskNum / usedCoreNum;
        uint32_t splitPrevCoreProcNum = splitNextCoreProcNum + 1;
        uint32_t splitCoreIdx = totalTaskNum % usedCoreNum;
        if (coreId < splitCoreIdx) {
            result[0] = coreId * splitPrevCoreProcNum;
            result[1] = result[0] + splitPrevCoreProcNum;
        } else if (coreId < usedCoreNum) {
            result[0] = splitCoreIdx * splitPrevCoreProcNum + (coreId - splitCoreIdx) * splitNextCoreProcNum;
            result[1] = result[0] + splitNextCoreProcNum;
        } else {
            result[0] = 0;
            result[1] = 0;
        }
    }

    __aicore__ inline void SplitCoreSlow(int (&result)[2], int coreId)
    {
         // 计算总任务量
        uint32_t totalTaskNum = 0;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            int64_t seqlenQ = seqOffsetsQGt.GetValue(batchId + 1) - seqOffsetsQGt.GetValue(batchId);
            int64_t numBlkQ = CeilDiv(seqlenQ, blockLen);
            int64_t seqlenK = seqOffsetsKGt.GetValue(batchId + 1) - seqOffsetsKGt.GetValue(batchId);
            int64_t numBlkK = CeilDiv(seqlenK, blockLen);
            int64_t numCtx = numContextGt.GetValue(batchId);
            int64_t numTg = numTargetGt.GetValue(batchId);
            int64_t DeltaQK = seqlenK - seqlenQ;
            bool isDeltaQK = DeltaQK % this->blockLen != 0;

            uint32_t seqTaskNum = ComputeSeqTaskNum(isDeltaQK, seqlenQ, numBlkQ, numBlkK, numCtx, numTg);
            totalTaskNum += headNum * seqTaskNum;

            for (auto headId = 0; headId < headNum; headId++) {
                blockNumber_[batchId * headNum + headId] = numBlkQ;
            }
        }
        int64_t eachCoreTaskNumLimit = CeilDiv(totalTaskNum, this->coreNum);

        // 为指定核心分配任务
        uint32_t batchId = 0;
        uint32_t processedBlocks = 0;
        
        SeqTask seqTask = SeqTask::Create(batchId / headNum, seqOffsetsQGt, seqOffsetsKGt, numContextGt,
                                          numTargetGt, blockLen);
        for (int core = 0; core < coreId && batchId < bxn; core++) {
            processedBlocks += AssignQBlocksToCore(batchId, seqTask, eachCoreTaskNumLimit);
        }
        
        // 记录当前core的Q block范围
        result[0] = processedBlocks;
        
        // 计算当前core的Q block范围
        processedBlocks += AssignQBlocksToCore(batchId, seqTask, eachCoreTaskNumLimit);
        
        result[1] = processedBlocks;
    }

    __aicore__ inline void Compute(int (&result)[2], int coreId)
    {
        if (this->fastMode) {
            SplitCoreFast(result, coreId);
        } else {
            SplitCoreSlow(result, coreId);
        }
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
                seqTask = SeqTask::Create(batchId / headNum, seqOffsetsQGt, seqOffsetsKGt, numContextGt,
                                          numTargetGt, blockLen);
            }
        }
        return assignedQBlocks;
    }

    __aicore__ inline uint32_t ComputeSeqTaskNum(bool isDeltaQK, int64_t seqlenQ, int64_t numBlkQ, int64_t numBlkK,
                                                 int64_t numCtx, int64_t numTg)
    {
        uint32_t seqTaskNum;
        if (maskType == CausalMaskT::MASK_TRIL) {
            if (isDeltaQK) {
                seqTaskNum = numBlkQ * (2 * numBlkK - numBlkQ + 1) / 2 + numBlkQ - 1;
            } else {
                seqTaskNum = numBlkQ * (2 * numBlkK - numBlkQ + 1) / 2;
            }
            if (numCtx > 0) {
                seqTaskNum += CeilDiv(seqlenQ - numTg, blockLen) - 1;  // -1避免重复计算
            }
        } else {
            seqTaskNum = numBlkQ * numBlkK;
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
    bool fastMode;

    GlobalTensor<oType> seqOffsetsQGt;
    GlobalTensor<oType> seqOffsetsKGt;
    GlobalTensor<oType> numContextGt;
    GlobalTensor<oType> numTargetGt;
    uint8_t blockNumber_[MAX_BATCH_SIZE * MAX_HEAD_NUM] = {0};
};
}  // namespace HstuDenseForward
#endif