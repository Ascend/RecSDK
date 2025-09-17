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

template <typename oType>
class BlockTaskAssign {
public:
    __aicore__ inline BlockTaskAssign(CausalMaskT maskType,
                                      uint32_t coreNum,
                                      int64_t blockLen,
                                      int64_t batchSize,
                                      int64_t headNum,
                                      GlobalTensor<oType>& seqOffsetsGt,
                                      GlobalTensor<int64_t>& blockNumberGt)
    {
        this->maskType = maskType;
        this->coreNum = coreNum;
        this->blockLen = blockLen;
        this->batchSize = batchSize;
        this->headNum = headNum;
        this->seqOffsetsGt = seqOffsetsGt;

        this->blockNumberGt = blockNumberGt;
        this->bxn = batchSize * headNum;
    }

    __aicore__ inline void Compute()
    {
        // 计算总任务量
        uint32_t totalTaskNum = 0;
        for (auto batchId = 0; batchId < batchSize; batchId++) {
            int64_t seqlen = seqOffsetsGt.GetValue(batchId + 1) - seqOffsetsGt.GetValue(batchId);
            int64_t numBlk = CeilDiv(seqlen, blockLen);

            uint32_t seqTaskNum = ComputeSeqTaskNum(seqlen, numBlk);
            totalTaskNum += headNum * seqTaskNum;

            // blockNumberGt 前batchSize x headNum (bxn)个位置记录每个batch每个head的block数
            for (auto headId = 0; headId < headNum; headId++) {
                blockNumberGt.SetValue(batchId * headNum + headId, numBlk);
            }
        }
        int64_t eachCoreTaskNumLimit = CeilDiv(totalTaskNum, this->coreNum);

        // 遍历workers 计算得到每一个works的任务量
        bool initFlag = false;
        uint32_t batchId = 0;
        uint32_t taskNum = Get1stTaskNum(batchId, initFlag);
        uint32_t processBlockNum = 0;  // 记录Qblock数
        for (uint32_t i = 0; i < this->coreNum && batchId < bxn; i++) {
            blockNumberGt.SetValue(bxn + i, processBlockNum);  // corei的startblockid
            uint32_t workLoads = 0;
            while (workLoads < eachCoreTaskNumLimit && batchId < bxn) {
                // 更新状态
                workLoads += taskNum;
                processBlockNum++;
                // 更新任务数
                UpdateTaskNum(taskNum, initFlag);
                // 更新待分配的Qblock数
                blockNumberGt.SetValue(batchId, blockNumberGt.GetValue(batchId) - 1);
                if (blockNumberGt.GetValue(batchId) == 0) {
                    batchId++;
                    taskNum = Get1stTaskNum(batchId, initFlag);
                }
            }
            blockNumberGt.SetValue(bxn + i + this->coreNum, processBlockNum);  // corei的endblockid
        }
        DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(blockNumberGt);
    }

private:
    __aicore__ inline uint32_t ComputeSeqTaskNum(int64_t seqlen, int64_t numBlk)
    {
        uint32_t seqTaskNum;
        if (maskType == CausalMaskT::MASK_TRIL) {
            seqTaskNum = numBlk * (numBlk + 1) / 2;
        } else {
            seqTaskNum = numBlk * numBlk;
        }
        return seqTaskNum;
    }

    __aicore__ inline uint32_t Get1stTaskNum(int64_t batchId, bool& initFlag)
    {
        initFlag = true;
        if (batchId >= bxn) {
            return 0;
        }
        // 得到batch的第一个Q block对应的计算量
        if (maskType == CausalMaskT::MASK_TRIL) {
            uint32_t taskNum = 1;
            int64_t batch = batchId / headNum;
            return taskNum;
        } else {
            return blockNumberGt.GetValue(batchId);
        }
    }

    __aicore__ inline void UpdateTaskNum(uint32_t& taskNum, bool& initFlag)
    {
        if (maskType != CausalMaskT::MASK_TRIL) {
            return;
        }
        if (initFlag) {
            initFlag = false;
            taskNum = 1;
        }
        taskNum++;
    }

    CausalMaskT maskType;
    uint32_t coreNum;
    int64_t blockLen;
    int64_t batchSize;
    int64_t headNum;
    uint32_t bxn;  // 总序列数

    GlobalTensor<oType> seqOffsetsGt;
    GlobalTensor<int64_t> blockNumberGt;
};
}  // namespace HstuDenseForward
#endif