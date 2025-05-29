/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef HSTU_DENSE_FORWARD_KERNEL_V200_FUN_H
#define HSTU_DENSE_FORWARD_KERNEL_V200_FUN_H
#include "hstu_dense_forward_kernel.h"

namespace HstuDenseForward {

template <typename qType>
class HstuDenseForwardKernelv200 : public HstuDenseForwardKernel<qType> {
public:
    __aicore__ inline HstuDenseForwardKernelv200() {}

    __aicore__ inline void PreInit(const HstuDenseForwardTilingData *__restrict tilingDataPtr)
    {
        seqBlockNumQk = this->xDim1 / this->blockHeight;
        qkTotalBlock = this->xDim0 * this->xDim2 * seqBlockNumQk;
    }

    __aicore__ inline void DoQkMatmul(QkMatmulArgs& qkPosArgs)
    {
        if (qkPosArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t qOffset = qkPosArgs.batchId * this->xDim1 * this->xDim2 * this->xDim3 + \
                          qkPosArgs.qSeqId * this->blockHeight * this->xDim2 * this->xDim3 + \
                          qkPosArgs.headId * this->xDim3;
        int64_t kOffset = qkPosArgs.batchId * this->xDim1 * this->xDim2 * this->xDim3 + \
                          qkPosArgs.kSeqId * this->blockHeight * this->xDim2 * this->xDim3 + \
                          qkPosArgs.headId * this->xDim3;

        if (qkPosArgs.kSeqId == 0) {
            this->DoCopyQImpl(qOffset);
        }

        this->DoQkMatmulImpl(qOffset, kOffset, qkPosArgs.taskId);
    }

    __aicore__ inline void Compute(const HstuDenseForwardTilingData *__restrict tilingDataPtr)
    {
        PreInit(tilingDataPtr);
        int64_t taskId = 0;
        int64_t transTaskId = 0;

        int64_t cubeCoreLen = qkTotalBlock / GetBlockNum();
        int64_t cubeCoreSplitId = qkTotalBlock % GetBlockNum();

        int64_t blockNumOfOneBatch = this->xDim2 * seqBlockNumQk;
        int64_t blockNumOfOneHead = seqBlockNumQk;

        int64_t lenOfThisCore;
        int64_t offsetOfThisCore;

        if (GetBlockIdx() / SPLIT_CORE >= cubeCoreSplitId) {
            lenOfThisCore = cubeCoreLen;
            offsetOfThisCore =
                cubeCoreSplitId * (cubeCoreLen + 1) + (GetBlockIdx() / SPLIT_CORE - cubeCoreSplitId) * cubeCoreLen;
        } else {
            lenOfThisCore = cubeCoreLen + 1;
            offsetOfThisCore = GetBlockIdx() / SPLIT_CORE * (cubeCoreLen + 1);
        }

        SVTransArgs lastSvTrans;
        SVTransArgs lastLastSvTrans;

        ScoreVectorArgs lastVectorScore;
        SvMatmulArgs lastSvMatmulArgs;
        SvMatmulArgs lastLastSvMatmulArgs;
        for (int64_t qBlockId = offsetOfThisCore; qBlockId < offsetOfThisCore + lenOfThisCore; qBlockId++) {
            int64_t batchId = qBlockId / blockNumOfOneBatch;
            int64_t batchRemain = qBlockId % blockNumOfOneBatch;

            int64_t headId = batchRemain / blockNumOfOneHead;
            int64_t headReamin = batchRemain % blockNumOfOneHead;

            int64_t qSeqId = headReamin;

            if ((headId + qSeqId) % SPLIT_CORE != GetBlockIdx() % SPLIT_CORE) {
                continue;
            }
            for (int64_t kSeqId = 0; kSeqId < seqBlockNumQk; kSeqId++) {
                if (kSeqId > qSeqId) {
                    continue;
                }

                int qkBlockId = qBlockId * seqBlockNumQk + kSeqId;
                QkMatmulArgs qkArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                this->DoQkMatmul(qkArgs);
                ScoreVectorArgs scoreArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                this->VecScore(scoreArgs);
                int64_t vSeqId = kSeqId;
                SvMatmulArgs svMatmulArgs = {transTaskId, taskId, qkBlockId, batchId, headId, qSeqId, kSeqId, vSeqId};
                this->DoSvMatmul(svMatmulArgs);
                taskId += 1;
            }
            this->DoFreeQImpl();
            SVTransArgs svTransArgs = {transTaskId, qBlockId * seqBlockNumQk, batchId, headId, qSeqId};
            this->DoTransSv(svTransArgs);
            transTaskId += 1;
        }
    }

private:
    int64_t seqBlockNumQk;
    int64_t qkTotalBlock;
};

}

#endif