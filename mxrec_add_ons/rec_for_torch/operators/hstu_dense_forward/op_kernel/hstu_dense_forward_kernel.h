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

#ifndef HSTU_DENSE_FORWARD_KERNEL_FUN_H
#define HSTU_DENSE_FORWARD_KERNEL_FUN_H
#ifdef SUPPORT_V200
    #include "hstu_dense_forward_kernel_patten_bsnd_v200.h"
#else
    #include "hstu_dense_forward_kernel_patten_bsnd.h"
#endif
#include "hstu_dense_causal_mask.h"
namespace HstuDenseForward {

struct QkMatmulArgs {
    int64_t taskId = INVALID_TASK_ID;
    int64_t qkBlockId;
    int64_t batchId;
    int64_t headId;
    int64_t qSeqId;
    int64_t kSeqId;
};

struct ScoreVectorArgs {
    int64_t taskId = INVALID_TASK_ID;
    int64_t scoreBlockId;
    int64_t batchId;
    int64_t headId;
    int64_t qSeqId;
    int64_t kSeqId;
};

struct SvMatmulArgs {
    int64_t transTaskId = INVALID_TASK_ID;
    int64_t taskId = INVALID_TASK_ID;
    int64_t scoreBlockId;
    int64_t batchId;
    int64_t headId;
    int64_t qSeqId;
    int64_t kSeqId;
    int64_t vSeqId;
};

struct SVTransArgs {
    int64_t transTaskId = INVALID_TASK_ID;
    int64_t scoreBlockId;
    int64_t batchId;
    int64_t headId;
    int64_t qSeqId;
};

template <typename qType>
class HstuDenseForwardKernel : public HstuDenseForwardKernelPattenBsnd<qType> {
public:
    __aicore__ inline HstuDenseForwardKernel() {}

    __aicore__ inline void PreInit(const HstuDenseForwardTilingData* __restrict tilingDataPtr)
    {
        seqBlockNumQk = DivCeil(this->xDim1, this->blockHeight); // 不满足一个block的按照一个block进行计算
        qkTotalBlock = this->xDim0 * this->xDim2 * seqBlockNumQk;
    }

    __aicore__ inline void VecScore(ScoreVectorArgs& scoreArgs)
    {
        if (scoreArgs.taskId == INVALID_TASK_ID) {
            return;
        }
        
        int64_t attnBiasOffset = scoreArgs.batchId * this->xDim2 * this->xDim1 * this->xDim1 + \
                                scoreArgs.headId * this->xDim1 * this->xDim1 + \
                                scoreArgs.qSeqId * this->blockHeight * this->xDim1 + \
                                scoreArgs.kSeqId * this->blockHeight;

        int64_t maskOffset = attnBiasOffset;
#ifdef SUPPORT_V200
        maskOffset = scoreArgs.batchId * this->xDim1 * this->xDim1 + \
                     scoreArgs.qSeqId * this->blockHeight * this->xDim1 + \
                     scoreArgs.kSeqId * this->blockHeight;
#endif
        uint32_t causalMask = ((scoreArgs.qSeqId == scoreArgs.kSeqId) &&
            (this->maskType == CausalMaskT::MASK_TRIL)) ? 1 : 0;

        int64_t m = (scoreArgs.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - scoreArgs.qSeqId * this->blockHeight);
        int64_t n = (scoreArgs.kSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - scoreArgs.kSeqId * this->blockHeight);

        this->template VecScoreImpl<uint32_t>(scoreArgs.taskId, attnBiasOffset, maskOffset, this->siluScale,
                                              causalMask, m, n);
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

        int64_t m = (qkPosArgs.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - qkPosArgs.qSeqId * this->blockHeight);
        int64_t n = (qkPosArgs.kSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - qkPosArgs.kSeqId * this->blockHeight);

        this->DoQkMatmulImpl(qOffset, kOffset, qkPosArgs.taskId, m, n, this->xDim3);
    }

    __aicore__ inline void DoSvMatmul(SvMatmulArgs& svArgs)
    {
        if (svArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t vOffset = svArgs.batchId * this->xDim1 * this->xDim2 * this->xDim3 +
                          svArgs.vSeqId * this->blockHeight * this->xDim2 * this->xDim3 +
                          svArgs.headId * this->xDim3;

        int64_t m = (svArgs.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - svArgs.qSeqId * this->blockHeight);
        int64_t n = (svArgs.vSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - svArgs.vSeqId * this->blockHeight);

        if (svArgs.vSeqId == 0) {
            // Override
            this->DoSvMatmulImpl(vOffset, svArgs.taskId, svArgs.transTaskId, 0, m, this->xDim3, n);
        } else {
            // Automic Add
            this->DoSvMatmulImpl(vOffset, svArgs.taskId, svArgs.transTaskId, 1, m, this->xDim3, n);
        }
    }

    __aicore__ inline void DoTransSv(SVTransArgs& args)
    {
        if (args.transTaskId == INVALID_TASK_ID) {
            return;
        }
        
        int64_t outStartOffset = args.batchId * this->xDim1 * this->xDim2 * this->xDim3 + \
                                args.qSeqId * this->blockHeight * this->xDim2 * this->xDim3 + \
                                args.headId * this->xDim3;

        int64_t m = (args.qSeqId != (seqBlockNumQk - 1)) ? this->blockHeight :
                                                                (this->xDim1 - args.qSeqId * this->blockHeight);

        this->DoTransSvImpl(args.transTaskId, outStartOffset, m);
    }

    __aicore__ inline void Compute(const HstuDenseForwardTilingData *__restrict tilingDataPtr)
    {
        PreInit(tilingDataPtr);
        int64_t taskId = 0;
        int64_t transTaskId = 0;

        int64_t cubeCoreLen = this->qkTotalBlock / GetBlockNum();
        int64_t cubeCoreSplitId = this->qkTotalBlock % GetBlockNum();

        int64_t blockNumOfOneBatch = this->xDim2 * this->seqBlockNumQk;
        int64_t blockNumOfOneHead = this->seqBlockNumQk;

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
            for (int64_t kSeqId = 0; kSeqId < this->seqBlockNumQk; kSeqId++) {
                if ((this->maskType == CausalMaskT::MASK_TRIL) and (kSeqId > qSeqId)) {
                    continue;
                }

                int qkBlockId = qBlockId * this->seqBlockNumQk + kSeqId;
                QkMatmulArgs qkArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                this->DoQkMatmul(qkArgs);
                if (taskId > 1) {
                    this->DoSvMatmul(lastLastSvMatmulArgs);
                }
                ScoreVectorArgs scoreArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                if (taskId > 0) {
                    this->VecScore(lastVectorScore);
                }
                lastVectorScore = scoreArgs;

                int64_t vSeqId = kSeqId;
                SvMatmulArgs svMatmulArgs = {transTaskId, taskId, qkBlockId, batchId, headId, qSeqId, kSeqId, vSeqId};
                lastLastSvMatmulArgs = lastSvMatmulArgs;
                lastSvMatmulArgs = svMatmulArgs;

                this->WaitQkMatmul();
                if (taskId > 1) {
                    this->WaitSvMatmul();
                }

                taskId += 1;
            }

            SVTransArgs svTransArgs = {transTaskId, qBlockId * this->seqBlockNumQk, batchId, headId, qSeqId};
            if (transTaskId > 1) {
                this->DoTransSv(lastLastSvTrans);
            }
            lastLastSvTrans = lastSvTrans;
            lastSvTrans = svTransArgs;
            transTaskId += 1;
        }
        if (taskId == 0) {
            return;
        }

        if (taskId == 1) {
            this->VecScore(lastVectorScore);
            pipe_barrier(PIPE_ALL);
            this->DoSvMatmul(lastSvMatmulArgs);
            this->WaitSvMatmul();
            this->DoTransSv(lastSvTrans);
            return;
        }

        this->DoSvMatmul(lastLastSvMatmulArgs);
        this->VecScore(lastVectorScore);
        pipe_barrier(PIPE_ALL);
        this->DoSvMatmul(lastSvMatmulArgs);
        this->WaitSvMatmul();
        this->DoTransSv(lastLastSvTrans);
        this->WaitSvMatmul();
        this->DoTransSv(lastSvTrans);
    }

private:
    int64_t seqBlockNumQk;
    int64_t qkTotalBlock;
};

}

#endif