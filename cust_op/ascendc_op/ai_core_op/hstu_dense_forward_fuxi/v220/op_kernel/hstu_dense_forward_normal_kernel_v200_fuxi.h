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

#ifndef HSTU_DENSE_FORWARD_KERNEL_V200_FUXI_H
#define HSTU_DENSE_FORWARD_KERNEL_V200_FUXI_H
#include "hstu_dense_kernel_patten_bsnd_v200_fuxi.h"

namespace HstuDenseForwardFuxi {

struct TpMaskArgs {
    int64_t taskId = INVALID_TASK_ID;
    int64_t batchId;
    int64_t qSeqId;
    int64_t kSeqId;
};

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

struct BiasMatmulArgs {
    int64_t taskId = INVALID_TASK_ID;
    int64_t batchId;
    int64_t headId;
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
class HstuDenseForwardNormalKernelv200Fuxi : public HstuDenseKernelPattenBsndV200Fuxi<qType> {
public:
    __aicore__ inline HstuDenseForwardNormalKernelv200Fuxi() {}

    __aicore__ inline void PreInit(const HstuDenseForwardFuxiTilingData *__restrict tilingDataPtr,
        int64_t& lenOfThisCore, int64_t& offsetOfThisCore)
    {
        seqBlockNumQk = this->seqLen / this->blockHeight;
        qkTotalBlock = this->batchSize * this->numHead * seqBlockNumQk;

        int64_t cubeCoreLen = qkTotalBlock / GetBlockNum();
        int64_t cubeCoreSplitId = qkTotalBlock % GetBlockNum();
        if (GetBlockIdx() / SPLIT_CORE >= cubeCoreSplitId) {
            lenOfThisCore = cubeCoreLen;
            offsetOfThisCore =
                cubeCoreSplitId * (cubeCoreLen + 1) + (GetBlockIdx() / SPLIT_CORE - cubeCoreSplitId) * cubeCoreLen;
            return;
        }

        lenOfThisCore = cubeCoreLen + 1;
        offsetOfThisCore = GetBlockIdx() / SPLIT_CORE * (cubeCoreLen + 1);
    }

    __aicore__ inline void DoQkMatmul(QkMatmulArgs& qkPosArgs)
    {
        if (qkPosArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t qOffset = qkPosArgs.batchId * this->seqLen * this->numHead * this->dim + \
                          qkPosArgs.qSeqId * this->blockHeight * this->numHead * this->dim + \
                          qkPosArgs.headId * this->dim;
        int64_t kOffset = qkPosArgs.batchId * this->seqLen * this->numHead * this->dim + \
                          qkPosArgs.kSeqId * this->blockHeight * this->numHead * this->dim + \
                          qkPosArgs.headId * this->dim;

        if (qkPosArgs.kSeqId == 0) {
            this->DoCopyQImpl(qOffset);
        }

        this->DoQkMatmulImpl(qOffset, kOffset);
    }

    __aicore__ inline void DoSvMatmul(SvMatmulArgs& svArgs)
    {
        if (svArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t vOffset = svArgs.batchId * this->seqLen * this->numHead * this->dim +
                          svArgs.vSeqId * this->blockHeight * this->numHead * this->dim +
                          svArgs.headId * this->dim;

        uint8_t enAtomicAdd = (svArgs.vSeqId == 0) ? 0 : 1;

        this->DoSvMatmulImpl(vOffset, enAtomicAdd);
    }

    __aicore__ inline void DoTvMatmul(BiasMatmulArgs& biasArgs)
    {
        if (biasArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t vOffset = biasArgs.batchId * this->seqLen * this->numHead * this->dim +
                          biasArgs.vSeqId * this->blockHeight * this->numHead * this->dim +
                          biasArgs.headId * this->dim;

        uint8_t enAtomicAdd = (biasArgs.vSeqId == 0) ? 0 : 1;

        this->DoTvMatmulImpl(vOffset, enAtomicAdd);
    }

    __aicore__ inline void DoPvMatmul(BiasMatmulArgs& biasArgs)
    {
        if (biasArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t vOffset = biasArgs.batchId * this->seqLen * this->numHead * this->dim +
                          biasArgs.vSeqId * this->blockHeight * this->numHead * this->dim +
                          biasArgs.headId * this->dim;

        uint8_t enAtomicAdd = (biasArgs.vSeqId == 0) ? 0 : 1;

        this->DoPvMatmulImpl(vOffset, enAtomicAdd);
    }

    __aicore__ inline void DoBiasMask(TpMaskArgs& tpMaskArgs)
    {
        if (tpMaskArgs.taskId == INVALID_TASK_ID) {
            return;
        }

        int64_t maskOffset = tpMaskArgs.batchId * this->seqLen * this->seqLen + \
            tpMaskArgs.qSeqId * this->blockHeight * this->seqLen + \
            tpMaskArgs.kSeqId * this->blockHeight;

        int64_t timestampOffset = maskOffset;
        int64_t positionOffset = maskOffset;

        this->DoBiasMaskImpl(maskOffset, timestampOffset, positionOffset);
    }

    __aicore__ inline void VecScore(ScoreVectorArgs& scoreArgs)
    {
        if (scoreArgs.taskId == INVALID_TASK_ID) {
            return;
        }
        
        int64_t maskOffset = scoreArgs.batchId * this->seqLen * this->seqLen + \
            scoreArgs.qSeqId * this->blockHeight * this->seqLen + \
            scoreArgs.kSeqId * this->blockHeight;

        this->VecScoreImpl(maskOffset, this->siluScale);
    }

    __aicore__ inline void DoTransResult(SVTransArgs& args)
    {
        if (args.transTaskId == INVALID_TASK_ID) {
            return;
        }

        int64_t seqOffsetLen = this->enableBias ? OUTPUT_DIM3_TIMES_3 : OUTPUT_DIM3_TIMES_1;
        
        int64_t outStartOffset = args.batchId * seqOffsetLen * this->seqLen * this->numHead * this->dim + \
                                args.qSeqId * seqOffsetLen * this->blockHeight * this->numHead * this->dim + \
                                args.headId * this->dim;

        this->DoTransResultImpl(outStartOffset);
    }

    __aicore__ inline void Compute(const HstuDenseForwardFuxiTilingData *__restrict tilingDataPtr)
    {
        int64_t lenOfThisCore;
        int64_t offsetOfThisCore;
        PreInit(tilingDataPtr, lenOfThisCore, offsetOfThisCore);

        int64_t taskId = 0;
        int64_t transTaskId = 0;

        int64_t blockNumOfOneBatch = this->numHead * seqBlockNumQk;
        int64_t blockNumOfOneHead = seqBlockNumQk;

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

                if (this->enableBias) {
                    TpMaskArgs tpMaskArgs = {taskId, batchId, qSeqId, kSeqId};
                    this->DoBiasMask(tpMaskArgs);
                }

                ScoreVectorArgs scoreArgs = {taskId, qkBlockId, batchId, headId, qSeqId, kSeqId};
                this->VecScore(scoreArgs);
                int64_t vSeqId = kSeqId;

                if (this->enableBias) {
                    BiasMatmulArgs biasArgs = {taskId, batchId, headId, vSeqId};
                    this->DoTvMatmul(biasArgs);
                    this->DoPvMatmul(biasArgs);
                }

                SvMatmulArgs svMatmulArgs = {transTaskId, taskId, qkBlockId, batchId, headId, qSeqId, kSeqId, vSeqId};
                this->DoSvMatmul(svMatmulArgs);
                taskId += 1;
            }
            this->DoFreeQImpl();
            SVTransArgs svTransArgs = {transTaskId, qBlockId * seqBlockNumQk, batchId, headId, qSeqId};
            this->DoTransResult(svTransArgs);
            transTaskId += 1;
        }
    }

private:
    int64_t seqBlockNumQk;
    int64_t qkTotalBlock;
};

}
#endif