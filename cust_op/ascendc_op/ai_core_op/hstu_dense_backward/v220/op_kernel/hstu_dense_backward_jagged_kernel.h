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


#ifndef HSTU_DENSE_BACKWARD_JAGGED_KERNEL_H
#define HSTU_DENSE_BACKWARD_JAGGED_KERNEL_H

#include "hstu_dense_backward_kernel.h"
#include "hstu_dense_backward_kernel_common.h"

namespace HstuDenseBackward {

struct JaggedTaskInfo {
    int64_t taskId;        // 基本块任务id，参与临时存储块的偏移计算
    int64_t batchId;       // 基本块batch id
    int64_t headId;        // 基本块head id
    int64_t rowId;         // 基本块在当前qk矩阵中的行id，基本单位为blockHeight
    int64_t colId;         // 基本块在当前qk矩阵中的列id，基本单位为blockHeight
    int64_t accumId;       // 基本块累加id，用来获取q/k/v梯度的累加位置
    int64_t blockLimit;    // 基本块在当前batch_head下的最大block偏移，超过后需要切换block
    int64_t curSeqLen;     // 当前计算块的序列长度
    int64_t qkLeftOffset;  // 基本块qk/gv乘法的左矩阵内存偏移
    int64_t qkRightOffset; // 基本块qk/gv乘法的右矩阵内存偏移
    int64_t kGradLeftOffset; // 基本块q/k梯度计算的左矩阵内存偏移，v的左矩阵在缓存中，单独计算
    int64_t vGradRightOffset; // 基本块q/k/v梯度计算的右矩阵内存偏移
    int64_t qGradRightOffset; // 不带bias场景下基本块q梯度计算的右矩阵内存偏移
    int64_t rowLine;          // 基本块需要计算的行数
    int64_t colLine;          // 基本块需要计算的列数
};

template <typename qType> class HstuDenseBackwardJaggedKernel : public HstuDenseBackwardKernel<qType> {
public:
    __aicore__ inline HstuDenseBackwardJaggedKernel() {}

    __aicore__ inline void Compute(Args &args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        backwardTilingData = &tilingData;
        if (backwardTilingData == nullptr) {
            return;
        }
        REGIST_MATMUL_OBJ(&this->pipe, GetSysWorkSpacePtr(), this->qkMatmul,
                          &tilingData.qkMatmul, this->qGradMatmul,
                          &tilingData.qGradMatmul, this->kGradMatmul,
                          &tilingData.kGradMatmul, this->vGradMatmul,
                          &tilingData.vGradMatmul);
        uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
        this->qkMatmul.SetUserDefInfo(tilingPtr);
        this->qGradMatmul.SetUserDefInfo(tilingPtr);
        this->kGradMatmul.SetUserDefInfo(tilingPtr);
        this->vGradMatmul.SetUserDefInfo(tilingPtr);

        this->Init(args);
        this->PreInit();

        this->ComputeJaggedFirst();
        if (this->enableBias) {
            this->ComputeJaggedSecond();
        } else {
            this->CopyQGradToOutput();
        }
    }

    __aicore__ inline void PreInit()
    {
        startColBlock = backwardTilingData->eachCoreStartColBlockId[GetBlockIdx()];
        endColBlock = backwardTilingData->eachCoreEndColBlockId[GetBlockIdx()];
        startRowBlock = backwardTilingData->eachCoreStartRowBlockId[GetBlockIdx()];
        endRowBlock = backwardTilingData->eachCoreEndRowBlockId[GetBlockIdx()];
    }

    __aicore__ inline void GenerateFirstTask(bool isCol = true)
    {
        int64_t batchId = 0;
        uint32_t curSeqLen = 0;
        uint32_t curBatchStartBlock = 0;
        int64_t startBlock = 0;
        if (isCol) {
            startBlock = startColBlock;
        } else {
            startBlock = startRowBlock;
        }

        while (batchId < MAX_BATCH_SIZE) {
            curSeqLen = backwardTilingData->seqOffset[batchId + 1] - backwardTilingData->seqOffset[batchId];
            auto curBatchBlock = this->headNum * ((curSeqLen + this->blockHeight - 1) / this->blockHeight);
            if (curBatchStartBlock + curBatchBlock > startBlock) {
                break;
            }
            curBatchStartBlock += curBatchBlock;
            batchId++;
        }

        int64_t curBlockIdInBatch = startBlock - curBatchStartBlock;
        auto curHeadBlock = (curSeqLen + this->blockHeight - 1) / this->blockHeight;

        int64_t headId = curBlockIdInBatch / curHeadBlock;

        if (isCol) {
            computeTaskInfo[0].rowId = 0;
            computeTaskInfo[0].colId = curBlockIdInBatch % curHeadBlock;
            computeTaskInfo[0].colLine = curSeqLen - computeTaskInfo[0].colId * this->blockHeight;
            if (computeTaskInfo[0].colLine > this->blockHeight) {
                computeTaskInfo[0].colLine = this->blockHeight;
            }
        } else {
            computeTaskInfo[0].rowId = curBlockIdInBatch % curHeadBlock;
            computeTaskInfo[0].colId = 0;
            computeTaskInfo[0].rowLine = curSeqLen - computeTaskInfo[0].rowId * this->blockHeight;
            if (computeTaskInfo[0].rowLine > this->blockHeight) {
                computeTaskInfo[0].rowLine = this->blockHeight;
            }
        }

        computeTaskInfo[0].taskId = 0;
        computeTaskInfo[0].batchId = batchId;
        computeTaskInfo[0].headId = headId;
        computeTaskInfo[0].accumId = 0;
        computeTaskInfo[0].blockLimit = curHeadBlock;
        computeTaskInfo[0].curSeqLen = curSeqLen;
    }

    __aicore__ inline void UpdateNextBlock(int64_t taskId, bool isCol = true)
    {
        int64_t lastTask = (taskId - 1) % COMPUTE_PIPE_NUM;
        int64_t curTask = taskId % COMPUTE_PIPE_NUM;

        if (isCol) {
            computeTaskInfo[curTask].colId += 1;
            if (computeTaskInfo[curTask].colId == computeTaskInfo[lastTask].blockLimit) {
                computeTaskInfo[curTask].colId = 0;
                computeTaskInfo[curTask].headId += 1;
            }
            computeTaskInfo[curTask].rowId = 0;
        } else {
            computeTaskInfo[curTask].rowId += 1;
            if (computeTaskInfo[curTask].rowId == computeTaskInfo[lastTask].blockLimit) {
                computeTaskInfo[curTask].rowId = 0;
                computeTaskInfo[curTask].headId += 1;
            }
            computeTaskInfo[curTask].colId = 0;
        }

        if (computeTaskInfo[curTask].headId == this->headNum) {
            computeTaskInfo[curTask].headId = 0;
            computeTaskInfo[curTask].batchId += 1;

            uint32_t curSeqLen =
                backwardTilingData->seqOffset[computeTaskInfo[curTask].batchId + 1] -
                backwardTilingData->seqOffset[computeTaskInfo[curTask].batchId];
            auto curHeadBlock = (curSeqLen + this->blockHeight - 1) / this->blockHeight;

            computeTaskInfo[curTask].blockLimit = curHeadBlock;
            computeTaskInfo[curTask].curSeqLen = curSeqLen;
        }

        if (isCol) {
            computeTaskInfo[curTask].colLine =
                computeTaskInfo[curTask].curSeqLen - computeTaskInfo[curTask].colId * this->blockHeight;
            if (computeTaskInfo[curTask].colLine > this->blockHeight) {
                computeTaskInfo[curTask].colLine = this->blockHeight;
            }
        } else {
            computeTaskInfo[curTask].rowLine =
                computeTaskInfo[curTask].curSeqLen - computeTaskInfo[curTask].rowId * this->blockHeight;
            if (computeTaskInfo[curTask].rowLine > this->blockHeight) {
                computeTaskInfo[curTask].rowLine = this->blockHeight;
            }
        }

        computeTaskInfo[curTask].accumId += 1;
    }

    __aicore__ inline void CalcBaseOffsetsJagged(int64_t taskId, bool isCol = true)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        computeTaskInfo[curTaskId].qkLeftOffset =
            backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].headId * this->headDim;

        computeTaskInfo[curTaskId].qkRightOffset =
            backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].headId * this->headDim;

        if (this->enableBias) {
            computeTaskInfo[curTaskId].kGradLeftOffset =
                computeTaskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
                computeTaskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
                computeTaskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
                computeTaskInfo[curTaskId].colId * this->blockHeight;
        } else {
            computeTaskInfo[curTaskId].kGradLeftOffset =
                (taskId % MID_USE_TIMES) * this->blockHeight * this->blockHeight;
        }

        if (isCol) {
            computeTaskInfo[curTaskId].vGradRightOffset =
                backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].headId * this->headDim;

            if (!this->enableBias) {
                computeTaskInfo[curTaskId].qGradRightOffset =
                    backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
                    computeTaskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
                    computeTaskInfo[curTaskId].headId * this->headDim;
            }

            computeTaskInfo[curTaskId].rowLine =
                computeTaskInfo[curTaskId].curSeqLen - computeTaskInfo[curTaskId].rowId * this->blockHeight;
            if (computeTaskInfo[curTaskId].rowLine > this->blockHeight) {
                computeTaskInfo[curTaskId].rowLine = this->blockHeight;
            }
        } else {
            computeTaskInfo[curTaskId].vGradRightOffset =
                backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].headId * this->headDim;

            computeTaskInfo[curTaskId].colLine =
                computeTaskInfo[curTaskId].curSeqLen - computeTaskInfo[curTaskId].colId * this->blockHeight;
            if (computeTaskInfo[curTaskId].colLine > this->blockHeight) {
                computeTaskInfo[curTaskId].colLine = this->blockHeight;
            }
        }
    }

    __aicore__ inline void DoJaggedQKMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;

        this->qkMatmul.SetTail(computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, this->headDim);
        this->DoQKMatmulImpl(computeTaskInfo[curTaskId].qkLeftOffset,
                             computeTaskInfo[curTaskId].qkRightOffset,
                             outOffset);
    }

    __aicore__ inline void DoJaggedGVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;

        this->qkMatmul.SetTail(computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, this->headDim);
        this->DoGVMatmulImpl(computeTaskInfo[curTaskId].qkLeftOffset,
                             computeTaskInfo[curTaskId].qkRightOffset,
                             outOffset);
    }

    __aicore__ inline void DoJaggedQGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;
        int64_t qGradRightOffset = computeTaskInfo[curTaskId].vGradRightOffset;

        if (!this->enableBias) {
            outOffset =
                backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].headId * computeTaskInfo[curTaskId].curSeqLen * this->headDim +
                computeTaskInfo[curTaskId].rowId * this->blockHeight * this->headDim;
            qGradRightOffset = computeTaskInfo[curTaskId].qGradRightOffset;
        }

        bool isNew = computeTaskInfo[curTaskId].colId == 0;

        this->qGradMatmul.SetTail(
            computeTaskInfo[curTaskId].rowLine, this->headDim, computeTaskInfo[curTaskId].colLine);
        this->DoQGradMatmulImpl(computeTaskInfo[curTaskId].kGradLeftOffset, qGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoJaggedKGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        bool isNew = false;
        if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
            isNew = computeTaskInfo[curTaskId].rowId == computeTaskInfo[curTaskId].colId;
        } else {
            isNew = computeTaskInfo[curTaskId].rowId == 0;
        }

        this->kGradMatmul.SetTail(
            computeTaskInfo[curTaskId].colLine, this->headDim, computeTaskInfo[curTaskId].rowLine);
        this->DoKGradMatmulImpl(computeTaskInfo[curTaskId].kGradLeftOffset,
                                computeTaskInfo[curTaskId].vGradRightOffset,
                                outOffset, isNew);
    }

    __aicore__ inline void DoJaggedVGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;

        int64_t scoreTempOffset = midResultIdx * this->blockHeight * this->blockHeight;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        bool isNew = false;
        if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
            isNew = computeTaskInfo[curTaskId].rowId == computeTaskInfo[curTaskId].colId;
        } else {
            isNew = computeTaskInfo[curTaskId].rowId == 0;
        }

        this->vGradMatmul.SetTail(
            computeTaskInfo[curTaskId].colLine, this->headDim, computeTaskInfo[curTaskId].rowLine);
        this->DoVGradMatmulImpl(scoreTempOffset,
                                computeTaskInfo[curTaskId].vGradRightOffset,
                                outOffset, isNew);
    }

    __aicore__ inline void VecScoreJagged(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t attnBiasOffset =
            computeTaskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].colId * this->blockHeight;
        int64_t attnBiasDiagonalOffset =
            computeTaskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].colId * this->blockHeight * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].rowId * this->blockHeight;
        
        int64_t maskOffset = 0;
        if (IfMask(this->maskType, MaskType::MASK_CUSTOM)) {
            maskOffset = computeTaskInfo[curTaskId].batchId * this->headNum * this->maxSeqLen * this->maxSeqLen +
                         computeTaskInfo[curTaskId].headId * this->maxSeqLen * this->maxSeqLen +
                         computeTaskInfo[curTaskId].rowId * this->blockHeight * this->maxSeqLen +
                         computeTaskInfo[curTaskId].colId * this->blockHeight;
        }

        bool useMask = false;
        if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
            useMask = computeTaskInfo[curTaskId].rowId == computeTaskInfo[curTaskId].colId;
        } else if (IfMask(this->maskType, MaskType::MASK_CUSTOM)) {
            useMask = true;
        }

        this->VecScoreImpl(taskId, attnBiasOffset, attnBiasDiagonalOffset, maskOffset,
                           computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, useMask);
    }

    __aicore__ inline void DoTransJagged(
        int64_t taskId, GlobalTensor<float> from, GlobalTensor<qType> to, bool isCol = true)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t fromOffset = midResultIdx * this->blockHeight * this->headDim;
        int64_t toOffset = 0;
        int64_t total = 0;
        if (isCol) {
            toOffset = backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum *
                       this->headDim + computeTaskInfo[curTaskId].colId * this->blockHeight * this->headNum *
                       this->headDim + computeTaskInfo[curTaskId].headId * this->headDim;
            total = computeTaskInfo[curTaskId].colLine * this->headDim;
        } else {
            toOffset = backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum *
                       this->headDim + computeTaskInfo[curTaskId].rowId * this->blockHeight * this->headNum *
                       this->headDim + computeTaskInfo[curTaskId].headId * this->headDim;
            total = computeTaskInfo[curTaskId].rowLine * this->headDim;
        }

        this->DoTransImpl(from, to, fromOffset, toOffset, total);
    }

    __aicore__ inline void FirstJaggedStagePipeline(int64_t taskId)
    {
        DoJaggedQKMatmul(taskId);
        DoJaggedGVMatmul(taskId);
        if (taskId > 1) {
            DoJaggedVGradMatmul(taskId - TWO);
            DoJaggedKGradMatmul(taskId - TWO);
            if (!this->enableBias) {
                DoJaggedQGradMatmul(taskId - TWO);
            }
        }
        if (taskId > 0) {
            VecScoreJagged(taskId - 1);
        }

        this->qkMatmul.WaitIterateAll();
        this->qkMatmul.End();
        this->qkMatmul.WaitIterateAll();
        this->qkMatmul.End();
        if (taskId > 1) {
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            if (!this->enableBias) {
                this->qGradMatmul.WaitIterateAll();
                this->qGradMatmul.End();
            }
            if (computeTaskInfo[(taskId - TWO) % COMPUTE_PIPE_NUM].accumId !=
                computeTaskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId) {
                DoTransJagged(taskId - TWO, this->vGradAccumTemp, this->vGrad);
                DoTransJagged(taskId - TWO, this->kGradAccumTemp, this->kGrad);
            }
        }
    }

    __aicore__ inline void FirstJaggedStageEnding(int64_t taskId)
    {
        if (taskId > 1) {
            DoJaggedVGradMatmul(taskId - TWO);
            DoJaggedKGradMatmul(taskId - TWO);
            if (!this->enableBias) {
                DoJaggedQGradMatmul(taskId - TWO);
            }
            VecScoreJagged(taskId - 1);
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            if (!this->enableBias) {
                this->qGradMatmul.WaitIterateAll();
                this->qGradMatmul.End();
            }
            if (computeTaskInfo[(taskId - TWO) % COMPUTE_PIPE_NUM].accumId !=
                computeTaskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId) {
                DoTransJagged(taskId - TWO, this->vGradAccumTemp, this->vGrad);
                DoTransJagged(taskId - TWO, this->kGradAccumTemp, this->kGrad);
            }

            DoJaggedVGradMatmul(taskId - 1);
            DoJaggedKGradMatmul(taskId - 1);
            if (!this->enableBias) {
                DoJaggedQGradMatmul(taskId - 1);
            }
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            if (!this->enableBias) {
                this->qGradMatmul.WaitIterateAll();
                this->qGradMatmul.End();
            }
            DoTransJagged(taskId - 1, this->vGradAccumTemp, this->vGrad);
            DoTransJagged(taskId - 1, this->kGradAccumTemp, this->kGrad);
        }

        if (taskId == 1) {
            VecScoreJagged(taskId - 1);
            DoJaggedVGradMatmul(taskId - 1);
            DoJaggedKGradMatmul(taskId - 1);
            if (!this->enableBias) {
                DoJaggedQGradMatmul(taskId - 1);
            }
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            if (!this->enableBias) {
                this->qGradMatmul.WaitIterateAll();
                this->qGradMatmul.End();
            }
            DoTransJagged(taskId - 1, this->vGradAccumTemp, this->vGrad);
            DoTransJagged(taskId - 1, this->kGradAccumTemp, this->kGrad);
        }
    }

    __aicore__ inline void ComputeJaggedFirst()
    {
        int64_t taskId = 0;

        if (startColBlock >= endColBlock) {
            return;
        }

        GenerateFirstTask();
        for (auto gColId = startColBlock; gColId < endColBlock; gColId += 1) {
            int64_t colId = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].colId;
            int64_t rowLimit = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].blockLimit;

            for (int64_t rowId = 0; rowId < rowLimit; rowId++) {
                if (IfMask(this->maskType, MaskType::MASK_TRIL) && rowId < colId) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                int64_t nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;
                computeTaskInfo[curTaskId].rowId = rowId;
                CalcBaseOffsetsJagged(taskId);

                FirstJaggedStagePipeline(taskId);

                taskId += 1;
                computeTaskInfo[nextTaskId] = computeTaskInfo[curTaskId];
                computeTaskInfo[nextTaskId].taskId = taskId;
            }

            UpdateNextBlock(taskId);
        }

        FirstJaggedStageEnding(taskId);
    }

    __aicore__ inline void SecondJaggedStagePipeline(int64_t taskId)
    {
        DoJaggedQGradMatmul(taskId);
        if (taskId > 0) {
            if (computeTaskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId !=
                computeTaskInfo[taskId % COMPUTE_PIPE_NUM].accumId) {
                DoTransJagged(taskId - 1, this->kGradAccumTemp, this->qGrad, false);
            }
        }
        this->qGradMatmul.WaitIterateAll();
        this->qGradMatmul.End();
    }

    __aicore__ inline void ComputeJaggedSecond()
    {
        SyncAll();
        if (startRowBlock >= endRowBlock) {
            return;
        }

        int64_t taskId = 0;
        GenerateFirstTask(false);
        for (int64_t gRowId = startRowBlock; gRowId < endRowBlock; gRowId += 1) {
            int64_t rowId = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].rowId;
            int64_t colLimit = computeTaskInfo[taskId % COMPUTE_PIPE_NUM].blockLimit;

            for (int64_t colId = 0; colId < colLimit; colId++) {
                if (IfMask(this->maskType, MaskType::MASK_TRIL) && rowId < colId) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                int64_t nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;
                computeTaskInfo[curTaskId].colId = colId;
                CalcBaseOffsetsJagged(curTaskId, false);

                SecondJaggedStagePipeline(taskId);

                taskId++;
                computeTaskInfo[nextTaskId] = computeTaskInfo[curTaskId];
                computeTaskInfo[nextTaskId].taskId = taskId;
            }
            UpdateNextBlock(taskId, false);
        }

        if (taskId > 0) {
            DoTransJagged(taskId - 1, this->kGradAccumTemp, this->qGrad, false);
        }
    }

    __aicore__ inline void CopyQGradToOutput()
    {
        SyncAll();

        if (GetBlockIdx() == 0) {
            this->DoCopyQGrad(backwardTilingData->seqOffset);
        }
    }

protected:
    uint32_t startColBlock;
    uint32_t endColBlock;
    uint32_t startRowBlock;
    uint32_t endRowBlock;

    JaggedTaskInfo computeTaskInfo[COMPUTE_PIPE_NUM];

    HstuDenseBackwardTilingData* __restrict backwardTilingData {nullptr};
};

} // namespace HstuDenseBackward

#endif