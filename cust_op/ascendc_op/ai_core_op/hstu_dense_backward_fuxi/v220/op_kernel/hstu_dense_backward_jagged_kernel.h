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

#include "hstu_dense_backward_kernel_matmul_fuxi.h"

namespace HstuDenseBackwardFuxi {

constexpr int64_t PREV_TASK_OFFSET = 1;
constexpr int64_t TWO_PREV_TASK_OFFSET = 2;

template <typename qType> class HstuDenseBackwardJaggedKernelFuxi : public HstuDenseBackwardKernelMatmulFuxi<qType> {
public:
    __aicore__ inline HstuDenseBackwardJaggedKernelFuxi() {}

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
                          &tilingData.vGradMatmul, this->biasMaskMatmul,
                          &tilingData.biasMaskMatmul);
        uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
        this->qkMatmul.SetUserDefInfo(tilingPtr);
        this->qGradMatmul.SetUserDefInfo(tilingPtr);
        this->kGradMatmul.SetUserDefInfo(tilingPtr);
        this->vGradMatmul.SetUserDefInfo(tilingPtr);
        this->biasMaskMatmul.SetUserDefInfo(tilingPtr);

        this->Init(args);
        this->PreInit();

        this->ComputeJaggedFirst();
        this->ComputeJaggedSecond();
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
        startBlock = isCol ? startColBlock : startRowBlock;

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

            uint32_t curSeqLen = backwardTilingData->seqOffset[computeTaskInfo[curTask].batchId + 1] -
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

    __aicore__ inline void CalcBaseOffsetsJagged(int64_t curTaskId, bool isCol = true)
    {
        computeTaskInfo[curTaskId].qkLeftOffset =
            backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].headId * this->headDim;

        computeTaskInfo[curTaskId].qkRightOffset =
            backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
            computeTaskInfo[curTaskId].headId * this->headDim;

        computeTaskInfo[curTaskId].kGradLeftOffset =
            computeTaskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].colId * this->blockHeight;

        if (isCol) {
            computeTaskInfo[curTaskId].vGradRightOffset =
                backwardTilingData->seqOffset[computeTaskInfo[curTaskId].batchId] * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
                computeTaskInfo[curTaskId].headId * this->headDim;

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
                             computeTaskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoJaggedGVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;

        this->qkMatmul.SetTail(computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, this->headDim);
        this->DoGVMatmulImpl(computeTaskInfo[curTaskId].qkLeftOffset,
                             computeTaskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoJaggedGpVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;
        this->qkMatmul.SetTail(computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, this->headDim);
        this->DoGpVMatmulImpl(computeTaskInfo[curTaskId].qkLeftOffset,
                              computeTaskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoJaggedGtVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;
        this->qkMatmul.SetTail(computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, this->headDim);
        this->DoGtVMatmulImpl(computeTaskInfo[curTaskId].qkLeftOffset,
                              computeTaskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoJaggedQGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        bool isNew = computeTaskInfo[curTaskId].colId == 0;

        this->qGradMatmul.SetTail(
            computeTaskInfo[curTaskId].rowLine, this->headDim, computeTaskInfo[curTaskId].colLine);
        this->DoQGradMatmulImpl(computeTaskInfo[curTaskId].kGradLeftOffset,
                                computeTaskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoJaggedKGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        auto colId = IfMask(this->maskType, MaskType::MASK_TRIL) ? computeTaskInfo[curTaskId].colId : 0;
        bool isNew = computeTaskInfo[curTaskId].rowId == colId;

        this->kGradMatmul.SetTail(
            computeTaskInfo[curTaskId].colLine, this->headDim, computeTaskInfo[curTaskId].rowLine);
        this->DoKGradMatmulImpl(computeTaskInfo[curTaskId].kGradLeftOffset,
                                computeTaskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoJaggedVGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;

        int64_t scoreTempOffset = midResultIdx * this->blockHeight * this->blockHeight;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        auto colId = IfMask(this->maskType, MaskType::MASK_TRIL) ? computeTaskInfo[curTaskId].colId : 0;
        bool isNew = computeTaskInfo[curTaskId].rowId == colId;

        this->vGradMatmul.SetTail(
            computeTaskInfo[curTaskId].colLine, this->headDim, computeTaskInfo[curTaskId].rowLine);
        this->DoVGradMatmulImpl(scoreTempOffset, computeTaskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoJaggedBtGtMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;

        int64_t scoreTempOffset = midResultIdx * this->blockHeight * this->blockHeight;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        auto colId = IfMask(this->maskType, MaskType::MASK_TRIL) ? computeTaskInfo[curTaskId].colId : 0;
        bool isNew = computeTaskInfo[curTaskId].rowId == colId;

        this->vGradMatmul.SetTail(
            computeTaskInfo[curTaskId].colLine, this->headDim, computeTaskInfo[curTaskId].rowLine);
        this->DoBtGtMatmulImpl(scoreTempOffset, computeTaskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoJaggedBpGpMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t midAccumIdx = computeTaskInfo[curTaskId].accumId % MID_USE_TIMES;

        int64_t scoreTempOffset = midResultIdx * this->blockHeight * this->blockHeight;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        auto colId = IfMask(this->maskType, MaskType::MASK_TRIL) ? computeTaskInfo[curTaskId].colId : 0;
        bool isNew = computeTaskInfo[curTaskId].rowId == colId;

        this->vGradMatmul.SetTail(
            computeTaskInfo[curTaskId].colLine, this->headDim, computeTaskInfo[curTaskId].rowLine);
        this->DoBpGpMatmulImpl(scoreTempOffset, computeTaskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
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
        
        int64_t btsOffset =
            computeTaskInfo[curTaskId].batchId * this->biasGradSeqLen * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].colId * this->blockHeight;
        int64_t bposOffset =
            computeTaskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
            computeTaskInfo[curTaskId].colId * this->blockHeight;
        
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
                           computeTaskInfo[curTaskId].rowLine, computeTaskInfo[curTaskId].colLine, useMask,
                           btsOffset, bposOffset);
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

    __aicore__ inline void StartStage1TaskA(int64_t taskId)
    {
        DoJaggedQKMatmul(taskId);
        DoJaggedGVMatmul(taskId);
        if (this->enableBias) {
            DoJaggedGpVMatmul(taskId);
            DoJaggedGtVMatmul(taskId);
        }
    }

    __aicore__ inline void WaitStage1TaskA()
    {
        this->qkMatmul.WaitIterateAll();
        this->qkMatmul.End();
        this->qkMatmul.WaitIterateAll();
        this->qkMatmul.End();
        // gtv gpV
        if (this->enableBias) {
            this->qkMatmul.WaitIterateAll();
            this->qkMatmul.End();
            this->qkMatmul.WaitIterateAll();
            this->qkMatmul.End();
        }
    }

    __aicore__ inline void StartStage1TaskB(int64_t taskId)
    {
        VecScoreJagged(taskId);
    }

    __aicore__ inline void StartStage1TaskC(int64_t taskId)
    {
        DoJaggedVGradMatmul(taskId);
        DoJaggedKGradMatmul(taskId);
        if (this->enableBias) {
            DoJaggedBtGtMatmul(taskId);
            DoJaggedBpGpMatmul(taskId);
        }
    }

    __aicore__ inline void WaitStage1TaskC()
    {
        this->vGradMatmul.WaitIterateAll();
        this->vGradMatmul.End();
        this->kGradMatmul.WaitIterateAll();
        this->kGradMatmul.End();
        // btGt bpGp
        if (this->enableBias) {
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
        }
    }

    __aicore__ inline void PostStage1TaskC(int64_t taskId)
    {
        DoTransJagged(taskId, this->kGradAccumTemp, this->kGrad);
        // Gv = Gv1 + BtGt + BpGp
        DoTransJagged(taskId, this->vGradAccumTemp, this->vGrad);
        if (this->enableBias) {
            DoTransJagged(taskId, this->tempBtsGtsAccum, this->vbtsGrad);
            DoTransJagged(taskId, this->tempBposGposAccum, this->vbposGrad);
        }
    }

    __aicore__ inline void FirstJaggedStagePipeline(int64_t taskId)
    {
        int64_t prevTaskId = taskId - PREV_TASK_OFFSET;
        int64_t twoPrevTaskId = taskId - TWO_PREV_TASK_OFFSET;

        StartStage1TaskA(taskId);

        if (taskId >= TWO_PREV_TASK_OFFSET) {
            StartStage1TaskC(twoPrevTaskId);
        }

        if (taskId >= PREV_TASK_OFFSET) {
            StartStage1TaskB(prevTaskId);
        }

        WaitStage1TaskA();

        if (taskId >= TWO_PREV_TASK_OFFSET) {
            WaitStage1TaskC();
            if (computeTaskInfo[(prevTaskId) % COMPUTE_PIPE_NUM].accumId !=
                computeTaskInfo[(twoPrevTaskId) % COMPUTE_PIPE_NUM].accumId) {
                PostStage1TaskC(twoPrevTaskId);
            }
        }
    }

    __aicore__ inline void FirstJaggedStageEnding(int64_t taskId)
    {
        int64_t prevTaskId = taskId - PREV_TASK_OFFSET;
        int64_t twoPrevTaskId = taskId - TWO_PREV_TASK_OFFSET;
        if (twoPrevTaskId >= 0) {
            StartStage1TaskC(twoPrevTaskId);
            StartStage1TaskB(prevTaskId);
            WaitStage1TaskC();

            if (computeTaskInfo[(prevTaskId) % COMPUTE_PIPE_NUM].accumId !=
                computeTaskInfo[(twoPrevTaskId) % COMPUTE_PIPE_NUM].accumId) {
                PostStage1TaskC(twoPrevTaskId);
            }

            StartStage1TaskC(prevTaskId);
            WaitStage1TaskC();
            PostStage1TaskC(prevTaskId);
        }

        if (prevTaskId == 0) {
            StartStage1TaskB(prevTaskId);
            StartStage1TaskC(prevTaskId);
            WaitStage1TaskC();
            PostStage1TaskC(prevTaskId);
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
                CalcBaseOffsetsJagged(curTaskId);

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

protected:
    uint32_t startColBlock;
    uint32_t endColBlock;
    uint32_t startRowBlock;
    uint32_t endRowBlock;

    JaggedTaskInfo computeTaskInfo[COMPUTE_PIPE_NUM];

    HstuDenseBackwardFuxiTilingData* __restrict backwardTilingData {nullptr};
};

} // namespace HstuDenseBackwardFuxi

#endif