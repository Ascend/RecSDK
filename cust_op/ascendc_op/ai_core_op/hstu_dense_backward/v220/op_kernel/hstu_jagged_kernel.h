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

#ifndef HSTU_JAGGED_F16_R0_KERNEL_H
#define HSTU_JAGGED_F16_R0_KERNEL_H

#include <cstdint>
#include "hstu_common_const.h"
#include "hstu_mask.h"
#include "hstu_split_core_policy.h"
#include "matmul_mgmt.h"
#include "q_accum.h"
#include "trans.h"
#include "vector_score.h"

using HstuDenseBackward::BlockMaskGenerator;
using HstuDenseBackward::BlockMaskParams;

namespace HstuDenseBackward {

template <typename seqOffsetType>
__aicore__ inline int64_t GetBatchSizeFromJaggedOffsetThis(GlobalTensor<seqOffsetType>& seqOffsetData,
                                                           int32_t seqOffsetLens)
{
    if (seqOffsetLens <= 0) {
        return 0;
    }

    // 二分法找出有效batch
    int64_t maxValue = seqOffsetData.GetValue(seqOffsetLens - 1);
    int64_t left = 0;
    int64_t right = seqOffsetLens - 1;
    int64_t firstMaxIdx = seqOffsetLens - 1;
    while (left <= right) {
        int64_t mid = left + (right - left) / 2;  // 二分法除以2找到剩余中间位置
        if (seqOffsetData.GetValue(mid) == maxValue) {
            firstMaxIdx = mid;
            right = mid - 1;
        } else if (seqOffsetData.GetValue(mid) < maxValue) {
            left = mid + 1;
        }
    }

    int64_t batchSize = static_cast<int64_t>(firstMaxIdx);
    return batchSize;
}

struct ColLineBaseInfo {
    uint32_t batchId;
    uint32_t headId;
    uint32_t colId;
    uint32_t colLine;
    uint32_t accumId;
    uint32_t blockLimit;
    uint32_t curSeqLen;
};

struct JaggedTaskInfoColMajor {
    uint32_t taskId;  // 基本块任务id，参与临时存储块的偏移计算
    ColLineBaseInfo* colBlockPtr;
    uint32_t rowId;
    int64_t kOrVOffset;  // 基本块qk/gv乘法的左矩阵内存偏移
    int64_t qOrGoffset;  // 基本块qk/gv乘法的右矩阵内存偏移
    uint32_t rowLine;    // 基本块需要计算的行数

    __aicore__ inline const uint32_t GetTaskId()
    {
        return taskId;
    }
    __aicore__ inline const int64_t GetKOrVOffset()
    {
        return kOrVOffset;
    }
    __aicore__ inline const int64_t GetQOrGoffset()
    {
        return qOrGoffset;
    }
    __aicore__ inline const uint32_t GetRowLine()
    {
        return rowLine;
    }
    __aicore__ inline const uint32_t GetColLine()
    {
        return colBlockPtr->colLine;
    }

    __aicore__ inline const uint32_t GetBatchId()
    {
        return colBlockPtr->batchId;
    }

    __aicore__ inline const uint32_t GetHeadId()
    {
        return colBlockPtr->headId;
    }

    __aicore__ inline const uint32_t GetRowId()
    {
        return rowId;
    }

    __aicore__ inline const uint32_t GetColId()
    {
        return colBlockPtr->colId;
    }

    __aicore__ inline const uint32_t GetAccumId()
    {
        return colBlockPtr->accumId;
    }

    __aicore__ inline const uint32_t GetBlockLimit()
    {
        return colBlockPtr->blockLimit;
    }

    __aicore__ inline const uint32_t GetCurSeqLen()
    {
        return colBlockPtr->curSeqLen;
    }
};

template <typename qType, typename seqOffsetType, uint32_t blockHeightQ, uint32_t blockHeightK, uint32_t headDimPadding,
          class MatmulMgmtType, class VectorScoreType>
class HstuJaggedKernel {
public:
    using MmInterface =
        HstuMatmulMgmtInterface<qType, blockHeightQ, blockHeightK, HstuDenseBackwardTilingData, MatmulMgmtType>;
    using VsInterface = VectorScoreInterface<qType, VectorScoreType>;
    __aicore__ inline HstuJaggedKernel() {}

    __aicore__ inline void Compute(Args& args, MmInterface* mmInterface, VsInterface* vectorScoreInterface)

    {
        mm_mgmt_ = mmInterface;
        vectorScoreInterface_ = vectorScoreInterface;
        Init(args);
        InitBlockSplit(args);

        ComputeJaggedFirst();
        SyncAll();
        qAccumKernel_.DoCopyQGrad();
    }
    // 初始化
    __aicore__ inline void InitShapeInfo(Args& args)
    {
        // 基础信息初始化
        totalBatchSize_ = backwardTilingData_->seqLen;
        batchSize_ = backwardTilingData_->batchSize;
        maxSeqLen_ = backwardTilingData_->maxSeqLen;
        headDim_ = backwardTilingData_->headDim;
        headNum_ = backwardTilingData_->headNum;
        biasGradSeqLen_ = backwardTilingData_->biasGradSeqLen;
    }

    __aicore__ inline void InitAttrInfo(Args& args)
    {
        siluScale_ = backwardTilingData_->siluScale;
        targetGroupSize_ = backwardTilingData_->targetGroupSize;
        alpha_ = backwardTilingData_->alpha;
        aivNum_ = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        enableTargetMask_ = backwardTilingData_->enableTargetMask == 1;
        enableContextMask_ = backwardTilingData_->enableContextMask == 1;
        enableBias_ = backwardTilingData_->enableBias == 1;
        maskType_ = MaskType(backwardTilingData_->maskType);
    }
    __aicore__ inline void InitGtInfo(Args& args)
    {
        numContextGt_.SetGlobalBuffer(reinterpret_cast<__gm__ seqOffsetType*>(args.numContext), batchSize_);
        numTargetGt_.SetGlobalBuffer(reinterpret_cast<__gm__ seqOffsetType*>(args.numTarget), batchSize_);
        seqOffsetsGt_.SetGlobalBuffer(reinterpret_cast<__gm__ seqOffsetType*>(args.seqOffset), this->batchSize_ + 1);
        int64_t totalElementOfAttnBias = batchSize_ * maxSeqLen_ * headNum_ * headDim_;
        baisGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.attnBias), totalElementOfAttnBias);
        maskGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.mask),
                                totalElementOfAttnBias * totalElementOfAttnBias);
        biasGradGt_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.attnBiasGrad), totalElementOfAttnBias);
    }

    __aicore__ inline void InitLayoutInfo(Args& args)
    {
        // Layout初始化
        uint32_t lastDimStride1 = 1;
        qLayout_ = TNDLayout(MakeShape(totalBatchSize_, headNum_, headDim_),
                             MakeStride(headNum_ * headDim_, headDim_, lastDimStride1));
        kLayout_ = TNDLayout(MakeShape(totalBatchSize_, headNum_, headDim_),
                             MakeStride(headNum_ * headDim_, headDim_, lastDimStride1));
        pipeBlockLayout_ = PipeBlockLayout(MakeShape(blockHeightQ, blockHeightK, headDim_),
                                           MakeStride(blockHeightK * headDim_, headDim_, lastDimStride1));
        bnssLayout_ = BNSSLayout(
            MakeShape(batchSize_, headNum_, maxSeqLen_, maxSeqLen_),
            MakeStride(headNum_ * maxSeqLen_ * maxSeqLen_, maxSeqLen_ * maxSeqLen_, maxSeqLen_, lastDimStride1));
    }

    __aicore__ inline void Init(Args& args)
    {
        backwardTilingData_ = args.tilingDataPtr;
        InitShapeInfo(args);
        InitAttrInfo(args);
        InitGtInfo(args);
        InitLayoutInfo(args);
        // Matmul 初始化
        AddrArgs addrArgs = {args.grad, args.q, args.k, args.v, args.qGrad, args.kGrad, args.vGrad, args.workspace};
        BaseShapeArgs baseShape = {totalBatchSize_, batchSize_, headNum_, headDim_, maxSeqLen_};
        mm_mgmt_->Init(&addrArgs, &baseShape);

        // QAccum初始化
        qAccumKernel_.Init(&pipe, &baseShape, mm_mgmt_->qGradAccumTemp_, mm_mgmt_->qGrad_, seqOffsetsGt_,
                           GetBlockNum() * VCORE_NUM_IN_ONE_AIC);
        // Trans初始化
        transKernel_.Init(&pipe, headNum_, headDim_);
        // VectorScore初始化
        VectorScoreAttrs vectorScoreAttrs = {siluScale_, alpha_, enableBias_, maskType_};
        VectorScoreGtInfo<qType> vectorScoreGtInfo = {mm_mgmt_->qkTemp_, mm_mgmt_->gvTemp_, maskGt_, baisGt_,
                                                      biasGradGt_};
        vectorScoreInterface_->Init(&pipe, bnssLayout_, &vectorScoreAttrs, &vectorScoreGtInfo);
    }

    __aicore__ inline void InitBlockSplit(Args& args)
    {
        const int blockId = GetBlockIdx();
        this->batchSize_ = GetBatchSizeFromJaggedOffsetThis(seqOffsetsGt_, this->batchSize_ + 1);

        int64_t bxn = this->batchSize_ * headNum_;
        auto coreNum = backwardTilingData_->aivNum;

        auto taskAssigner = BlockTaskAssign(seqOffsetsGt_, coreNum, blockHeightQ, batchSize_, headNum_);
        int colBlock[2] = {0};
        int rowBlock[2] = {0};

        taskAssigner.ComputeCausal(colBlock, blockId, true);
        taskAssigner.ComputeCausal(rowBlock, blockId, false);

        startColBlock_ = colBlock[0];
        endColBlock_ = colBlock[1];
    }

    __aicore__ inline ColLineBaseInfo GenerateFirstTask(bool isCol = true)
    {
        uint32_t batchId = 0;
        uint32_t curSeqLen = 0;
        uint32_t curBatchStartBlock = 0;
        uint32_t startBlock = 0;

        startBlock = startColBlock_;

        while (batchId < MAX_BATCH_SIZE) {
            curSeqLen = seqOffsetsGt_.GetValue(batchId + 1) - seqOffsetsGt_.GetValue(batchId);
            auto curBatchBlock = headNum_ * ((curSeqLen + blockHeightQ - 1) / blockHeightQ);
            if (curBatchStartBlock + curBatchBlock > startBlock) {
                break;
            }
            curBatchStartBlock += curBatchBlock;
            batchId++;
        }

        uint32_t curBlockIdInBatch = startBlock - curBatchStartBlock;
        uint32_t curHeadBlock = (curSeqLen + blockHeightQ - 1) / blockHeightQ;

        uint32_t headId = curBlockIdInBatch / curHeadBlock;

        uint32_t colId = curBlockIdInBatch % curHeadBlock;
        uint32_t colLine = curSeqLen - colId * blockHeightQ;
        colLine = colLine > blockHeightQ ? blockHeightQ : colLine;
        ColLineBaseInfo colBaseInfo = {batchId, headId, colId, colLine, 0, curHeadBlock, curSeqLen};
        return colBaseInfo;
    }

    __aicore__ inline ColLineBaseInfo UpdateNextBlock(const ColLineBaseInfo& lastColLineInfo)
    {
        uint32_t batchId = lastColLineInfo.batchId;
        uint32_t headId = lastColLineInfo.headId;
        uint32_t colId = lastColLineInfo.colId;
        uint32_t accumId = lastColLineInfo.accumId;
        uint32_t blockLimit = lastColLineInfo.blockLimit;
        uint32_t curSeqLen = lastColLineInfo.curSeqLen;
        uint32_t colLine;

        colId += 1;
        if (colId == lastColLineInfo.blockLimit) {
            colId = 0;
            headId += 1;
        }

        if (headId == headNum_) {
            headId = 0;
            batchId += 1;

            curSeqLen = seqOffsetsGt_.GetValue(batchId + 1) - seqOffsetsGt_.GetValue(batchId);
            auto curHeadBlock = (curSeqLen + blockHeightQ - 1) / blockHeightQ;

            blockLimit = curHeadBlock;
        }

        colLine = curSeqLen - colId * blockHeightQ;
        colLine = colLine > blockHeightQ ? blockHeightQ : colLine;

        accumId += 1;

        const ColLineBaseInfo colBaseInfo = {batchId, headId, colId, colLine, accumId, blockLimit, curSeqLen};
        return colBaseInfo;
    }

    __aicore__ inline void InitTaskInfoCalcBaseOffsetsJagged(int64_t taskId, uint32_t rowId, ColLineBaseInfo& colInfo)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        computeTaskInfo_[curTaskId].taskId = taskId;
        computeTaskInfo_[curTaskId].rowId = rowId;
        computeTaskInfo_[curTaskId].colBlockPtr = &colInfo;
        uint32_t totalBatchSizeId = seqOffsetsGt_.GetValue(colInfo.batchId);

        computeTaskInfo_[curTaskId].qOrGoffset =
            qLayout_(MakeCoord(totalBatchSizeId + rowId * blockHeightQ, colInfo.headId, 0));
        computeTaskInfo_[curTaskId].kOrVOffset =
            kLayout_(MakeCoord(totalBatchSizeId + colInfo.colId * blockHeightK, colInfo.headId, 0));
        computeTaskInfo_[curTaskId].rowLine = colInfo.curSeqLen - computeTaskInfo_[curTaskId].rowId * blockHeightQ;
        computeTaskInfo_[curTaskId].rowLine =
            computeTaskInfo_[curTaskId].rowLine > blockHeightQ ? blockHeightQ : computeTaskInfo_[curTaskId].rowLine;
    }

    __aicore__ inline void DoJaggedQKMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        mm_mgmt_->DoQkMatmul(midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                             computeTaskInfo_[curTaskId].kOrVOffset, computeTaskInfo_[curTaskId].rowLine,
                             computeTaskInfo_[curTaskId].GetColLine());
    }

    __aicore__ inline void DoJaggedGVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        mm_mgmt_->DoGvMatmul(midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                             computeTaskInfo_[curTaskId].kOrVOffset, computeTaskInfo_[curTaskId].rowLine,
                             computeTaskInfo_[curTaskId].GetColLine());
    }

    __aicore__ inline void DoJaggedQGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;

        int64_t qGradOutOffset =
            seqOffsetsGt_.GetValue(computeTaskInfo_[curTaskId].GetBatchId()) * headNum_ * headDim_ +
            computeTaskInfo_[curTaskId].GetHeadId() * computeTaskInfo_[curTaskId].GetCurSeqLen() * headDim_ +
            computeTaskInfo_[curTaskId].rowId * blockHeightQ * headDim_;

        mm_mgmt_->DoQGradMatmul(qGradOutOffset, midScoreOffset, computeTaskInfo_[curTaskId].kOrVOffset,
                                computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine());
    }

    __aicore__ inline void DoJaggedKGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (computeTaskInfo_[curTaskId].taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        int64_t kAccumOffset = (computeTaskInfo_[curTaskId].GetAccumId() % MID_USE_TIMES) * blockHeightK * headDim_;
        bool isFirstBlock = maskType_ == MaskType::MASK_TRIL ? blockMaskParams_[curTaskId].IsFirstBlockNeedOverride()
                                                             : computeTaskInfo_[curTaskId].GetRowId() == 0;
        mm_mgmt_->DoKGradMatmul(kAccumOffset, midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                                computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine(),
                                isFirstBlock);
    }

    __aicore__ inline void DoJaggedVGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (computeTaskInfo_[curTaskId].taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        int64_t vAccumOffset = (computeTaskInfo_[curTaskId].GetAccumId() % MID_USE_TIMES) * blockHeightK * headDim_;
        bool isFirstBlock = maskType_ == MaskType::MASK_TRIL ? blockMaskParams_[curTaskId].IsFirstBlockNeedOverride()
                                                             : computeTaskInfo_[curTaskId].GetRowId() == 0;
        mm_mgmt_->DoVGradMatmul(vAccumOffset, midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                                computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine(),
                                isFirstBlock);
    }

    __aicore__ inline void VecScoreJagged(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t maskOffset = 0;

        bool useMask = false;
        useMask = blockMaskParams_[curTaskId].NeedMask();
        int64_t batchId = computeTaskInfo_[curTaskId].GetBatchId();
        int64_t headId = computeTaskInfo_[curTaskId].GetHeadId();
        int64_t rowId = computeTaskInfo_[curTaskId].GetRowId();
        int64_t colId = computeTaskInfo_[curTaskId].GetColId();
        vectorScoreInterface_->VecScoreJagged(curTaskId * blockHeightQ * blockHeightK, batchId, headId, rowId, colId,
                                              computeTaskInfo_[curTaskId].rowLine,
                                              computeTaskInfo_[curTaskId].GetColLine(), blockMaskParams_[curTaskId]);
    }

    __aicore__ inline void DoTransJagged(int64_t taskId, GlobalTensor<float> from, GlobalTensor<qType> to,
                                         bool isCol = true)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = computeTaskInfo_[curTaskId].GetAccumId() % MID_USE_TIMES;
        int64_t fromOffset = midResultIdx * blockHeightQ * headDim_;
        int64_t toOffset = 0;
        int64_t total = 0;
        if (isCol) {
            toOffset = computeTaskInfo_[curTaskId].kOrVOffset;
            total = computeTaskInfo_[curTaskId].GetColLine() * headDim_;
        } else {
            toOffset = computeTaskInfo_[curTaskId].qOrGoffset;
            total = computeTaskInfo_[curTaskId].rowLine * headDim_;
        }

        transKernel_.DoTransOfStrideHeadDim(from, to, fromOffset, toOffset, total);
    }

    __aicore__ inline void FirstJaggedStagePipeline(int64_t taskId)
    {
        DoJaggedQKMatmul(taskId);
        DoJaggedGVMatmul(taskId);
        if (taskId > 1) {
            DoJaggedVGradMatmul(taskId - TWO);
            DoJaggedKGradMatmul(taskId - TWO);
            DoJaggedQGradMatmul(taskId - TWO);
        }
        if (taskId > 0) {
            VecScoreJagged(taskId - 1);
        }
        mm_mgmt_->QkOrGvMatmulWait();
        mm_mgmt_->QkOrGvMatmulWait();
        if (taskId > 1) {
            mm_mgmt_->vGradMatmulWait();
            mm_mgmt_->kGradMatmulWait();
            mm_mgmt_->qGradMatmulWait();
            if (computeTaskInfo_[(taskId - TWO) % COMPUTE_PIPE_NUM].GetAccumId() !=
                computeTaskInfo_[(taskId - 1) % COMPUTE_PIPE_NUM].GetAccumId()) {
                DoTransJagged(taskId - TWO, mm_mgmt_->vGradAccumTemp_, mm_mgmt_->vGrad_);
                DoTransJagged(taskId - TWO, mm_mgmt_->kGradAccumTemp_, mm_mgmt_->kGrad_);
            }
        }
    }

    __aicore__ inline void FirstJaggedStageEnding(int64_t taskId)
    {
        if (taskId > 1) {
            DoJaggedVGradMatmul(taskId - TWO);
            DoJaggedKGradMatmul(taskId - TWO);
            DoJaggedQGradMatmul(taskId - TWO);
            VecScoreJagged(taskId - 1);
            mm_mgmt_->vGradMatmulWait();
            mm_mgmt_->kGradMatmulWait();
            mm_mgmt_->qGradMatmulWait();
            if (computeTaskInfo_[(taskId - TWO) % COMPUTE_PIPE_NUM].GetAccumId() !=
                computeTaskInfo_[(taskId - 1) % COMPUTE_PIPE_NUM].GetAccumId()) {
                DoTransJagged(taskId - TWO, mm_mgmt_->vGradAccumTemp_, mm_mgmt_->vGrad_);
                DoTransJagged(taskId - TWO, mm_mgmt_->kGradAccumTemp_, mm_mgmt_->kGrad_);
            }

            DoJaggedVGradMatmul(taskId - 1);
            DoJaggedKGradMatmul(taskId - 1);
            DoJaggedQGradMatmul(taskId - 1);

            mm_mgmt_->vGradMatmulWait();
            mm_mgmt_->kGradMatmulWait();
            mm_mgmt_->qGradMatmulWait();
            DoTransJagged(taskId - 1, mm_mgmt_->vGradAccumTemp_, mm_mgmt_->vGrad_);
            DoTransJagged(taskId - 1, mm_mgmt_->kGradAccumTemp_, mm_mgmt_->kGrad_);
        }

        if (taskId == 1) {
            VecScoreJagged(taskId - 1);
            DoJaggedVGradMatmul(taskId - 1);
            DoJaggedKGradMatmul(taskId - 1);
            DoJaggedQGradMatmul(taskId - 1);
            mm_mgmt_->vGradMatmulWait();
            mm_mgmt_->kGradMatmulWait();
            mm_mgmt_->qGradMatmulWait();
            DoTransJagged(taskId - 1, mm_mgmt_->vGradAccumTemp_, mm_mgmt_->vGrad_);
            DoTransJagged(taskId - 1, mm_mgmt_->kGradAccumTemp_, mm_mgmt_->kGrad_);
        }
    }
    __aicore__ inline int64_t GetNumContext(int64_t batchId)
    {
        if (enableContextMask_) {
            return numContextGt_.GetValue(batchId);
        }
        return 0;
    }

    __aicore__ inline int64_t GetNumTarget(int64_t batchId)
    {
        if (enableTargetMask_) {
            return numTargetGt_.GetValue(batchId);
        }
        return 0;
    }

    __aicore__ inline void ComputeJaggedFirst()
    {
        int64_t taskId = 0;

        if (startColBlock_ >= endColBlock_) {
            return;
        }

        ColLineBaseInfo ColLineInfoGlobal[COMPUTE_PIPE_NUM];
        ColLineInfoGlobal[startColBlock_ % COMPUTE_PIPE_NUM] = GenerateFirstTask();

        for (auto gColId = startColBlock_; gColId < endColBlock_; gColId += 1) {
            ColLineBaseInfo& thisColLineInfo = ColLineInfoGlobal[gColId % COMPUTE_PIPE_NUM];

            const int64_t colId = thisColLineInfo.colId;
            const int64_t rowLimit = thisColLineInfo.blockLimit;

            for (int64_t rowId = 0; rowId < rowLimit; rowId++) {
                JaggedTaskInfoColMajor& args = this->computeTaskInfo_[taskId % COMPUTE_PIPE_NUM];
                args.taskId = taskId;
                args.colBlockPtr = &thisColLineInfo;

                this->blockMaskParams_[taskId % COMPUTE_PIPE_NUM] = {static_cast<uint32_t>(rowId),
                                                                     static_cast<uint32_t>(colId),
                                                                     static_cast<uint32_t>(thisColLineInfo.curSeqLen),
                                                                     blockHeightQ,
                                                                     GetNumContext(thisColLineInfo.batchId),
                                                                     GetNumTarget(thisColLineInfo.batchId),
                                                                     targetGroupSize_,
                                                                     1};

                BlockMaskParams& maskinfo = this->blockMaskParams_[taskId % COMPUTE_PIPE_NUM];
                if (maskType_ == MaskType::MASK_TRIL && maskinfo.NoComputation()) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                int64_t nextTaskId = (taskId + 1) % COMPUTE_PIPE_NUM;
                InitTaskInfoCalcBaseOffsetsJagged(taskId, rowId, thisColLineInfo);

                FirstJaggedStagePipeline(taskId);

                taskId += 1;
            }
            const int64_t nextGCol = gColId + 1;
            ColLineInfoGlobal[nextGCol % COMPUTE_PIPE_NUM] =
                UpdateNextBlock(ColLineInfoGlobal[gColId % COMPUTE_PIPE_NUM]);
        }

        FirstJaggedStageEnding(taskId);
    }

    // Mode
    bool enableBias_ = false;
    MaskType maskType_ = MaskType::MASK_NONE;
    // targetMask
    bool enableTargetMask_ = false;
    bool enableContextMask_ = false;

    // Shape
    TNDLayout qLayout_;
    TNDLayout kLayout_;
    PipeBlockLayout pipeBlockLayout_;
    BNSSLayout bnssLayout_;

    uint32_t batchSize_ = 0;
    uint32_t maxSeqLen_ = 0;
    uint32_t biasGradSeqLen_ = 0;
    uint32_t headDim_ = 0;
    uint32_t headNum_ = 0;
    uint32_t totalBatchSize_ = 0;

    // Attr
    float siluScale_ = 1.0f;
    uint32_t aivNum_ = 1;

    // task
    BlockInfo taskInfo_[COMPUTE_PIPE_NUM];

    // MaskType
    int64_t targetGroupSize_ = 0;
    float alpha_ = 1.0f;
    BlockMaskParams blockMaskParams_[COMPUTE_PIPE_NUM] = {};

    // Tpipe
    TPipe pipe;  // pipe.InitBuffer等初始化

    // Gt
    GlobalTensor<seqOffsetType> numContextGt_;
    GlobalTensor<seqOffsetType> numTargetGt_;

    GlobalTensor<qType> baisGt_;
    GlobalTensor<qType> maskGt_;
    GlobalTensor<qType> biasGradGt_;

    // Matmul
    HstuMatmulMgmtInterface<qType, blockHeightQ, blockHeightK, HstuDenseBackwardTilingData, MatmulMgmtType>* mm_mgmt_;

    // QAccum
    qBlockAccumKernel<float, qType, seqOffsetType> qAccumKernel_;
    // Trans
    TransStrideHdDKernel<float, qType> transKernel_;
    // VectorScore
    HstuVectorScoreCommon<qType, blockHeightQ, blockHeightK> vectorScoreKernel_;
    VectorScoreInterface<qType, VectorScoreType>* vectorScoreInterface_;

protected:
    uint32_t startColBlock_ = 0;
    uint32_t endColBlock_ = 0;
    JaggedTaskInfoColMajor computeTaskInfo_[COMPUTE_PIPE_NUM] = {};  // 局部循环/函数内赋值, 默认初始化
    const HstuDenseBackwardTilingData* __restrict backwardTilingData_{
        nullptr};                               // Compute()赋值: backwardTilingData_ = args.tilingDataPtr
    GlobalTensor<seqOffsetType> seqOffsetsGt_;  // PreInit中SetGlobalBuffer
};
}  // namespace HstuDenseBackward
#endif