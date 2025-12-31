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
#include <cstdio>
#include "hstu_dense_backward_kernel_common.h"
#include "hstu_mask.h"
#include "hstu_split_core_policy.h"
#include "matmul_mgmt_j_f16_r0.h"

using HstuDenseBackward::BlockMaskGenerator;
using HstuDenseBackward::BlockMaskParams;

namespace HstuDenseBackward {

template <typename oType>
__aicore__ inline int64_t GetBatchSizeFromJaggedOffsetThis(GlobalTensor<oType>& seqOffsetData, int32_t seqOffsetLens)
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

struct JaggedFp8R0TaskInfo {
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
    __aicore__ inline const uint32_t GetKOrVOffset()
    {
        return kOrVOffset;
    }
    __aicore__ inline const uint32_t GetQOrGoffset()
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

template <typename qType, typename oType, uint32_t blockHeightQ, uint32_t blockHeightK, uint32_t headNum,
          uint32_t headDim>
class HstuJaggedF16R0Kernel {
public:
    __aicore__ inline HstuJaggedF16R0Kernel() {}

    __aicore__ inline void Compute(Args& args)
    {
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm_mgmt_.qkOrGvMatmul_, (TCubeTiling*)nullptr,
                          mm_mgmt_.vGradMatmul_, (TCubeTiling*)nullptr, mm_mgmt_.qGradMatmul_, (TCubeTiling*)nullptr,
                          mm_mgmt_.kGradMatmul_, (TCubeTiling*)nullptr);

        backwardTilingData_ = args.tilingDataPtr;

        this->Init(args);
        this->PreInit(args);

        this->ComputeJaggedFirst();
        this->CopyQGradToOutput();
    }

    __aicore__ inline void Init(Args& args)
    {
        batchSize_ = backwardTilingData_->batchSize;
        maxSeqLen_ = backwardTilingData_->maxSeqLen;

        biasGradSeqLen_ = backwardTilingData_->biasGradSeqLen;
        siluScale_ = backwardTilingData_->siluScale;
        targetGroupSize_ = backwardTilingData_->targetGroupSize;
        alpha_ = backwardTilingData_->alpha;
        aivNum_ = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;

        AddrArgs addrArgs = {args.grad, args.q, args.k, args.v, args.qGrad, args.kGrad, args.vGrad, args.workspace};
        BaseShapeArgs baseShape = {batchSize_, headNum, headDim, maxSeqLen_};
        mm_mgmt_.Init(&addrArgs, &baseShape);

        enableTargetMask_ = backwardTilingData_->enableTargetMask == 1;
        enableContextMask_ = backwardTilingData_->enableContextMask == 1;
        numContextGt_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.numContext), batchSize_);
        numTargetGt_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.numTarget), batchSize_);

        // 20通过ub的大小计算得到
        vecOnceDataNum_ = blockHeightQ * 20;

        pipe.InitBuffer(queueVecScoreQK_, 1, vecOnceDataNum_ * sizeof(float));
        pipe.InitBuffer(queueVecScoreGV_, 1, vecOnceDataNum_ * sizeof(float));
        pipe.InitBuffer(queueVecScoreMask_, 1, vecOnceDataNum_ * sizeof(float));
        pipe.InitBuffer(queueVecScoreBias_, 1, vecOnceDataNum_ * sizeof(float));
        pipe.InitBuffer(tbufMid_, vecOnceDataNum_ * sizeof(float));
        pipe.InitBuffer(tbufMidQk_, vecOnceDataNum_ * sizeof(float));
        pipe.InitBuffer(tbufMidGV_, vecOnceDataNum_ * sizeof(float));

        pipe.InitBuffer(queueOutputScore_, 1, vecOnceDataNum_ * sizeof(qType));
        pipe.InitBuffer(queueOutputBias_, 1, vecOnceDataNum_ * sizeof(qType));
    }

    __aicore__ inline void PreInit(Args& args)
    {
        const int blockId = GetBlockIdx();
        seqOffsetsGt_.SetGlobalBuffer(reinterpret_cast<__gm__ oType*>(args.seqOffset), this->batchSize_ + 1);
        this->batchSize_ = GetBatchSizeFromJaggedOffsetThis(seqOffsetsGt_, this->batchSize_ + 1);

        int64_t bxn = this->batchSize_ * headNum;
        auto coreNum = backwardTilingData_->aivNum;

        auto taskAssigner = BlockTaskAssign(seqOffsetsGt_, coreNum, blockHeightQ, batchSize_, headNum);
        int colBlock[2] = {0};
        int rowBlock[2] = {0};

        taskAssigner.ComputeCausal(colBlock, blockId, true);
        taskAssigner.ComputeCausal(rowBlock, blockId, false);

        startColBlock_ = colBlock[0];
        endColBlock_ = colBlock[1];
        startRowBlock_ = rowBlock[0];
        endRowBlock_ = rowBlock[1];
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
            auto curBatchBlock = headNum * ((curSeqLen + blockHeightQ - 1) / blockHeightQ);
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

        if (headId == headNum) {
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

        computeTaskInfo_[curTaskId].qOrGoffset = seqOffsetsGt_.GetValue(colInfo.batchId) * headNum * headDim +
                                                 rowId * blockHeightQ * headNum * headDim + colInfo.headId * headDim;

        computeTaskInfo_[curTaskId].kOrVOffset = seqOffsetsGt_.GetValue(colInfo.batchId) * headNum * headDim +
                                                 colInfo.colId * blockHeightQ * headNum * headDim +
                                                 colInfo.headId * headDim;

        computeTaskInfo_[curTaskId].rowLine = colInfo.curSeqLen - computeTaskInfo_[curTaskId].rowId * blockHeightQ;
        computeTaskInfo_[curTaskId].rowLine =
            computeTaskInfo_[curTaskId].rowLine > blockHeightQ ? blockHeightQ : computeTaskInfo_[curTaskId].rowLine;
    }

    __aicore__ inline void DoJaggedQKMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        mm_mgmt_.DoQkMatmul(midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                            computeTaskInfo_[curTaskId].kOrVOffset, computeTaskInfo_[curTaskId].rowLine,
                            computeTaskInfo_[curTaskId].GetColLine());
    }

    __aicore__ inline void DoJaggedGVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        mm_mgmt_.DoGvMatmul(midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                            computeTaskInfo_[curTaskId].kOrVOffset, computeTaskInfo_[curTaskId].rowLine,
                            computeTaskInfo_[curTaskId].GetColLine());
    }

    __aicore__ inline void DoJaggedQGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;

        int64_t qGradOutOffset =
            seqOffsetsGt_.GetValue(computeTaskInfo_[curTaskId].GetBatchId()) * headNum * headDim +
            computeTaskInfo_[curTaskId].GetHeadId() * computeTaskInfo_[curTaskId].GetCurSeqLen() * headDim +
            computeTaskInfo_[curTaskId].rowId * blockHeightQ * headDim;

        mm_mgmt_.DoQGradMatmul(qGradOutOffset, midScoreOffset, computeTaskInfo_[curTaskId].kOrVOffset,
                               computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine());
    }

    __aicore__ inline void DoJaggedKGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (computeTaskInfo_[curTaskId].taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        int64_t kAccumOffset = (computeTaskInfo_[curTaskId].GetAccumId() % MID_USE_TIMES) * blockHeightK * headDim;
        bool isFirstBlock = blockMaskParams_[curTaskId].IsFirstBlockNeedOverride();
        mm_mgmt_.DoKGradMatmul(kAccumOffset, midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                               computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine(),
                               isFirstBlock);
    }

    __aicore__ inline void DoJaggedVGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midScoreOffset = (computeTaskInfo_[curTaskId].taskId % COMPUTE_PIPE_NUM) * blockHeightQ * blockHeightK;
        int64_t vAccumOffset = (computeTaskInfo_[curTaskId].GetAccumId() % MID_USE_TIMES) * blockHeightK * headDim;
        bool isFirstBlock = blockMaskParams_[curTaskId].IsFirstBlockNeedOverride();
        mm_mgmt_.DoVGradMatmul(vAccumOffset, midScoreOffset, computeTaskInfo_[curTaskId].qOrGoffset,
                               computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine(),
                               isFirstBlock);
    }

    __aicore__ inline void VecScoreJagged(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t attnBiasOffset =
            computeTaskInfo_[curTaskId].GetBatchId() * headNum * biasGradSeqLen_ * biasGradSeqLen_ +
            computeTaskInfo_[curTaskId].GetHeadId() * biasGradSeqLen_ * biasGradSeqLen_ +
            computeTaskInfo_[curTaskId].rowId * blockHeightQ * biasGradSeqLen_ +
            computeTaskInfo_[curTaskId].GetColId() * blockHeightQ;
        int64_t attnBiasDiagonalOffset =
            computeTaskInfo_[curTaskId].GetBatchId() * headNum * biasGradSeqLen_ * biasGradSeqLen_ +
            computeTaskInfo_[curTaskId].GetHeadId() * biasGradSeqLen_ * biasGradSeqLen_ +
            computeTaskInfo_[curTaskId].GetColId() * blockHeightQ * biasGradSeqLen_ +
            computeTaskInfo_[curTaskId].rowId * blockHeightQ;

        int64_t maskOffset = 0;

        bool useMask = false;
        useMask = blockMaskParams_[curTaskId].NeedMask();

        this->VecScoreImpl(taskId, attnBiasOffset, attnBiasDiagonalOffset, maskOffset,
                           computeTaskInfo_[curTaskId].rowLine, computeTaskInfo_[curTaskId].GetColLine(), useMask);
    }

    __aicore__ inline void CopyInPadding(LocalTensor<qType> dstTensor, GlobalTensor<qType> srcTensor, int64_t rowNum,
                                         int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (seqLen - colNum) * sizeof(qType);
        uint32_t dstStride = (blockHeightQ - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint8_t rightPadding = (blockHeightQ - colNum) % (DATA_ALIGN_BYTES / sizeof(qType));

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<qType> padParams{true, 0, rightPadding, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    __aicore__ inline void CopyOutPadding(GlobalTensor<qType> dstTensor, LocalTensor<qType> srcTensor, int64_t rowNum,
                                          int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (blockHeightQ - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint32_t dstStride = (seqLen - colNum) * sizeof(qType);

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void VecScoreImpl(int64_t taskId, int64_t attnBiasOffset, int64_t attnBiasDiagonalOffset,
                                        int64_t maskOffset, int64_t totalRowNum, int64_t totalColNum, bool useMask)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;

        int64_t total = blockHeightQ * blockHeightQ;
        int64_t remain = total;
        int64_t thisLen = vecOnceDataNum_;
        BlockMaskGenerator generator(&blockMaskParams_[curTaskId]);
        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t baseOffset = total - remain;

            int64_t startRowNum = baseOffset / blockHeightQ;
            int64_t thisRowNum = thisLen / blockHeightQ;
            int64_t validRowNum = totalRowNum - startRowNum;
            validRowNum = validRowNum > thisRowNum ? thisRowNum : validRowNum;
            validRowNum = validRowNum < 0 ? 0 : validRowNum;

            int64_t qkOffset = midResultIdx * blockHeightQ * blockHeightQ + baseOffset;
            int64_t curAttnBiasOffset = attnBiasOffset + startRowNum * biasGradSeqLen_;
            int64_t curBiasGradOutOffset = midResultIdx * blockHeightQ * blockHeightQ + startRowNum * blockHeightQ;
            int64_t curMaskOffset = 0;
            curMaskOffset = maskOffset + baseOffset;

            if (validRowNum > 0) {
                ValidVecScore(thisLen, validRowNum, totalColNum, qkOffset, curMaskOffset, curAttnBiasOffset,
                              curBiasGradOutOffset, useMask, generator, startRowNum);
            }

            remain = remain - thisLen;
        }
    }
    __aicore__ inline void CastQType2Float(LocalTensor<float> dstTensor, LocalTensor<qType> srcTensor,
                                           LocalTensor<qType> midTensor, int64_t len)
    {
        DataCopy<qType>(midTensor, srcTensor, len);
        Cast(dstTensor, midTensor, RoundMode::CAST_NONE, len);
    }

    __aicore__ inline void CastInputData(LocalTensor<float>& inputQK, LocalTensor<float>& inputGV,
                                         LocalTensor<float>& inputMask, LocalTensor<float>& inputBias, int64_t thisLen,
                                         bool useMask)
    {
        LocalTensor<qType> outputMidTemp = tbufMid_.Get<qType>();
        if (!std::is_same<qType, float>::value) {
            CastQType2Float(inputQK, inputQK.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            CastQType2Float(inputGV, inputGV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32(int64_t thisLen, bool useMask)
    {
        LocalTensor<float> inputQK = queueVecScoreQK_.template DeQue<float>();
        LocalTensor<float> inputGV = queueVecScoreGV_.template DeQue<float>();

        LocalTensor<float> midQK = tbufMidQk_.Get<float>();
        LocalTensor<float> midGV = tbufMidGV_.Get<float>();
        Cast(midQK, inputQK.template ReinterpretCast<qType>(), RoundMode::CAST_NONE, thisLen);
        Cast(midGV, inputGV.template ReinterpretCast<qType>(), RoundMode::CAST_NONE, thisLen);
        queueVecScoreGV_.FreeTensor(inputGV);
        queueVecScoreQK_.FreeTensor(inputQK);
        LocalTensor<float> inputMask =
            useMask ? queueVecScoreMask_.DeQue<float>() : queueVecScoreMask_.AllocTensor<float>();
        LocalTensor<float> inputBias = queueVecScoreBias_.AllocTensor<float>();
        // CastInputData(inputQK, inputGV, inputMask, inputBias, thisLen, useMask);
        LocalTensor<float> dsiluTemp = tbufMid_.Get<float>();
        // 变量 inputQK inputGV inputMask inputBias
        // v = qk_input * alpha
        Muls<float>(midQK, midQK, alpha_, thisLen);
        // inputBias = sigmoid_fast(v);   sigmoid_v = sigmoid_fast(v);
        Sigmoid<float>(inputBias, midQK, thisLen);
        // inputBias = inputBias * inputMask; sigmoid_v = sigmoid_v * mask
        if (useMask) {
            Mul<float>(inputBias, inputBias, inputMask, thisLen);
        }
        // silu_out =  inputBias * inputQK  silu_out = v * sigmoid_v
        Mul<float>(inputMask, inputBias, midQK, thisLen);

        // dsilu_temp = sigmoid_v * (1 + v * (1 - sigmoid_v))
        // dsilu_temp = v - v*sigmoid_v
        Sub<float>(dsiluTemp, midQK, inputMask, thisLen);
        // dsilu_temp = 1 + dsilu_temp
        Adds<float>(dsiluTemp, dsiluTemp, 1, thisLen);
        // dsilu_temp = sigmoid_v * dsilu_temp
        Mul<float>(dsiluTemp, dsiluTemp, inputBias, thisLen);
        // scoreTemp = silu_out * silu_scale
        Muls(inputMask, inputMask, siluScale_, thisLen);

        // scoreGradTemp = gv_input * silu_scale * dsilu_temp * alpha
        Muls<float>(midGV, midGV, siluScale_ * alpha_, thisLen);
        Mul<float>(inputBias, midGV, dsiluTemp, thisLen);
        // Muls<float>(inputGV, inputGV, alpha_, thisLen);

        LocalTensor<qType> outputScore = queueOutputScore_.AllocTensor<qType>();
        LocalTensor<qType> outputBias = queueOutputBias_.AllocTensor<qType>();
        Cast(outputScore, inputMask, RoundMode::CAST_RINT, thisLen);
        Cast(outputBias, inputBias, RoundMode::CAST_RINT, thisLen);
        queueVecScoreMask_.FreeTensor(inputMask);
        queueVecScoreBias_.FreeTensor(inputBias);
        queueOutputScore_.EnQue(outputScore);
        queueOutputBias_.EnQue(outputBias);
    }

    __aicore__ inline void ValidVecScore(int64_t thisLen, int64_t validRowNum, int64_t totalColNum, int64_t qkOffset,
                                         int64_t curMaskOffset, int64_t curAttnBiasOffset, int64_t curBiasGradOutOffset,
                                         bool useMask, BlockMaskGenerator& generator, int64_t rowInBlock)
    {
        int64_t gvOffset = qkOffset;
        int64_t scoreTempOffset = qkOffset;
        LocalTensor<float> inputQK = queueVecScoreQK_.AllocTensor<float>();
        DataCopy<qType>(inputQK.template ReinterpretCast<qType>(), mm_mgmt_.qkTemp_[qkOffset], thisLen);
        queueVecScoreQK_.EnQue(inputQK);

        LocalTensor<float> inputGV = queueVecScoreGV_.AllocTensor<float>();
        DataCopy<qType>(inputGV.template ReinterpretCast<qType>(), mm_mgmt_.gvTemp_[gvOffset], thisLen);
        queueVecScoreGV_.EnQue(inputGV);
        if (useMask) {
            LocalTensor<float> inputMask = queueVecScoreMask_.AllocTensor<float>();
            generator.GenMask(inputMask, rowInBlock, thisLen / blockHeightQ, blockHeightQ);
            queueVecScoreMask_.EnQue(inputMask);
        }

        CalcuScoreWithFloat32(thisLen, useMask);

        LocalTensor<qType> outputScore = queueOutputScore_.DeQue<qType>();
        LocalTensor<qType> outputBias = queueOutputBias_.DeQue<qType>();
        DataCopy<qType>(mm_mgmt_.qkTemp_[scoreTempOffset], outputScore, thisLen);
        DataCopy<qType>(mm_mgmt_.gvTemp_[scoreTempOffset], outputBias, thisLen);

        queueOutputScore_.FreeTensor(outputScore);
        queueOutputBias_.FreeTensor(outputBias);
    }

    __aicore__ inline void DoTransJagged(int64_t taskId, GlobalTensor<float> from, GlobalTensor<qType> to,
                                         bool isCol = true)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = computeTaskInfo_[curTaskId].GetAccumId() % MID_USE_TIMES;
        int64_t fromOffset = midResultIdx * blockHeightQ * headDim;
        int64_t toOffset = 0;
        int64_t total = 0;
        if (isCol) {
            toOffset = computeTaskInfo_[curTaskId].kOrVOffset;
            total = computeTaskInfo_[curTaskId].GetColLine() * headDim;
        } else {
            toOffset = computeTaskInfo_[curTaskId].qOrGoffset;
            total = computeTaskInfo_[curTaskId].rowLine * headDim;
        }

        this->DoTransImpl(from, to, fromOffset, toOffset, total);
    }

    __aicore__ inline void DoTransImpl(GlobalTensor<float> from, GlobalTensor<qType> to, int64_t fromOffset,
                                       int64_t toOffset, int64_t total = 0)
    {
        int64_t remain = total;
        int64_t copyLenEachLoopAlignHeadDim = vecOnceDataNum_ / headDim * headDim;
        int64_t thisLen = copyLenEachLoopAlignHeadDim;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curFromOffset = total - remain;
            int64_t curToOffset = curFromOffset * headNum;

            LocalTensor<float> input = queueVecScoreQK_.AllocTensor<float>();
            DataCopy(input, from[fromOffset + curFromOffset], thisLen);
            queueVecScoreQK_.EnQue(input);

            LocalTensor<float> newInput = queueVecScoreQK_.DeQue<float>();
            LocalTensor<qType> output = queueOutputScore_.AllocTensor<qType>();
            if (std::is_same<qType, float>::value) {
                DataCopy(output.template ReinterpretCast<float>(), newInput, thisLen);
            } else {
                Cast(output, newInput, RoundMode::CAST_RINT, thisLen);
            }
            queueOutputScore_.EnQue(output);
            queueVecScoreQK_.FreeTensor(newInput);

            LocalTensor<qType> newOutput = queueOutputScore_.DeQue<qType>();
            uint16_t blockCount = thisLen / headDim;
            uint16_t blockLen = headDim * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t dstStride = (headNum * headDim - headDim) * sizeof(qType) / DATA_ALIGN_BYTES;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};
            DataCopy(to[toOffset + curToOffset], newOutput, copyParams);
            queueOutputScore_.FreeTensor(newOutput);

            remain = remain - thisLen;
        }
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
        mm_mgmt_.QkOrGvMatmulWait();
        mm_mgmt_.QkOrGvMatmulWait();
        if (taskId > 1) {
            mm_mgmt_.vGradMatmulWait();
            mm_mgmt_.kGradMatmulWait();
            mm_mgmt_.qGradMatmulWait();
            if (computeTaskInfo_[(taskId - TWO) % COMPUTE_PIPE_NUM].GetAccumId() !=
                computeTaskInfo_[(taskId - 1) % COMPUTE_PIPE_NUM].GetAccumId()) {
                DoTransJagged(taskId - TWO, mm_mgmt_.vGradAccumTemp_, mm_mgmt_.vGrad_);
                DoTransJagged(taskId - TWO, mm_mgmt_.kGradAccumTemp_, mm_mgmt_.kGrad_);
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
            mm_mgmt_.vGradMatmulWait();
            mm_mgmt_.kGradMatmulWait();
            mm_mgmt_.qGradMatmulWait();
            if (computeTaskInfo_[(taskId - TWO) % COMPUTE_PIPE_NUM].GetAccumId() !=
                computeTaskInfo_[(taskId - 1) % COMPUTE_PIPE_NUM].GetAccumId()) {
                DoTransJagged(taskId - TWO, mm_mgmt_.vGradAccumTemp_, mm_mgmt_.vGrad_);
                DoTransJagged(taskId - TWO, mm_mgmt_.kGradAccumTemp_, mm_mgmt_.kGrad_);
            }

            DoJaggedVGradMatmul(taskId - 1);
            DoJaggedKGradMatmul(taskId - 1);
            DoJaggedQGradMatmul(taskId - 1);

            mm_mgmt_.vGradMatmulWait();
            mm_mgmt_.kGradMatmulWait();
            mm_mgmt_.qGradMatmulWait();
            DoTransJagged(taskId - 1, mm_mgmt_.vGradAccumTemp_, mm_mgmt_.vGrad_);
            DoTransJagged(taskId - 1, mm_mgmt_.kGradAccumTemp_, mm_mgmt_.kGrad_);
        }

        if (taskId == 1) {
            VecScoreJagged(taskId - 1);
            DoJaggedVGradMatmul(taskId - 1);
            DoJaggedKGradMatmul(taskId - 1);
            DoJaggedQGradMatmul(taskId - 1);
            mm_mgmt_.vGradMatmulWait();
            mm_mgmt_.kGradMatmulWait();
            mm_mgmt_.qGradMatmulWait();
            DoTransJagged(taskId - 1, mm_mgmt_.vGradAccumTemp_, mm_mgmt_.vGrad_);
            DoTransJagged(taskId - 1, mm_mgmt_.kGradAccumTemp_, mm_mgmt_.kGrad_);
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
                JaggedFp8R0TaskInfo& args = this->computeTaskInfo_[taskId % COMPUTE_PIPE_NUM];
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
                if (maskinfo.NoComputation()) {
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

    __aicore__ inline void DoCopyQGrad(GlobalTensor<oType>& seqOffsets)
    {
        int64_t batchIdx = GetBlockIdx();
        int64_t taskNum = batchSize_ * headNum;
        int64_t coreTask = taskNum / aivNum_;
        int64_t coreSplitId = taskNum % aivNum_;

        int64_t taskNumOfThisCore = 0;
        int64_t offsetOfThisCore = 0;
        if (batchIdx >= coreSplitId) {
            taskNumOfThisCore = coreTask;
            offsetOfThisCore = coreSplitId * (coreTask + 1) + (batchIdx - coreSplitId) * coreTask;
        } else {
            taskNumOfThisCore = coreTask + 1;
            offsetOfThisCore = batchIdx * (coreTask + 1);
        }

        for (int64_t taskId = 0; taskId < taskNumOfThisCore; taskId++) {
            int64_t thisBatchIdx = (offsetOfThisCore + taskId) / headNum;
            int64_t headIdx = (offsetOfThisCore + taskId) % headNum;

            int64_t curSeqLen =
                static_cast<int64_t>(seqOffsets.GetValue(thisBatchIdx + 1) - seqOffsets.GetValue(thisBatchIdx));
            DoCopyBlockQGrad(thisBatchIdx, headIdx, curSeqLen, seqOffsets);
        }
    }

    __aicore__ inline void DoCopyBlockQGrad(int64_t thisBatchIdx, int64_t headIdx, int64_t curSeqLen,
                                            GlobalTensor<oType>& seqOffsets)
    {
        int64_t totalLen = curSeqLen * headDim;
        int64_t remain = totalLen;
        int64_t copyLenEachLoopAlignHeadDim = vecOnceDataNum_ / headDim * headDim;
        int64_t thisLen = copyLenEachLoopAlignHeadDim;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curOffset =
                (headNum * seqOffsets.GetValue(thisBatchIdx) * headDim) + (headIdx * totalLen) + (totalLen - remain);
            LocalTensor<float> input = queueVecScoreQK_.AllocTensor<float>();
            DataCopy<float>(input, mm_mgmt_.qGradAccumTemp_[curOffset], thisLen);
            queueVecScoreQK_.EnQue(input);

            LocalTensor<float> newInput = queueVecScoreQK_.DeQue<float>();
            LocalTensor<qType> output = queueOutputScore_.AllocTensor<qType>();
            if (std::is_same<qType, float>::value) {
                DataCopy(output.template ReinterpretCast<float>(), newInput, thisLen);
            } else {
                Cast(output, newInput, RoundMode::CAST_RINT, thisLen);
            }
            queueOutputScore_.EnQue(output);
            queueVecScoreQK_.FreeTensor(newInput);

            LocalTensor<qType> newOutput = queueOutputScore_.DeQue<qType>();

            uint16_t blockCount = thisLen / headDim;
            uint16_t blockLen = headDim * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t dstStride = (headNum - 1) * headDim * sizeof(qType) / DATA_ALIGN_BYTES;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};

            int64_t curOutOffset = seqOffsetsGt_.GetValue(thisBatchIdx) * headNum * headDim + headIdx * headDim +
                                   (totalLen - remain) * headNum;
            DataCopy<qType>(mm_mgmt_.qGrad_[curOutOffset], newOutput, copyParams);
            queueOutputScore_.FreeTensor(newOutput);

            remain = remain - thisLen;
        }
    }

    __aicore__ inline void CopyQGradToOutput()
    {
        SyncAll();
        this->DoCopyQGrad(seqOffsetsGt_);
    }
    // targetMask
    bool enableTargetMask_ = false;   // 初始化在Init中: enableTargetMask_ = backwardTilingData_->enableTargetMask;
    bool enableContextMask_ = false;  // 初始化在Init中: enableContextMask_ = backwardTilingData_->enableContextMask;

    // Shape
    int64_t batchSize_ = 0;       // 初始化在Init中: batchSize_ = backwardTilingData_->batchSize;
    int64_t maxSeqLen_ = 0;       // 初始化在Init中: maxSeqLen_ = backwardTilingData_->maxSeqLen;
    int64_t biasGradSeqLen_ = 0;  // 初始化在Init中: biasGradSeqLen_ = backwardTilingData_->biasGradSeqLen;

    // Attr
    float siluScale_ = 1.0f;  // 初始化在Init中: siluScale_ = backwardTilingData_->siluScale;
    uint32_t aivNum_ = 1;     // 没有在Init看到明确赋值（通常由调度脚本/tiling时传入）

    // task
    BlockInfo taskInfo_[COMPUTE_PIPE_NUM];  // 没有看到构造初始化（结构体数组默认构造）

    // MaskType
    int64_t targetGroupSize_ = 0;  // 初始化在Init中: targetGroupSize_ = backwardTilingData_->targetGroupSize;
    float alpha_ = 1.0f;           // 初始化在Init中: alpha_ = backwardTilingData_->alpha;
    BlockMaskParams blockMaskParams_[COMPUTE_PIPE_NUM] = {};  // 局部赋值, 结构体数组，默认初始化

    // Tpipe
    TPipe pipe;  // pipe.InitBuffer等初始化

    // vec score
    int64_t vecOnceDataNum_ = 0;                   // Init中初始化: vecOnceDataNum_ = ...
    TQue<TPosition::VECIN, 1> queueVecScoreQK_;    // Init中 pipe.InitBuffer
    TQue<TPosition::VECIN, 1> queueVecScoreGV_;    // Init中 pipe.InitBuffer
    TQue<TPosition::VECIN, 1> queueVecScoreMask_;  // Init中 pipe.InitBuffer
    TQue<TPosition::VECIN, 1> queueVecScoreBias_;  // Init中 pipe.InitBuffer

    TQue<TPosition::VECOUT, 1> queueOutputScore_;  // Init中 pipe.InitBuffer
    TQue<TPosition::VECOUT, 1> queueOutputBias_;   // Init中 pipe.InitBuffer
    TBuf<TPosition::VECCALC> tbufMid_;             // Init中 pipe.InitBuffer
    TBuf<TPosition::VECCALC> tbufMidQk_;           // Init中 pipe.InitBuffer
    TBuf<TPosition::VECCALC> tbufMidGV_;           // Init中 pipe.InitBuffer

    // Gt
    GlobalTensor<int64_t> numContextGt_;  // PreInit中SetGlobalBuffer赋指针和长度
    GlobalTensor<int64_t> numTargetGt_;   // PreInit中SetGlobalBuffer赋指针和长度

    // Matmul
    MmMgmtFp16R0Jagged<qType, blockHeightQ, blockHeightK, headNum, headDim> mm_mgmt_;  // Init中Init方法传入指针和shape

protected:
    uint32_t startColBlock_ = 0;                                  // PreInit中startColBlock_ = colBlock[0];
    uint32_t endColBlock_ = 0;                                    // PreInit中endColBlock_ = colBlock[1];
    uint32_t startRowBlock_ = 0;                                  // PreInit中startRowBlock_ = rowBlock[0];
    uint32_t endRowBlock_ = 0;                                    // PreInit中endRowBlock_ = rowBlock[1];
    JaggedFp8R0TaskInfo computeTaskInfo_[COMPUTE_PIPE_NUM] = {};  // 局部循环/函数内赋值, 默认初始化
    const HstuDenseBackwardTilingData* __restrict backwardTilingData_{
        nullptr};                       // Compute()赋值: backwardTilingData_ = args.tilingDataPtr
    GlobalTensor<oType> seqOffsetsGt_;  // PreInit中SetGlobalBuffer
};
}  // namespace HstuDenseBackward
#endif