/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef HSTU_DENSE_BACKWARD_KERNEL_H
#define HSTU_DENSE_BACKWARD_KERNEL_H

#include "hstu_dense_backward_kernel_common.h"

namespace HstuDenseBackward {

template <typename qType>
class HstuDenseBackwardKernel : public HstuDenseBackwardKernelInterface<qType> {
public:
    __aicore__ inline HstuDenseBackwardKernel() {}

    __aicore__ inline void Compute(Args &args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        REGIST_MATMUL_OBJ(&this->pipe, GetSysWorkSpacePtr(), this->qkMatmul, &tilingData.qkMatmul, this->qGradMatmul,
                          &tilingData.qGradMatmul, this->kGradMatmul, &tilingData.kGradMatmul, this->vGradMatmul,
                          &tilingData.vGradMatmul);
        uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
        this->qkMatmul.SetUserDefInfo(tilingPtr);
        this->qGradMatmul.SetUserDefInfo(tilingPtr);
        this->kGradMatmul.SetUserDefInfo(tilingPtr);
        this->vGradMatmul.SetUserDefInfo(tilingPtr);

        this->Init(args);
        this->ComputeFirst();
        this->ComputeSecond();
    }

    __aicore__ inline void CalcBaseOffsets(int64_t curTaskId, bool isCol = true)
    {
        this->taskInfo[curTaskId].qkLeftOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum * this->headDim +
                                           this->taskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
                                           this->taskInfo[curTaskId].headId * this->headDim;
        this->taskInfo[curTaskId].qkRightOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum * this->headDim +
                                            this->taskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
                                            this->taskInfo[curTaskId].headId * this->headDim;
        this->taskInfo[curTaskId].kGradLeftOffset = this->taskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
                                              this->taskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
                                              this->taskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
                                              this->taskInfo[curTaskId].colId * this->blockHeight;
        if (isCol) {
            this->taskInfo[curTaskId].vGradRightOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum * this->headDim +
                                                   this->taskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
                                                   this->taskInfo[curTaskId].headId * this->headDim;

            this->taskInfo[curTaskId].rowLine = this->seqLen - this->taskInfo[curTaskId].rowId * this->blockHeight;
            if (this->taskInfo[curTaskId].rowLine > this->blockHeight) {
                this->taskInfo[curTaskId].rowLine = this->blockHeight;
            }
        } else {
            this->taskInfo[curTaskId].vGradRightOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum * this->headDim +
                                                   this->taskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
                                                   this->taskInfo[curTaskId].headId * this->headDim;

            this->taskInfo[curTaskId].colLine = this->seqLen - this->taskInfo[curTaskId].colId * this->blockHeight;
            if (this->taskInfo[curTaskId].colLine > this->blockHeight) {
                this->taskInfo[curTaskId].colLine = this->blockHeight;
            }
        }
    }

    __aicore__ inline void DoQKMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;

        this->qkMatmul.SetTail(this->taskInfo[curTaskId].rowLine, this->taskInfo[curTaskId].colLine, this->headDim);
        DoQKMatmulImpl(this->taskInfo[curTaskId].qkLeftOffset, this->taskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoQKMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        this->qkMatmul.SetTensorA(this->q[left]);
        this->qkMatmul.SetTensorB(this->k[right], true);

        this->qkMatmul.template IterateAll<false>(this->qkTemp[out], 0, false, true);
    }

    __aicore__ inline void DoGVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * this->blockHeight * this->blockHeight;

        this->qkMatmul.SetTail(this->taskInfo[curTaskId].rowLine, this->taskInfo[curTaskId].colLine, this->headDim);
        DoGVMatmulImpl(this->taskInfo[curTaskId].qkLeftOffset, this->taskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoGVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        this->qkMatmul.SetTensorA(this->grad[left]);
        this->qkMatmul.SetTensorB(this->v[right], true);

        this->qkMatmul.template IterateAll<false>(this->gvTemp[out], 0, false, true);
    }

    __aicore__ inline void DoQGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = this->taskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        bool isNew = this->taskInfo[curTaskId].colId == 0;

        this->qGradMatmul.SetTail(this->taskInfo[curTaskId].rowLine, this->headDim, this->taskInfo[curTaskId].colLine);
        DoQGradMatmulImpl(this->taskInfo[curTaskId].kGradLeftOffset, this->taskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoQGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        this->qGradMatmul.SetTensorA(this->attnBiasGrad[left]);
        this->qGradMatmul.SetTensorB(this->k[right]);
        if (isNew) {
            this->qGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 0, false, true);
        } else {
            this->qGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoKGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = this->taskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        bool isNew = false;
        if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
            isNew = this->taskInfo[curTaskId].rowId == this->taskInfo[curTaskId].colId;
        } else {
            isNew = this->taskInfo[curTaskId].rowId == 0;
        }

        this->kGradMatmul.SetTail(this->taskInfo[curTaskId].colLine, this->headDim, this->taskInfo[curTaskId].rowLine);
        DoKGradMatmulImpl(this->taskInfo[curTaskId].kGradLeftOffset, this->taskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoKGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        this->kGradMatmul.SetTensorA(this->attnBiasGrad[left], true);
        this->kGradMatmul.SetTensorB(this->q[right]);
        if (isNew) {
            this->kGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 0, false, true);
        } else {
            this->kGradMatmul.template IterateAll<false>(this->kGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoVGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t midAccumIdx = this->taskInfo[curTaskId].accumId % MID_USE_TIMES;

        int64_t scoreTempOffset = midResultIdx * this->blockHeight * this->blockHeight;
        int64_t outOffset = midAccumIdx * this->blockHeight * this->headDim;

        bool isNew = false;
        if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
            isNew = this->taskInfo[curTaskId].rowId == this->taskInfo[curTaskId].colId;
        } else {
            isNew = this->taskInfo[curTaskId].rowId == 0;
        }

        this->vGradMatmul.SetTail(this->taskInfo[curTaskId].colLine, this->headDim, this->taskInfo[curTaskId].rowLine);
        DoVGradMatmulImpl(scoreTempOffset, this->taskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoVGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        this->vGradMatmul.SetTensorA(this->scoreTemp[left], true);
        this->vGradMatmul.SetTensorB(this->grad[right]);
        if (isNew) {
            this->vGradMatmul.template IterateAll<false>(this->vGradAccumTemp[out], 0, false, true);
        } else {
            this->vGradMatmul.template IterateAll<false>(this->vGradAccumTemp[out], 1, false, true);
        }
    }



    __aicore__ inline void CastQType2Float(LocalTensor<float> dstTensor, LocalTensor<qType> srcTensor,
                                           LocalTensor<qType> midTensor, int64_t len)
    {
        DataCopy<qType>(midTensor, srcTensor, len);
        Cast(dstTensor, midTensor, RoundMode::CAST_NONE, len);
    }

    __aicore__ inline void CastInputData(LocalTensor<float> &inputQK, LocalTensor<float> &inputGV,
                                         LocalTensor<float> &inputMask, LocalTensor<float> &inputBias, int64_t thisLen,
                                         bool useMask)
    {
        LocalTensor<qType> outputMidTemp = this->queueOutputTemp.template AllocTensor<qType>();
        if (!std::is_same<qType, float>::value) {
            CastQType2Float(inputQK, inputQK.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            CastQType2Float(inputGV, inputGV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            if (useMask) {
                CastQType2Float(inputMask, inputMask.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
            if (this->enableBias) {
                CastQType2Float(inputBias, inputBias.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
        }
        this->queueOutputTemp.template FreeTensor(outputMidTemp);
    }

    __aicore__ inline void CalcuScoreWithFloat32(int64_t thisLen, bool useMask)
    {
        auto inputQK = this->queueVecScoreQK.template DeQue<float>();
        auto inputGV = this->queueVecScoreGV.template DeQue<float>();
        LocalTensor<float> inputMask = useMask ? this->queueVecScoreMask.template DeQue<float>() :
                                                 this->queueVecScoreMask.template AllocTensor<float>();
        LocalTensor<float> inputBias = this->enableBias ? this->queueVecScoreBias.template DeQue<float>() :
                                                    this->queueVecScoreBias.template AllocTensor<float>();

        CastInputData(inputQK, inputGV, inputMask, inputBias, thisLen, useMask);

        if (this->enableBias) {
            // qkb = qk + attn_bias
            Add<float>(inputQK, inputQK, inputBias, thisLen);
        }

        // score = F.silu(qkb) * this->siluScale * this->mask
        Silu<float>(inputBias, inputQK, thisLen);
        Muls<float>(inputBias, inputBias, this->siluScale, thisLen);
        if (useMask) {
            Mul<float>(inputBias, inputBias, inputMask, thisLen);
        }

        // score_grad = gv * this->siluScale * this->mask
        Muls<float>(inputGV, inputGV, this->siluScale, thisLen);
        if (useMask) {
            Mul<float>(inputGV, inputGV, inputMask, thisLen);
        }

        // bias_grad = (F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))) * score_grad
        //  F.sigmoid(qkb)
        LocalTensor<uint8_t> sigmoidBuffer = this->queueOutputTemp.template AllocTensor<uint8_t>();
        Sigmoid<float>(inputMask, inputQK, sigmoidBuffer, thisLen);
        //  qkb * F.sigmoid(qkb)
        LocalTensor<float> tmpBuffer = sigmoidBuffer.template ReinterpretCast<float>();
        Mul<float>(tmpBuffer, inputQK, inputMask, thisLen);
        //  qkb * (1 - F.sigmoid(qkb)) = qkb - qkb * F.sigmoid(qkb)
        Sub<float>(tmpBuffer, inputQK, tmpBuffer, thisLen);
        //  1 + qkb * (1 - F.sigmoid(qkb))
        Adds<float>(tmpBuffer, tmpBuffer, 1, thisLen);
        //  F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
        Mul<float>(inputQK, inputMask, tmpBuffer, thisLen);

        this->queueVecScoreMask.template FreeTensor(inputMask);
        this->queueOutputTemp.template FreeTensor(sigmoidBuffer);

        //  (F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))) * score_grad
        Mul<float>(inputQK, inputQK, inputGV, thisLen);
        this->queueVecScoreGV.template FreeTensor(inputGV);

        LocalTensor<qType> outputScore = this->queueOutputScore.template AllocTensor<qType>();
        LocalTensor<qType> outputBias = this->queueOutputBias.template AllocTensor<qType>();
        if (!std::is_same<qType, float>::value) {
            Cast(outputScore, inputBias, RoundMode::CAST_RINT, thisLen);
            Cast(outputBias, inputQK, RoundMode::CAST_RINT, thisLen);
        } else {
            LocalTensor<float> newOutputScore = outputScore.template ReinterpretCast<float>();
            DataCopy(newOutputScore, inputBias, thisLen);

            LocalTensor<float> newOutputBias = outputBias.template ReinterpretCast<float>();
            DataCopy(newOutputBias, inputQK, thisLen);
        }
        this->queueVecScoreQK.template FreeTensor(inputQK);
        this->queueVecScoreBias.template FreeTensor(inputBias);

        this->queueOutputScore.template EnQue(outputScore);
        this->queueOutputBias.template EnQue(outputBias);
    }

    __aicore__ inline void VecScore(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t attnBiasOffset = this->taskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
                                 this->taskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
                                 this->taskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
                                 this->taskInfo[curTaskId].colId * this->blockHeight;
        int64_t attnBiasDiagonalOffset = this->taskInfo[curTaskId].batchId * this->headNum * this->biasGradSeqLen * this->biasGradSeqLen +
                                         this->taskInfo[curTaskId].headId * this->biasGradSeqLen * this->biasGradSeqLen +
                                         this->taskInfo[curTaskId].colId * this->blockHeight * this->biasGradSeqLen +
                                         this->taskInfo[curTaskId].rowId * this->blockHeight;

        int64_t maskOffset = 0;
        if (IfMask(this->maskType, MaskType::MASK_CUSTOM)) {
            maskOffset = this->taskInfo[curTaskId].batchId * this->headNum * this->maxSeqLen * this->maxSeqLen +
                         this->taskInfo[curTaskId].headId * this->maxSeqLen * this->maxSeqLen +
                         this->taskInfo[curTaskId].rowId * this->blockHeight * this->maxSeqLen + this->taskInfo[curTaskId].colId * this->blockHeight;
        }

        bool useMask = false;
        if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
            useMask = this->taskInfo[curTaskId].rowId == this->taskInfo[curTaskId].colId;
        } else if (IfMask(this->maskType, MaskType::MASK_CUSTOM)) {
            useMask = true;
        }

        VecScoreImpl(taskId, attnBiasOffset, attnBiasDiagonalOffset, maskOffset, this->taskInfo[curTaskId].rowLine,
                     this->taskInfo[curTaskId].colLine, useMask);
    }

    __aicore__ inline void CopyInPadding(LocalTensor<qType> dstTensor, GlobalTensor<qType> srcTensor, int64_t rowNum,
                                         int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (seqLen - colNum) * sizeof(qType);
        uint32_t dstStride = (this->blockHeight - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint8_t rightPadding = (this->blockHeight - colNum) % (DATA_ALIGN_BYTES / sizeof(qType));

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<qType> padParams{true, 0, rightPadding, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    __aicore__ inline void CopyOutPadding(GlobalTensor<qType> dstTensor, LocalTensor<qType> srcTensor, int64_t rowNum,
                                          int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (this->blockHeight - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint32_t dstStride = (seqLen - colNum) * sizeof(qType);

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void ValidVecScore(int64_t thisLen, int64_t validRowNum, int64_t totalColNum, int64_t qkOffset,
                                         int64_t curMaskOffset, int64_t curAttnBiasOffset, bool useMask)
    {
        int64_t gvOffset = qkOffset;
        int64_t scoreTempOffset = qkOffset;
        LocalTensor<float> inputQK = this->queueVecScoreQK.template AllocTensor<float>();
        DataCopy<qType>(inputQK.template ReinterpretCast<qType>(), this->qkTemp[qkOffset], thisLen);
        this->queueVecScoreQK.template EnQue(inputQK);

        LocalTensor<float> inputGV = this->queueVecScoreGV.template AllocTensor<float>();
        DataCopy<qType>(inputGV.template ReinterpretCast<qType>(), this->gvTemp[gvOffset], thisLen);
        this->queueVecScoreGV.template EnQue(inputGV);
        if (useMask) {
            LocalTensor<float> inputMask = this->queueVecScoreMask.template AllocTensor<float>();
            if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
                DataCopy<qType>(inputMask.template ReinterpretCast<qType>(), this->maskTemp[curMaskOffset], thisLen);
            }
            if (IfMask(this->maskType, MaskType::MASK_CUSTOM)) {
                CopyInPadding(inputMask.template ReinterpretCast<qType>(), this->mask[curMaskOffset], validRowNum,
                              totalColNum, this->maxSeqLen);
            }
            this->queueVecScoreMask.template EnQue(inputMask);
        }
        if (this->enableBias) {
            LocalTensor<float> inputBias = this->queueVecScoreBias.template AllocTensor<float>();
            CopyInPadding(inputBias.template ReinterpretCast<qType>(), this->attnBias[curAttnBiasOffset], validRowNum,
                          totalColNum, this->biasGradSeqLen);
            this->queueVecScoreBias.template EnQue(inputBias);
        }

        CalcuScoreWithFloat32(thisLen, useMask);

        LocalTensor<qType> outputScore = this->queueOutputScore.template DeQue<qType>();
        LocalTensor<qType> outputBias = this->queueOutputBias.template DeQue<qType>();
        DataCopy<qType>(this->scoreTemp[scoreTempOffset], outputScore, thisLen);
        CopyOutPadding(this->attnBiasGrad[curAttnBiasOffset], outputBias, validRowNum, totalColNum, this->biasGradSeqLen);
        this->queueOutputScore.template FreeTensor(outputScore);
        this->queueOutputBias.template FreeTensor(outputBias);
    }

    __aicore__ inline void VecScoreImpl(int64_t taskId, int64_t attnBiasOffset, int64_t attnBiasDiagonalOffset,
                                        int64_t maskOffset, int64_t totalRowNum, int64_t totalColNum, bool useMask)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;

        int64_t total = this->blockHeight * this->blockHeight;
        int64_t remain = total;
        int64_t thisLen = this->vecOnceDataNum;
        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t baseOffset = total - remain;

            int64_t startRowNum = baseOffset / this->blockHeight;
            int64_t thisRowNum = thisLen / this->blockHeight;
            int64_t validRowNum = totalRowNum - startRowNum;
            validRowNum = validRowNum > thisRowNum ? thisRowNum : validRowNum;
            validRowNum = validRowNum < 0 ? 0 : validRowNum;

            int64_t qkOffset = midResultIdx * this->blockHeight * this->blockHeight + baseOffset;
            int64_t curAttnBiasOffset = attnBiasOffset + startRowNum * this->biasGradSeqLen;
            int64_t curMaskOffset = 0;
            if (IfMask(this->maskType, MaskType::MASK_TRIL)) {
                curMaskOffset = maskOffset + baseOffset;
            } else if (IfMask(this->maskType, MaskType::MASK_CUSTOM)) {
                curMaskOffset = maskOffset + startRowNum * this->maxSeqLen;
            }

            if (validRowNum > 0) {
                ValidVecScore(thisLen, validRowNum, totalColNum, qkOffset, curMaskOffset, curAttnBiasOffset, useMask);
            }

            if (this->enableBias && IfMask(this->maskType, MaskType::MASK_TRIL) && !useMask) {
                LocalTensor<qType> outputTempTensor = this->queueOutputTemp.template AllocTensor<qType>();
                Duplicate<qType>(outputTempTensor, 0, thisLen);
                this->queueOutputTemp.template EnQue(outputTempTensor);

                int64_t curAttnBiasDiagonalOffset = attnBiasDiagonalOffset + startRowNum * this->biasGradSeqLen;
                outputTempTensor = this->queueOutputTemp.template DeQue<qType>();
                CopyOutPadding(this->attnBiasGrad[curAttnBiasDiagonalOffset], outputTempTensor, thisRowNum, totalRowNum,
                               this->biasGradSeqLen);
                this->queueOutputTemp.template FreeTensor(outputTempTensor);
            }

            remain = remain - thisLen;
        }
    }

    __aicore__ inline void DoTrans(int64_t taskId, GlobalTensor<float> from, GlobalTensor<qType> to, bool isCol = true)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = this->taskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t fromOffset = midResultIdx * this->blockHeight * this->headDim;
        int64_t toOffset = 0;
        int64_t total = 0;
        if (isCol) {
            toOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum * this->headDim +
                       this->taskInfo[curTaskId].colId * this->blockHeight * this->headNum * this->headDim +
                       this->taskInfo[curTaskId].headId * this->headDim;
            total = this->taskInfo[curTaskId].colLine * this->headDim;
        } else {
            toOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum * this->headDim +
                       this->taskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
                       this->taskInfo[curTaskId].headId * this->headDim;
            total = this->taskInfo[curTaskId].rowLine * this->headDim;
        }

        DoTransImpl(from, to, fromOffset, toOffset, total);
    }

    __aicore__ inline void DoTransImpl(GlobalTensor<float> from, GlobalTensor<qType> to, int64_t fromOffset,
                                       int64_t toOffset, int64_t total = 0)
    {
        int64_t remain = total;
        int64_t thisLen = this->vecOnceDataNum;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curFromOffset = total - remain;
            int64_t curToOffset = curFromOffset * this->headNum;

            LocalTensor<float> input = this->queueVecScoreQK.template AllocTensor<float>();
            DataCopy(input, from[fromOffset + curFromOffset], thisLen);
            this->queueVecScoreQK.template EnQue(input);

            LocalTensor<float> newInput = this->queueVecScoreQK.template DeQue<float>();
            LocalTensor<qType> output = this->queueOutputTemp.template AllocTensor<qType>();
            if (std::is_same<qType, float>::value) {
                DataCopy(output.template ReinterpretCast<float>(), newInput, thisLen);
            } else {
                Cast(output, newInput, RoundMode::CAST_RINT, thisLen);
            }
            this->queueOutputTemp.template EnQue(output);
            this->queueVecScoreQK.template FreeTensor(newInput);

            LocalTensor<qType> newOutput = this->queueOutputTemp.template DeQue<qType>();
            uint16_t blockCount = thisLen / this->headDim;
            uint16_t blockLen = this->headDim * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t dstStride = (this->headNum * this->headDim - this->headDim) * sizeof(qType) / DATA_ALIGN_BYTES;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};
            DataCopy(to[toOffset + curToOffset], newOutput, copyParams);
            this->queueOutputTemp.template FreeTensor(newOutput);

            remain = remain - thisLen;
        }
    }

    __aicore__ inline void FirstStagePipeline(int64_t taskId)
    {
        DoQKMatmul(taskId);
        DoGVMatmul(taskId);
        if (taskId > 1) {
            DoVGradMatmul(taskId - TWO);
            DoKGradMatmul(taskId - TWO);
        }
        if (taskId > 0) {
            VecScore(taskId - 1);
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
            if (this->taskInfo[(taskId - TWO) % COMPUTE_PIPE_NUM].accumId !=
                this->taskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId) {
                DoTrans(taskId - TWO, this->vGradAccumTemp, this->vGrad);
                DoTrans(taskId - TWO, this->kGradAccumTemp, this->kGrad);
            }
        }
    }

    __aicore__ inline void FirstStageEnding(int64_t taskId)
    {
        if (taskId > 1) {
            DoVGradMatmul(taskId - TWO);
            DoKGradMatmul(taskId - TWO);
            VecScore(taskId - 1);
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            if (this->taskInfo[(taskId - TWO) % COMPUTE_PIPE_NUM].accumId !=
                this->taskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId) {
                DoTrans(taskId - TWO, this->vGradAccumTemp, this->vGrad);
                DoTrans(taskId - TWO, this->kGradAccumTemp, this->kGrad);
            }

            DoVGradMatmul(taskId - 1);
            DoKGradMatmul(taskId - 1);
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            DoTrans(taskId - 1, this->vGradAccumTemp, this->vGrad);
            DoTrans(taskId - 1, this->kGradAccumTemp, this->kGrad);
        }

        if (taskId == 1) {
            VecScore(taskId - 1);

            DoVGradMatmul(taskId - 1);
            DoKGradMatmul(taskId - 1);
            this->vGradMatmul.WaitIterateAll();
            this->vGradMatmul.End();
            this->kGradMatmul.WaitIterateAll();
            this->kGradMatmul.End();
            DoTrans(taskId - 1, this->vGradAccumTemp, this->vGrad);
            DoTrans(taskId - 1, this->kGradAccumTemp, this->kGrad);
        }
    }

    __aicore__ inline void this->ComputeFirst()
    {
        int64_t taskId = 0;
        int64_t accumId = 0;

        int64_t totalAivNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        int64_t startId = GetBlockIdx();
        int64_t nextCol = totalAivNum * TWO - GetBlockIdx() * TWO - 1;

        for (int64_t gColId = startId; gColId < this->totalColBlockNum;) {
            int64_t batchId = gColId / (this->headNum * this->colBlockNum);
            int64_t colIdInBatch = gColId % (this->headNum * this->colBlockNum);
            int64_t headId = colIdInBatch / this->colBlockNum;
            int64_t colId = colIdInBatch % this->colBlockNum;
            int64_t colLine = this->seqLen - colId * this->blockHeight;
            colLine = colLine > this->blockHeight ? this->blockHeight : colLine;

            for (int64_t rowId = 0; rowId < this->rowBlockNum; rowId++) {
                if (IfMask(this->maskType, MaskType::MASK_TRIL) && rowId < colId) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                this->taskInfo[curTaskId] = BlockInfo{taskId, batchId, headId, rowId, colId, accumId};
                this->taskInfo[curTaskId].colLine = colLine;
                CalcBaseOffsets(curTaskId);

                FirstStagePipeline(taskId);

                taskId += 1;
            }
            accumId++;
            gColId += nextCol;
            nextCol = totalAivNum * TWO - nextCol;
        }

        FirstStageEnding(taskId);
    }

    __aicore__ inline void SecondStagePipeline(int64_t taskId)
    {
        DoQGradMatmul(taskId);
        if (taskId > 0) {
            if (this->taskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId != this->taskInfo[taskId % COMPUTE_PIPE_NUM].accumId) {
                DoTrans(taskId - 1, this->kGradAccumTemp, this->qGrad, 0);
            }
        }
        this->qGradMatmul.WaitIterateAll();
        this->qGradMatmul.End();
    }

    __aicore__ inline void this->ComputeSecond()
    {
        SyncAll();

        int64_t taskId = 0;
        int64_t accumId = 0;

        int64_t totalAivNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        int64_t startId = GetBlockIdx();
        int64_t nextRow = totalAivNum * TWO - GetBlockIdx() * TWO - 1;

        for (int64_t gRowId = startId; gRowId < this->totalRowBlockNum;) {
            int64_t batchId = gRowId / (this->headNum * this->rowBlockNum);
            int64_t rowIdInBatch = gRowId % (this->headNum * this->rowBlockNum);
            int64_t headId = rowIdInBatch / this->rowBlockNum;
            int64_t rowId = rowIdInBatch % this->rowBlockNum;
            int64_t rowLine = this->seqLen - rowId * this->blockHeight;
            rowLine = rowLine > this->blockHeight ? this->blockHeight : rowLine;

            for (int64_t colId = 0; colId < this->colBlockNum; colId++) {
                if (IfMask(this->maskType, MaskType::MASK_TRIL) && rowId < colId) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                this->taskInfo[curTaskId] = BlockInfo{taskId, batchId, headId, rowId, colId, accumId};
                this->taskInfo[curTaskId].rowLine = rowLine;
                CalcBaseOffsets(curTaskId, false);

                SecondStagePipeline(taskId);

                taskId++;
            }

            accumId++;
            gRowId += nextRow;
            nextRow = totalAivNum * TWO - nextRow;
        }

        if (taskId > 0) {
            DoTrans(taskId - 1, this->kGradAccumTemp, this->qGrad, 0);
        }
    }
};
}  // namespace HstuDenseBackward
#endif
