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

#ifndef HSTU_DENSE_BACKWARD_KERNEL_H
#define HSTU_DENSE_BACKWARD_KERNEL_H

#include "hstu_dense_backward_kernel_common.h"

namespace HstuDenseBackwardFuxi {

template <typename qType> class HstuDenseBackwardKernelFuxi {
public:
    __aicore__ inline HstuDenseBackwardKernelFuxi() {}

    __aicore__ inline void Init(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        GM_ADDR workspace = args.workspace;

        batchSize = tilingData.batchSize;
        seqLen = tilingData.seqLen;
        headNum = tilingData.headNum;
        headDim = tilingData.headDim;

        maxSeqLen = tilingData.maxSeqLen;
        biasGradSeqLen = tilingData.biasGradSeqLen;
        siluScale = tilingData.siluScale;

        blockHeight = tilingData.blockHeight;

        maskType = tilingData.maskType;
        enableBias = tilingData.enableBias;

        rowBlockNum = (seqLen + blockHeight - 1) / blockHeight;
        colBlockNum = (seqLen + blockHeight - 1) / blockHeight;
        totalRowBlockNum = batchSize * headNum * rowBlockNum;
        totalColBlockNum = batchSize * headNum * colBlockNum;
        totalBlockNum = totalRowBlockNum * colBlockNum;

        int64_t totalElementOfQ = batchSize * maxSeqLen * headNum * headDim;
        int64_t totalElementOfAttnBias = batchSize * headNum * biasGradSeqLen * biasGradSeqLen;
        int64_t totalElementOfBts = batchSize * biasGradSeqLen * biasGradSeqLen;

        grad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.grad), totalElementOfQ);
        q.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.q), totalElementOfQ);
        k.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.k), totalElementOfQ);
        v.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.v), totalElementOfQ);
        if (enableBias) {
            bts.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.bts), totalElementOfBts);
            bpos.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.bpos), totalElementOfBts);
        }
        if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
            mask.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.mask), totalElementOfAttnBias);
        }
        gradPosition.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.gradBposIn), totalElementOfQ);
        gradTimestamp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.gradBtsIn), totalElementOfQ);
        
        qGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.qGrad), totalElementOfQ);
        kGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.kGrad), totalElementOfQ);
        vGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.vGrad), totalElementOfQ);
        vbposGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.vbposGrad), totalElementOfQ);
        vbtsGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.vbtsGrad), totalElementOfQ);
        btsGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.btsGrad), totalElementOfQ);
        bposGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.bposGrad), totalElementOfQ);  // for b s s

        int64_t qkMatmulTempSpace = blockHeight * blockHeight;
        int64_t gvMatmulTempSpace = blockHeight * blockHeight;
        int64_t gpvMatmulTempSpace = blockHeight * blockHeight;
        int64_t gtVMatmulTempSpace = blockHeight * blockHeight;

        int64_t vGradAccumTempSpace = blockHeight * headDim;
        int64_t kGradAccumTempSpace = blockHeight * headDim;
        int64_t bposGradAccumTempSpace = blockHeight * headDim;
        int64_t btsGradAccumTempSpace = blockHeight * headDim;

        int64_t scoreTempSpace = blockHeight * blockHeight;
        int64_t bposGradTempSpace = blockHeight * blockHeight;
        int64_t btsGradTempSpace = blockHeight * blockHeight;

        int64_t maskTempSpace = blockHeight * blockHeight;
        int64_t attnBiasGradTempSpace = batchSize * headNum * maxSeqLen * maxSeqLen * sizeof(qType);

        int64_t totalTempSpaceForOneVec =
            MID_USE_TIMES *
                ((vGradAccumTempSpace + kGradAccumTempSpace +
                  bposGradAccumTempSpace + btsGradAccumTempSpace) * sizeof(float) +
                 (qkMatmulTempSpace + gvMatmulTempSpace + scoreTempSpace + bposGradTempSpace + btsGradTempSpace +
                  gpvMatmulTempSpace + gtVMatmulTempSpace) * sizeof(qType)) +
            maskTempSpace * sizeof(qType);

        attnBiasGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(workspace), totalElementOfAttnBias);

        curAICWorkspace = reinterpret_cast<__gm__ uint8_t *>(workspace) + attnBiasGradTempSpace +
            GetBlockIdx() * totalTempSpaceForOneVec;
        qkTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), qkMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += qkMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        gvTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), gvMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += gvMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        tempGposVT.SetGlobalBuffer(
            reinterpret_cast<__gm__ qType *>(curAICWorkspace), gpvMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += gpvMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        tempGtsVT.SetGlobalBuffer(
            reinterpret_cast<__gm__ qType *>(curAICWorkspace), gtVMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += gtVMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        scoreTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), scoreTempSpace * MID_USE_TIMES);
        curAICWorkspace += scoreTempSpace * sizeof(qType) * MID_USE_TIMES;

        tempBposM.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), bposGradTempSpace * MID_USE_TIMES);
        curAICWorkspace += bposGradTempSpace * sizeof(qType) * MID_USE_TIMES;

        tempBtsM.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), btsGradTempSpace * MID_USE_TIMES);
        curAICWorkspace += btsGradTempSpace * sizeof(qType) * MID_USE_TIMES;

        vGradAccumTemp.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(curAICWorkspace), vGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += vGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        kGradAccumTemp.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(curAICWorkspace), kGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += kGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        tempBtsGtsAccum.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(curAICWorkspace), btsGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += btsGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;
        
        tempBposGposAccum.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(curAICWorkspace), bposGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += bposGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        maskTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), maskTempSpace);

        vecOnceDataNum = DATA_ALIGN_BYTES / sizeof(float) * blockHeight;
        pipe.InitBuffer(queueVecScoreQK, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreGV, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreMask, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreBias, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));

        pipe.InitBuffer(queueVecScoreGposV, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreGtsV, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreBts, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreBpos, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));

        pipe.InitBuffer(queueOutputScore, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputBias, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputTemp, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));

        pipe.InitBuffer(queueOutputGradBts, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputGradBpos, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputBts, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputBpos, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));

        CreateMask();
    }

    __aicore__ inline void CreateMask()
    {
        if (IfMask(maskType, MaskType::MASK_TRIL)) {
            // create lower triangular
            int64_t total = blockHeight * blockHeight;
            int64_t remain = total;
            int64_t thisLen = vecOnceDataNum;
            while (remain > 0) {
                if (remain < thisLen) {
                    thisLen = remain;
                }

                int64_t baseOffset = total - remain;
                int32_t validNums = 1 + baseOffset / blockHeight;

                LocalTensor<qType> input = queueVecScoreMask.AllocTensor<qType>();
                Duplicate<qType>(input, 0, thisLen);
                for (int i = 0; i < thisLen / blockHeight; i++) {
                    if (validNums + i >= blockHeight) {
                        Duplicate<qType>(input[i * blockHeight], 1, blockHeight);
                    } else {
                        Duplicate<qType>(input[i * blockHeight], 1, validNums + i);
                    }
                }
                queueVecScoreMask.EnQue(input);

                LocalTensor<qType> newInput = queueVecScoreMask.DeQue<qType>();
                LocalTensor<qType> output = queueOutputTemp.AllocTensor<qType>();
                DataCopy(output, newInput, thisLen);
                queueOutputTemp.EnQue(output);
                queueVecScoreMask.FreeTensor(newInput);

                output = queueOutputTemp.DeQue<qType>();
                DataCopy(maskTemp[baseOffset], output, thisLen);
                queueOutputTemp.FreeTensor(output);

                remain -= thisLen;
            }

            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void CastQType2Float(LocalTensor<float> dstTensor, LocalTensor<qType> srcTensor,
                                           LocalTensor<qType> midTensor, int64_t len)
    {
        DataCopy<qType>(midTensor, srcTensor, len);
        Cast(dstTensor, midTensor, RoundMode::CAST_NONE, len);
    }

    __aicore__ inline void CastInputData(LocalTensor<float>& inputQK, LocalTensor<float>& inputGV,
        LocalTensor<float>& inputBts, LocalTensor<float>& inputBpos,
        LocalTensor<float>& inputGpV, LocalTensor<float>& inputGtV,
        LocalTensor<float>& inputMask, int64_t thisLen, bool useMask)
    {
        LocalTensor<qType> outputMidTemp = queueOutputTemp.AllocTensor<qType>();
        if (!std::is_same<qType, float>::value) {
            CastQType2Float(inputQK, inputQK.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            CastQType2Float(inputGV, inputGV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            if (useMask) {
                CastQType2Float(inputMask, inputMask.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
            if (enableBias) {
                CastQType2Float(inputGpV, inputGpV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
                CastQType2Float(inputGtV, inputGtV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
                CastQType2Float(inputBts, inputBts.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
                CastQType2Float(inputBpos, inputBpos.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
        }
        queueOutputTemp.FreeTensor(outputMidTemp);
    }

    __aicore__ inline void CalcuScoreWithFloat32(int64_t thisLen, bool useMask)
    {
        auto inputQK = queueVecScoreQK.DeQue<float>();
        auto inputGV = queueVecScoreGV.DeQue<float>();
        LocalTensor<float> inputMask = useMask ? queueVecScoreMask.DeQue<float>() :
            queueVecScoreMask.AllocTensor<float>();

        LocalTensor<float> inputBias = queueVecScoreBias.AllocTensor<float>();
        auto inputBts = enableBias ? queueVecScoreBts.DeQue<float>() : queueVecScoreBts.AllocTensor<float>();
        auto inputBpos = enableBias ? queueVecScoreBpos.DeQue<float>() : queueVecScoreBpos.AllocTensor<float>();
        auto inputGposV = enableBias ? queueVecScoreGposV.DeQue<float>() : queueVecScoreGposV.AllocTensor<float>();
        auto inputGtsV = enableBias ? queueVecScoreGtsV.DeQue<float>() : queueVecScoreGtsV.AllocTensor<float>();

        CastInputData(inputQK, inputGV, inputBts, inputBpos, inputGposV, inputGtsV, inputMask, thisLen, useMask);

        if (enableBias) {
            if (useMask) {
                // Gts = GtsV * mask
                Mul<float>(inputGtsV, inputGtsV, inputMask, thisLen);
                // Gpos = GposV * mask
                Mul<float>(inputGposV, inputGposV, inputMask, thisLen);
                // Bts = Bts * mask
                Mul<float>(inputBts, inputBts, inputMask, thisLen);
                // Bpos = Bpos * mask
                Mul<float>(inputBpos, inputBpos, inputMask, thisLen);
            }
        }

        // score = F.silu(qkb) * siluScale * mask
        Silu<float>(inputBias, inputQK, thisLen);
        Muls<float>(inputBias, inputBias, siluScale, thisLen);
        if (useMask) {
            // for scoreTemp
            Mul<float>(inputBias, inputBias, inputMask, thisLen);
        }

        // score_grad = gv * siluScale * mask
        Muls<float>(inputGV, inputGV, siluScale, thisLen);
        if (useMask) {
            Mul<float>(inputGV, inputGV, inputMask, thisLen);
        }

        // bias_grad = (F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))) * score_grad
        // F.sigmoid(qkb)
        LocalTensor<uint8_t> sigmoidBuffer = queueOutputTemp.AllocTensor<uint8_t>();
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

        queueVecScoreMask.FreeTensor(inputMask);
        queueOutputTemp.FreeTensor(sigmoidBuffer);

        //  (F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))) * score_grad
        Mul<float>(inputQK, inputQK, inputGV, thisLen); // for Gb
        queueVecScoreGV.FreeTensor(inputGV);

        LocalTensor<qType> outputScore = queueOutputScore.AllocTensor<qType>();
        LocalTensor<qType> outputBias = queueOutputBias.AllocTensor<qType>();
        LocalTensor<qType> outputGradBts = queueOutputGradBts.AllocTensor<qType>();
        LocalTensor<qType> outputGradBpos = queueOutputGradBpos.AllocTensor<qType>();
        LocalTensor<qType> outputBts = queueOutputBts.AllocTensor<qType>();
        LocalTensor<qType> outputBpos = queueOutputBpos.AllocTensor<qType>();
        if (!std::is_same<qType, float>::value) {
            Cast(outputScore, inputBias, RoundMode::CAST_RINT, thisLen);
            Cast(outputBias, inputQK, RoundMode::CAST_RINT, thisLen);
            if (this->enableBias) {
                Cast(outputGradBts, inputGtsV, RoundMode::CAST_RINT, thisLen);
                Cast(outputGradBpos, inputGposV, RoundMode::CAST_RINT, thisLen);
                Cast(outputBts, inputBts, RoundMode::CAST_RINT, thisLen);
                Cast(outputBpos, inputBpos, RoundMode::CAST_RINT, thisLen);
            }
        } else {
            LocalTensor<float> newOutputScore = outputScore.template ReinterpretCast<float>();
            DataCopy(newOutputScore, inputBias, thisLen);

            LocalTensor<float> newOutputBias = outputBias.template ReinterpretCast<float>();
            DataCopy(newOutputBias, inputQK, thisLen);

            if (this->enableBias) {
                LocalTensor<float> newOutputGradBts = outputGradBts.template ReinterpretCast<float>();
                DataCopy(newOutputGradBts, inputGtsV, thisLen);

                LocalTensor<float> newOutputGradBpos = outputGradBpos.template ReinterpretCast<float>();
                DataCopy(newOutputGradBpos, inputGposV, thisLen);

                LocalTensor<float> newOutputBts = outputBts.template ReinterpretCast<float>();
                DataCopy(newOutputBts, inputBts, thisLen);

                LocalTensor<float> newOutputBpos = outputBpos.template ReinterpretCast<float>();
                DataCopy(newOutputBpos, inputBpos, thisLen);
            }
        }
        queueVecScoreQK.FreeTensor(inputQK);
        queueVecScoreBias.FreeTensor(inputBias);
        queueVecScoreBts.FreeTensor(inputBts);
        queueVecScoreBpos.FreeTensor(inputBpos);
        queueVecScoreGposV.FreeTensor(inputGposV);
        queueVecScoreGtsV.FreeTensor(inputGtsV);

        queueOutputScore.EnQue(outputScore);
        queueOutputBias.EnQue(outputBias);
        queueOutputGradBts.EnQue(outputGradBts);
        queueOutputGradBpos.EnQue(outputGradBpos);
        queueOutputBts.EnQue(outputBts);
        queueOutputBpos.EnQue(outputBpos);
    }

    __aicore__ inline void CopyInPadding(LocalTensor<qType> dstTensor, GlobalTensor<qType> srcTensor,
                                         int64_t rowNum, int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (seqLen - colNum) * sizeof(qType);
        uint32_t dstStride = (blockHeight - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint8_t rightPadding = (blockHeight - colNum) % (DATA_ALIGN_BYTES / sizeof(qType));

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<qType> padParams{true, 0, rightPadding, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    __aicore__ inline void CopyOutPadding(GlobalTensor<qType> dstTensor, LocalTensor<qType> srcTensor,
                                          int64_t rowNum, int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (blockHeight - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint32_t dstStride = (seqLen - colNum) * sizeof(qType);

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void ValidVecScore(int64_t thisLen, int64_t validRowNum, int64_t totalColNum, int64_t qkOffset,
        int64_t curMaskOffset, int64_t curAttnBiasOffset, int64_t curBtsOffset, int64_t curBposOffset,
        bool useMask)
    {
        int64_t gvOffset = qkOffset;
        int64_t scoreTempOffset = qkOffset;
        LocalTensor<float> inputQK = queueVecScoreQK.AllocTensor<float>();
        DataCopy<qType>(inputQK.template ReinterpretCast<qType>(), qkTemp[qkOffset], thisLen);
        queueVecScoreQK.EnQue(inputQK);

        LocalTensor<float> inputGV = queueVecScoreGV.AllocTensor<float>();
        DataCopy<qType>(inputGV.template ReinterpretCast<qType>(), gvTemp[gvOffset], thisLen);
        queueVecScoreGV.EnQue(inputGV);
        if (useMask) {
            LocalTensor<float> inputMask = queueVecScoreMask.AllocTensor<float>();
            if (IfMask(maskType, MaskType::MASK_TRIL)) {
                DataCopy<qType>(inputMask.template ReinterpretCast<qType>(), maskTemp[curMaskOffset], thisLen);
            }
            if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
                CopyInPadding(inputMask.template ReinterpretCast<qType>(), mask[curMaskOffset], validRowNum,
                    totalColNum, maxSeqLen);
            }
            queueVecScoreMask.EnQue(inputMask);
        }
        if (enableBias) {
            // input bias
            LocalTensor<float> inputBts = queueVecScoreBts.AllocTensor<float>();
            CopyInPadding(inputBts.template ReinterpretCast<qType>(), bts[curBtsOffset], validRowNum, totalColNum,
                          biasGradSeqLen);
            queueVecScoreBts.EnQue(inputBts);
            LocalTensor<float> inputBpos = queueVecScoreBpos.AllocTensor<float>();
            CopyInPadding(inputBpos.template ReinterpretCast<qType>(), bpos[curBposOffset], validRowNum, totalColNum,
                          biasGradSeqLen);
            queueVecScoreBpos.EnQue(inputBpos);

            // input bias grad
            LocalTensor<float> inputGpV = queueVecScoreGposV.AllocTensor<float>();
            DataCopy<qType>(inputGpV.template ReinterpretCast<qType>(), tempGposVT[gvOffset], thisLen);
            queueVecScoreGposV.EnQue(inputGpV);

            LocalTensor<float> inputGtV = queueVecScoreGtsV.AllocTensor<float>();
            DataCopy<qType>(inputGtV.template ReinterpretCast<qType>(), tempGtsVT[gvOffset], thisLen);
            queueVecScoreGtsV.EnQue(inputGtV);
        }

        CalcuScoreWithFloat32(thisLen, useMask);

        LocalTensor<qType> outputScore = queueOutputScore.DeQue<qType>();
        LocalTensor<qType> outputBias = queueOutputBias.DeQue<qType>();
        LocalTensor<qType> outputGradBts = queueOutputGradBts.DeQue<qType>();
        LocalTensor<qType> outputGradBpos = queueOutputGradBpos.DeQue<qType>();
        LocalTensor<qType> outputBts = queueOutputBts.DeQue<qType>();
        LocalTensor<qType> outputBpos = queueOutputBpos.DeQue<qType>();
        DataCopy<qType>(scoreTemp[scoreTempOffset], outputScore, thisLen);
        CopyOutPadding(attnBiasGrad[curAttnBiasOffset], outputBias, validRowNum, totalColNum, biasGradSeqLen);
        if (this->enableBias) {
            DataCopy<qType>(tempBtsM[scoreTempOffset], outputBts, thisLen);
            DataCopy<qType>(tempBposM[scoreTempOffset], outputBpos, thisLen);
            CopyOutPadding(btsGrad[curAttnBiasOffset], outputGradBts, validRowNum, totalColNum, biasGradSeqLen);
            CopyOutPadding(bposGrad[curAttnBiasOffset], outputGradBpos, validRowNum, totalColNum, biasGradSeqLen);
        }

        queueOutputScore.FreeTensor(outputScore);
        queueOutputBias.FreeTensor(outputBias);
        queueOutputBts.FreeTensor(outputBts);
        queueOutputBpos.FreeTensor(outputBpos);
        queueOutputGradBts.FreeTensor(outputGradBts);
        queueOutputGradBpos.FreeTensor(outputGradBpos);
    }

    __aicore__ inline void VecScoreImpl(int64_t taskId, int64_t attnBiasOffset, int64_t attnBiasDiagonalOffset,
                                        int64_t maskOffset, int64_t totalRowNum, int64_t totalColNum, bool useMask,
                                        int64_t btsOffset, int64_t bposOffset)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;

        int64_t total = blockHeight * blockHeight;
        int64_t remain = total;
        int64_t thisLen = vecOnceDataNum;
        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t baseOffset = total - remain;

            int64_t startRowNum = baseOffset / blockHeight;
            int64_t thisRowNum = thisLen / blockHeight;
            int64_t validRowNum = totalRowNum - startRowNum;
            validRowNum = validRowNum > thisRowNum ? thisRowNum : validRowNum;
            validRowNum = validRowNum < 0 ? 0 : validRowNum;

            int64_t qkOffset = midResultIdx * blockHeight * blockHeight + baseOffset;
            int64_t curAttnBiasOffset = attnBiasOffset + startRowNum * biasGradSeqLen;
            int64_t curBtsOffset = btsOffset + startRowNum * biasGradSeqLen;
            int64_t curBposOffset = bposOffset + startRowNum * biasGradSeqLen;
            int64_t curMaskOffset = 0;
            if (IfMask(maskType, MaskType::MASK_TRIL)) {
                curMaskOffset = maskOffset + baseOffset;
            } else if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
                curMaskOffset = maskOffset + startRowNum * maxSeqLen;
            }

            if (validRowNum > 0) {
                ValidVecScore(thisLen, validRowNum, totalColNum, qkOffset, curMaskOffset,
                              curAttnBiasOffset, curBtsOffset, curBposOffset, useMask);
            }
            
            if (enableBias && IfMask(maskType, MaskType::MASK_TRIL) && !useMask) {
                LocalTensor<qType> outputTempTensor = queueOutputTemp.AllocTensor<qType>();
                Duplicate<qType>(outputTempTensor, 0, thisLen);
                queueOutputTemp.EnQue(outputTempTensor);

                int64_t curAttnBiasDiagonalOffset = attnBiasDiagonalOffset + startRowNum * biasGradSeqLen;
                outputTempTensor = queueOutputTemp.DeQue<qType>();
                CopyOutPadding(attnBiasGrad[curAttnBiasDiagonalOffset], outputTempTensor, thisRowNum, totalRowNum,
                    biasGradSeqLen);
                queueOutputTemp.FreeTensor(outputTempTensor);
            }

            remain = remain - thisLen;
        }
    }

    __aicore__ inline void DoTransImpl(GlobalTensor<float> from, GlobalTensor<qType> to,
                                       int64_t fromOffset, int64_t toOffset, int64_t total = 0)
    {
        int64_t remain = total;
        int64_t thisLen = vecOnceDataNum;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curFromOffset = total - remain;
            int64_t curToOffset = curFromOffset * headNum;

            LocalTensor<float> input = queueVecScoreQK.AllocTensor<float>();
            DataCopy(input, from[fromOffset + curFromOffset], thisLen);
            queueVecScoreQK.EnQue(input);

            LocalTensor<float> newInput = queueVecScoreQK.DeQue<float>();
            LocalTensor<qType> output = queueOutputTemp.AllocTensor<qType>();
            if (std::is_same<qType, float>::value) {
                DataCopy(output.template ReinterpretCast<float>(), newInput, thisLen);
            } else {
                Cast(output, newInput, RoundMode::CAST_RINT, thisLen);
            }
            queueOutputTemp.EnQue(output);
            queueVecScoreQK.FreeTensor(newInput);

            LocalTensor<qType> newOutput = queueOutputTemp.DeQue<qType>();
            uint16_t blockCount = thisLen / headDim;
            uint16_t blockLen = headDim * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t dstStride = (headNum * headDim - headDim) * sizeof(qType) / DATA_ALIGN_BYTES;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};
            DataCopy(to[toOffset + curToOffset], newOutput, copyParams);
            queueOutputTemp.FreeTensor(newOutput);

            remain = remain - thisLen;
        }
    }

    GM_ADDR curAICWorkspace;

    // Shape
    int64_t batchSize;
    int64_t seqLen;
    int64_t headNum;
    int64_t headDim;
    int64_t maxSeqLen;
    int64_t biasGradSeqLen;
    int64_t blockHeight;

    // Attr
    int32_t maskType;
    int32_t enableBias;
    float siluScale;

    // Tiling
    int64_t rowBlockNum;
    int64_t colBlockNum;
    int64_t totalRowBlockNum;
    int64_t totalColBlockNum;
    int64_t totalBlockNum;

    // task
    BlockInfo taskInfo[COMPUTE_PIPE_NUM];

    // Tpipe
    TPipe pipe;

    // vec score
    int64_t vecOnceDataNum;
    TQue<TPosition::VECIN, 1> queueVecScoreQK;
    TQue<TPosition::VECIN, 1> queueVecScoreGV;
    TQue<TPosition::VECIN, 1> queueVecScoreMask;
    TQue<TPosition::VECIN, 1> queueVecScoreBias;

    TQue<TPosition::VECIN, 1> queueVecScoreGposV;
    TQue<TPosition::VECIN, 1> queueVecScoreGtsV;
    TQue<TPosition::VECIN, 1> queueVecScoreBts;
    TQue<TPosition::VECIN, 1> queueVecScoreBpos;

    TQue<TPosition::VECOUT, 1> queueOutputGradBts;
    TQue<TPosition::VECOUT, 1> queueOutputGradBpos;
    TQue<TPosition::VECOUT, 1> queueOutputBts;
    TQue<TPosition::VECOUT, 1> queueOutputBpos;

    TQue<TPosition::VECOUT, 1> queueOutputScore;
    TQue<TPosition::VECOUT, 1> queueOutputBias;
    TQue<TPosition::VECOUT, 1> queueOutputTemp;

    // Gt int
    GlobalTensor<qType> grad;
    GlobalTensor<qType> q;
    GlobalTensor<qType> k;
    GlobalTensor<qType> v;
    GlobalTensor<qType> bpos;
    GlobalTensor<qType> bts;
    GlobalTensor<qType> mask;
    GlobalTensor<qType> gradPosition;
    GlobalTensor<qType> gradTimestamp;

    // Gt out
    GlobalTensor<qType> qGrad;
    GlobalTensor<qType> kGrad;
    GlobalTensor<qType> vGrad;
    GlobalTensor<qType> attnBiasGrad;
    GlobalTensor<qType> bposGrad;
    GlobalTensor<qType> btsGrad;
    GlobalTensor<qType> vbposGrad;
    GlobalTensor<qType> vbtsGrad;

    GlobalTensor<qType> qkTemp;
    GlobalTensor<qType> gvTemp;
    GlobalTensor<qType> tempGposVT;
    GlobalTensor<qType> tempGtsVT;
    GlobalTensor<qType> tempBposM;
    GlobalTensor<qType> tempBtsM;
    GlobalTensor<qType> scoreTemp;
    GlobalTensor<float> kGradAccumTemp; // qGrad share temp space with kGrad
    GlobalTensor<float> vGradAccumTemp;
    GlobalTensor<float> tempBtsGtsAccum;
    GlobalTensor<float> tempBposGposAccum;
    GlobalTensor<qType> maskTemp;
};
} // namespace HstuDenseBackwardFuxi
#endif
