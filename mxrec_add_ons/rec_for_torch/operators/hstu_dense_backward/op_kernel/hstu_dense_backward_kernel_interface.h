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


#ifndef HSTU_DENSE_BACKWARD_KERNEL_INTERFACE_H
#define HSTU_DENSE_BACKWARD_KERNEL_INTERFACE_H

#include "hstu_dense_backward_kernel_common.h"

namespace HstuDenseBackward {

template <typename qType>
class HstuDenseBackwardKernelInterface {
public:
    __aicore__ inline HstuDenseBackwardKernelInterface() {}

    __aicore__ inline void InitGlobalBuffer(Args &args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

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

        grad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.grad), totalElementOfQ);
        q.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.q), totalElementOfQ);
        k.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.k), totalElementOfQ);
        v.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.v), totalElementOfQ);
        if (enableBias) {
            attnBias.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.attnBias), totalElementOfAttnBias);
        }
        if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
            mask.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.mask), totalElementOfAttnBias);
        }

        qGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.qGrad), totalElementOfQ);
        kGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.kGrad), totalElementOfQ);
        vGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.vGrad), totalElementOfQ);
        attnBiasGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(args.attnBiasGrad), totalElementOfAttnBias);
    }

    __aicore__ inline void InitPipe(Args &args)
    {
        GM_ADDR workspace = args.workspace;
        int64_t qkMatmulTempSpace = blockHeight * blockHeight;
        int64_t gvMatmulTempSpace = blockHeight * blockHeight;
        int64_t vGradAccumTempSpace = blockHeight * headDim;
        int64_t kGradAccumTempSpace = blockHeight * headDim;
        int64_t scoreTempSpace = blockHeight * blockHeight;
        int64_t maskTempSpace = blockHeight * blockHeight;

        int64_t totalTempSpaceForOneVec =
            MID_USE_TIMES * ((vGradAccumTempSpace + kGradAccumTempSpace) * sizeof(float) +
                             (qkMatmulTempSpace + gvMatmulTempSpace + scoreTempSpace) * sizeof(qType)) +
            maskTempSpace * sizeof(qType);

        curAICWorkspace = reinterpret_cast<__gm__ uint8_t *>(workspace) + GetBlockIdx() * totalTempSpaceForOneVec;

        qkTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), qkMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += qkMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        gvTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), gvMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += gvMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        scoreTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), scoreTempSpace * MID_USE_TIMES);
        curAICWorkspace += scoreTempSpace * sizeof(qType) * MID_USE_TIMES;

        vGradAccumTemp.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(curAICWorkspace),
                                       vGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += vGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        kGradAccumTemp.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(curAICWorkspace),
                                       kGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += kGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        maskTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(curAICWorkspace), maskTempSpace);

        vecOnceDataNum = DATA_ALIGN_BYTES / sizeof(float) * blockHeight;
        pipe.InitBuffer(queueVecScoreQK, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreGV, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreMask, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreBias, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));

        pipe.InitBuffer(queueOutputScore, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputBias, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputTemp, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
    }

    __aicore__ inline void Init(Args &args)
    {
        InitGlobalBuffer(args);
        InitPipe(args);
        CreateMask();
    }

  __aicore__ inline void DuplicateInput(LocalTensor<qType> &input, int64_t thisLen, int32_t validNums)
    {
        Duplicate<qType>(input, 0, thisLen);
        for (int i = 0; i < thisLen / blockHeight; i++) {
            if (validNums + i >= blockHeight) {
                Duplicate<qType>(input[i * blockHeight], 1, blockHeight);
            } else {
                Duplicate<qType>(input[i * blockHeight], 1, validNums + i);
            }
        }
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
                DuplicateInput(input, thisLen, validNums);
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
            this->CastQType2Float(inputQK, inputQK.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            this->CastQType2Float(inputGV, inputGV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            if (useMask) {
                this->CastQType2Float(inputMask, inputMask.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
            if (this->enableBias) {
                this->CastQType2Float(inputBias, inputBias.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
        }
        this->queueOutputTemp.template FreeTensor(outputMidTemp);
    }

    __aicore__ inline void CalcBaseOffsets(int64_t curTaskId, bool isCol = true)
    {
        this->taskInfo[curTaskId].qkLeftOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum *
            this->headDim + this->taskInfo[curTaskId].rowId * this->blockHeight * this->headNum * this->headDim +
            this->taskInfo[curTaskId].headId * this->headDim;
        this->taskInfo[curTaskId].qkRightOffset = this->taskInfo[curTaskId].batchId * this->seqLen * this->headNum *
            this->headDim + this->taskInfo[curTaskId].colId * this->blockHeight *
            this->headNum * this->headDim + this->taskInfo[curTaskId].headId * this->headDim;
        this->taskInfo[curTaskId].kGradLeftOffset = this->taskInfo[curTaskId].batchId * this->headNum *
            this->biasGradSeqLen * this->biasGradSeqLen + this->taskInfo[curTaskId].headId * this->biasGradSeqLen *
            this->biasGradSeqLen + this->taskInfo[curTaskId].rowId * this->blockHeight * this->biasGradSeqLen +
            this->taskInfo[curTaskId].colId * this->blockHeight;
        if (isCol) {
            this->taskInfo[curTaskId].vGradRightOffset = this->taskInfo[curTaskId].batchId * this->seqLen *
                this->headNum * this->headDim + this->taskInfo[curTaskId].rowId * this->blockHeight *
                this->headNum * this->headDim + this->taskInfo[curTaskId].headId * this->headDim;

            this->taskInfo[curTaskId].rowLine = this->seqLen - this->taskInfo[curTaskId].rowId * this->blockHeight;
            if (this->taskInfo[curTaskId].rowLine > this->blockHeight) {
                this->taskInfo[curTaskId].rowLine = this->blockHeight;
            }
        } else {
            this->taskInfo[curTaskId].vGradRightOffset = this->taskInfo[curTaskId].batchId * this->seqLen *
                this->headNum * this->headDim + this->taskInfo[curTaskId].colId * this->blockHeight *
                this->headNum * this->headDim + this->taskInfo[curTaskId].headId * this->headDim;

            this->taskInfo[curTaskId].colLine = this->seqLen - this->taskInfo[curTaskId].colId * this->blockHeight;
            if (this->taskInfo[curTaskId].colLine > this->blockHeight) {
                this->taskInfo[curTaskId].colLine = this->blockHeight;
            }
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

    TQue<TPosition::VECOUT, 1> queueOutputScore;
    TQue<TPosition::VECOUT, 1> queueOutputBias;
    TQue<TPosition::VECOUT, 1> queueOutputTemp;

    // Gt
    GlobalTensor<qType> grad;
    GlobalTensor<qType> q;
    GlobalTensor<qType> k;
    GlobalTensor<qType> v;
    GlobalTensor<qType> attnBias;
    GlobalTensor<qType> mask;

    GlobalTensor<qType> qGrad;
    GlobalTensor<qType> kGrad;
    GlobalTensor<qType> vGrad;
    GlobalTensor<qType> attnBiasGrad;

    GlobalTensor<qType> qkTemp;
    GlobalTensor<qType> gvTemp;
    GlobalTensor<qType> scoreTemp;
    GlobalTensor<float> kGradAccumTemp;  // qGrad share temp space with kGrad
    GlobalTensor<float> vGradAccumTemp;
    GlobalTensor<qType> maskTemp;

    // Matmul
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, CopyQKA1<qType>, CopyQKB1<qType>>>
        qkMatmul;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, CopyQGradA1<qType>, CopyVGradB1<qType>>>
        qGradMatmul;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, CopyKGradA1<qType>, CopyVGradB1<qType>>>
        kGradMatmul;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopyVGradB1<qType>>>
        vGradMatmul;
};
}

#endif