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

#include <cstdint>
#include "hstu_dense_backward_kernel_common.h"
#include "hstu_mask.h"

namespace HstuDenseBackward {

struct BlockInfo {
    int64_t taskId;
    int64_t batchId;
    int64_t headId;
    int64_t rowId;
    int64_t colId;
    int64_t accumId;
    int64_t qkLeftOffset;
    int64_t qkRightOffset;
    int64_t kGradLeftOffset;
    int64_t vGradRightOffset;
    int64_t rowLine;
    int64_t colLine;
};

template <typename qType>
class HstuDenseBackwardKernel {
public:
    __aicore__ inline HstuDenseBackwardKernel() {}

    __aicore__ inline void Compute(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), qkMatmul, &tilingData.qkMatmul, qGradMatmul,
                          &tilingData.qGradMatmul, kGradMatmul, &tilingData.kGradMatmul, vGradMatmul,
                          &tilingData.vGradMatmul);
        uint64_t tilingPtr = reinterpret_cast<uint64_t>(args.tiling);
        qkMatmul.SetUserDefInfo(tilingPtr);
        qGradMatmul.SetUserDefInfo(tilingPtr);
        kGradMatmul.SetUserDefInfo(tilingPtr);
        vGradMatmul.SetUserDefInfo(tilingPtr);

        Init(args);
        ComputeFirst();
        ComputeSecond();
    }

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
        targetGroupSize = tilingData.targetGroupSize;
        alpha = tilingData.alpha;

        blockHeight = tilingData.blockHeight;

        maskType = tilingData.maskType;
        enableBias = tilingData.enableBias;
        enableContextMask = tilingData.enableContextMask;
        enableTargetMask = tilingData.enableTargetMask;
        isNormal = tilingData.isNormal;
        aivNum = tilingData.aivNum;

        rowBlockNum = (seqLen + blockHeight - 1) / blockHeight;
        colBlockNum = (seqLen + blockHeight - 1) / blockHeight;
        totalRowBlockNum = batchSize * headNum * rowBlockNum;
        totalColBlockNum = batchSize * headNum * colBlockNum;
        totalBlockNum = totalRowBlockNum * colBlockNum;

        int64_t totalElementOfQ = batchSize * maxSeqLen * headNum * headDim;
        int64_t totalElementOfAttnBias = batchSize * headNum * biasGradSeqLen * biasGradSeqLen;

        grad.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.grad), totalElementOfQ);
        q.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.q), totalElementOfQ);
        k.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.k), totalElementOfQ);
        v.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.v), totalElementOfQ);
        if (enableBias) {
            attnBias.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.attnBias), totalElementOfAttnBias);
        }
        if (enableContextMask) {
            numContextGt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.numContext), batchSize);
        }
        if (enableTargetMask) {
            numTargetGt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.numTarget), batchSize);
        }
        if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
            mask.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.mask), totalElementOfAttnBias);
        }

        qGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.qGrad), totalElementOfQ);
        kGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.kGrad), totalElementOfQ);
        vGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.vGrad), totalElementOfQ);
        if (isNormal || enableBias) {
            attnBiasGrad.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(args.attnBiasGrad), totalElementOfAttnBias);
        }

        int64_t qkMatmulTempSpace = blockHeight * blockHeight;
        int64_t gvMatmulTempSpace = blockHeight * blockHeight;
        int64_t vGradAccumTempSpace = blockHeight * headDim;
        int64_t kGradAccumTempSpace = blockHeight * headDim;
        int64_t scoreTempSpace = blockHeight * blockHeight;
        int64_t maskTempSpace = blockHeight * blockHeight;
        int64_t biasGradTempSpace = blockHeight * blockHeight;
        int64_t qGradAccumTempSpace = batchSize * headNum * maxSeqLen * headDim;

        int64_t totalTempSpaceForOneVec =
            MID_USE_TIMES * ((vGradAccumTempSpace + kGradAccumTempSpace) * sizeof(float) +
                             (qkMatmulTempSpace + gvMatmulTempSpace + scoreTempSpace) * sizeof(qType)) +
            maskTempSpace * sizeof(qType);

        if (!isNormal && !enableBias) {
            totalTempSpaceForOneVec += biasGradTempSpace * sizeof(qType) * MID_USE_TIMES;
        }

        curAICWorkspace = reinterpret_cast<__gm__ uint8_t*>(workspace) + GetBlockIdx() * totalTempSpaceForOneVec;

        qkTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace), qkMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += qkMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        gvTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace), gvMatmulTempSpace * MID_USE_TIMES);
        curAICWorkspace += gvMatmulTempSpace * sizeof(qType) * MID_USE_TIMES;

        scoreTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace), scoreTempSpace * MID_USE_TIMES);
        curAICWorkspace += scoreTempSpace * sizeof(qType) * MID_USE_TIMES;

        vGradAccumTemp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(curAICWorkspace),
                                       vGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += vGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        kGradAccumTemp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(curAICWorkspace),
                                       kGradAccumTempSpace * MID_USE_TIMES);
        curAICWorkspace += kGradAccumTempSpace * sizeof(float) * MID_USE_TIMES;

        maskTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace), maskTempSpace);
        curAICWorkspace += maskTempSpace * sizeof(qType);

        if (!isNormal && !enableBias) {
            biasGradTemp.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(curAICWorkspace),
                                         biasGradTempSpace * MID_USE_TIMES);
            curAICWorkspace += biasGradTempSpace * sizeof(qType) * MID_USE_TIMES;

            qGradAccumTemp.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(workspace) +
                                                aivNum * totalTempSpaceForOneVec),
                qGradAccumTempSpace);

            // 所有核共享一片globalMemory，且存在累加操作，每次执行需要清理内存防止上次执行结果残留数据影响本次结果
            // 多核执行后需要调用SyncAll保证多核间同步正常
            int64_t unitClear = qGradAccumTempSpace / aivNum;
            int64_t leftClear = qGradAccumTempSpace % aivNum;
            uint64_t globalOffset = GetBlockIdx() * unitClear;
            uint64_t clearLen = unitClear;
            if (GetBlockIdx() == aivNum - 1) {
                clearLen += leftClear;
            }
            GlobalTensor<float> thisBlockQGrad;
            thisBlockQGrad.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(workspace) +
                                                aivNum * totalTempSpaceForOneVec + globalOffset * sizeof(float)),
                clearLen);
            InitGlobalMemory(thisBlockQGrad, clearLen, static_cast<float>(0));
            SyncAll();
        }

        vecOnceDataNum = DATA_ALIGN_BYTES / sizeof(float) * blockHeight;
        pipe.InitBuffer(queueVecScoreQK, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreGV, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreMask, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));
        pipe.InitBuffer(queueVecScoreBias, USE_BUFFER_NUM, vecOnceDataNum * sizeof(float));

        pipe.InitBuffer(queueOutputScore, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputBias, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));
        pipe.InitBuffer(queueOutputTemp, USE_BUFFER_NUM, vecOnceDataNum * sizeof(qType));

        CreateMask();
    }

    __aicore__ inline void CalcBaseOffsets(int64_t curTaskId, bool isCol = true)
    {
        taskInfo[curTaskId].qkLeftOffset = taskInfo[curTaskId].batchId * seqLen * headNum * headDim +
                                           taskInfo[curTaskId].rowId * blockHeight * headNum * headDim +
                                           taskInfo[curTaskId].headId * headDim;
        taskInfo[curTaskId].qkRightOffset = taskInfo[curTaskId].batchId * seqLen * headNum * headDim +
                                            taskInfo[curTaskId].colId * blockHeight * headNum * headDim +
                                            taskInfo[curTaskId].headId * headDim;
        taskInfo[curTaskId].kGradLeftOffset = taskInfo[curTaskId].batchId * headNum * biasGradSeqLen * biasGradSeqLen +
                                              taskInfo[curTaskId].headId * biasGradSeqLen * biasGradSeqLen +
                                              taskInfo[curTaskId].rowId * blockHeight * biasGradSeqLen +
                                              taskInfo[curTaskId].colId * blockHeight;
        if (isCol) {
            taskInfo[curTaskId].vGradRightOffset = taskInfo[curTaskId].batchId * seqLen * headNum * headDim +
                                                   taskInfo[curTaskId].rowId * blockHeight * headNum * headDim +
                                                   taskInfo[curTaskId].headId * headDim;

            taskInfo[curTaskId].rowLine = seqLen - taskInfo[curTaskId].rowId * blockHeight;
            if (taskInfo[curTaskId].rowLine > blockHeight) {
                taskInfo[curTaskId].rowLine = blockHeight;
            }
        } else {
            taskInfo[curTaskId].vGradRightOffset = taskInfo[curTaskId].batchId * seqLen * headNum * headDim +
                                                   taskInfo[curTaskId].colId * blockHeight * headNum * headDim +
                                                   taskInfo[curTaskId].headId * headDim;

            taskInfo[curTaskId].colLine = seqLen - taskInfo[curTaskId].colId * blockHeight;
            if (taskInfo[curTaskId].colLine > blockHeight) {
                taskInfo[curTaskId].colLine = blockHeight;
            }
        }
    }

    __aicore__ inline void DoQKMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * blockHeight * blockHeight;

        qkMatmul.SetTail(taskInfo[curTaskId].rowLine, taskInfo[curTaskId].colLine, headDim);
        DoQKMatmulImpl(taskInfo[curTaskId].qkLeftOffset, taskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoQKMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(q[left]);
        qkMatmul.SetTensorB(k[right], true);

        qkMatmul.template IterateAll<false>(qkTemp[out], 0, false, true);
    }

    __aicore__ inline void DoGVMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t outOffset = midResultIdx * blockHeight * blockHeight;

        qkMatmul.SetTail(taskInfo[curTaskId].rowLine, taskInfo[curTaskId].colLine, headDim);
        DoGVMatmulImpl(taskInfo[curTaskId].qkLeftOffset, taskInfo[curTaskId].qkRightOffset, outOffset);
    }

    __aicore__ inline void DoGVMatmulImpl(int64_t left, int64_t right, int64_t out)
    {
        qkMatmul.SetTensorA(grad[left]);
        qkMatmul.SetTensorB(v[right], true);

        qkMatmul.template IterateAll<false>(gvTemp[out], 0, false, true);
    }

    __aicore__ inline void DoQGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = taskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * blockHeight * headDim;

        bool isNew = taskInfo[curTaskId].colId == 0;

        qGradMatmul.SetTail(taskInfo[curTaskId].rowLine, headDim, taskInfo[curTaskId].colLine);
        DoQGradMatmulImpl(taskInfo[curTaskId].kGradLeftOffset, taskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoQGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        if (isNormal || enableBias) {
            qGradMatmul.SetTensorA(attnBiasGrad[left]);
            qGradMatmul.SetTensorB(k[right]);
            uint8_t enAtomic = isNew ? 0 : 1;
            qGradMatmul.template IterateAll<false>(kGradAccumTemp[out], enAtomic, false, true);
        } else {
            qGradMatmul.SetTensorA(biasGradTemp[left]);
            qGradMatmul.SetTensorB(k[right]);
            uint8_t enAtomic = 1;
            qGradMatmul.template IterateAll<false>(qGradAccumTemp[out], enAtomic, false, true);
        }
    }

    __aicore__ inline void DoKGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midAccumIdx = taskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t outOffset = midAccumIdx * blockHeight * headDim;

        bool isNew = false;
        if (IfMask(maskType, MaskType::MASK_TRIL)) {
            isNew = this->blockMaskParams[curTaskId].IsFirstBlockNeedOverride();
        } else {
            isNew = taskInfo[curTaskId].rowId == 0;
        }

        kGradMatmul.SetTail(taskInfo[curTaskId].colLine, headDim, taskInfo[curTaskId].rowLine);
        DoKGradMatmulImpl(taskInfo[curTaskId].kGradLeftOffset, taskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoKGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        if (isNormal || enableBias) {
            kGradMatmul.SetTensorA(attnBiasGrad[left], true);
        } else {
            kGradMatmul.SetTensorA(biasGradTemp[left], true);
        }

        kGradMatmul.SetTensorB(q[right]);
        if (isNew) {
            kGradMatmul.template IterateAll<false>(kGradAccumTemp[out], 0, false, true);
        } else {
            kGradMatmul.template IterateAll<false>(kGradAccumTemp[out], 1, false, true);
        }
    }

    __aicore__ inline void DoVGradMatmul(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;
        int64_t midAccumIdx = taskInfo[curTaskId].accumId % MID_USE_TIMES;

        int64_t scoreTempOffset = midResultIdx * blockHeight * blockHeight;
        int64_t outOffset = midAccumIdx * blockHeight * headDim;

        bool isNew = false;
        if (IfMask(maskType, MaskType::MASK_TRIL)) {
            isNew = this->blockMaskParams[curTaskId].IsFirstBlockNeedOverride();
        } else {
            isNew = taskInfo[curTaskId].rowId == 0;
        }

        vGradMatmul.SetTail(taskInfo[curTaskId].colLine, headDim, taskInfo[curTaskId].rowLine);
        DoVGradMatmulImpl(scoreTempOffset, taskInfo[curTaskId].vGradRightOffset, outOffset, isNew);
    }

    __aicore__ inline void DoVGradMatmulImpl(int64_t left, int64_t right, int64_t out, bool isNew)
    {
        vGradMatmul.SetTensorA(scoreTemp[left], true);
        vGradMatmul.SetTensorB(grad[right]);
        if (isNew) {
            vGradMatmul.template IterateAll<false>(vGradAccumTemp[out], 0, false, true);
        } else {
            vGradMatmul.template IterateAll<false>(vGradAccumTemp[out], 1, false, true);
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
                                         LocalTensor<float>& inputMask, LocalTensor<float>& inputBias, int64_t thisLen,
                                         bool useMask)
    {
        LocalTensor<qType> outputMidTemp = queueOutputTemp.AllocTensor<qType>();
        if (!std::is_same<qType, float>::value) {
            CastQType2Float(inputQK, inputQK.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            CastQType2Float(inputGV, inputGV.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            if (useMask && IfMask(maskType, MaskType::MASK_CUSTOM)) {
                CastQType2Float(inputMask, inputMask.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
            if (enableBias) {
                CastQType2Float(inputBias, inputBias.template ReinterpretCast<qType>(), outputMidTemp, thisLen);
            }
        }
        queueOutputTemp.FreeTensor(outputMidTemp);
    }

    __aicore__ inline void CalcuScoreWithFloat32(int64_t thisLen, bool useMask)
    {
        auto inputQK = queueVecScoreQK.DeQue<float>();
        auto inputGV = queueVecScoreGV.DeQue<float>();
        LocalTensor<float> inputMask =
            useMask ? queueVecScoreMask.DeQue<float>() : queueVecScoreMask.AllocTensor<float>();
        LocalTensor<float> inputBias =
            enableBias ? queueVecScoreBias.DeQue<float>() : queueVecScoreBias.AllocTensor<float>();

        CastInputData(inputQK, inputGV, inputMask, inputBias, thisLen, useMask);

        if (enableBias) {
            // qkb = qk + attn_bias
            Add<float>(inputQK, inputQK, inputBias, thisLen);
        }

        Muls<float>(inputQK, inputQK, alpha, thisLen);
        Silu<float>(inputBias, inputQK, thisLen);
        Muls<float>(inputBias, inputBias, siluScale, thisLen);
        if (useMask) {
            Mul<float>(inputBias, inputBias, inputMask, thisLen);
        }

        // score_grad = gv * siluScale * mask
        Muls<float>(inputGV, inputGV, siluScale, thisLen);
        if (useMask) {
            Mul<float>(inputGV, inputGV, inputMask, thisLen);
        }

        // bias_grad = (F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))) * score_grad
        //  F.sigmoid(qkb)
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
        Mul<float>(inputQK, inputQK, inputGV, thisLen);
        Muls<float>(inputQK, inputQK, alpha, thisLen);

        queueVecScoreGV.FreeTensor(inputGV);

        LocalTensor<qType> outputScore = queueOutputScore.AllocTensor<qType>();
        LocalTensor<qType> outputBias = queueOutputBias.AllocTensor<qType>();
        if (!std::is_same<qType, float>::value) {
            Cast(outputScore, inputBias, RoundMode::CAST_RINT, thisLen);
            Cast(outputBias, inputQK, RoundMode::CAST_RINT, thisLen);
        } else {
            LocalTensor<float> newOutputScore = outputScore.template ReinterpretCast<float>();
            DataCopy(newOutputScore, inputBias, thisLen);

            LocalTensor<float> newOutputBias = outputBias.template ReinterpretCast<float>();
            DataCopy(newOutputBias, inputQK, thisLen);
        }
        queueVecScoreQK.FreeTensor(inputQK);
        queueVecScoreBias.FreeTensor(inputBias);

        queueOutputScore.EnQue(outputScore);
        queueOutputBias.EnQue(outputBias);
    }

    __aicore__ inline void VecScore(int64_t taskId)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t attnBiasOffset = taskInfo[curTaskId].batchId * headNum * biasGradSeqLen * biasGradSeqLen +
                                 taskInfo[curTaskId].headId * biasGradSeqLen * biasGradSeqLen +
                                 taskInfo[curTaskId].rowId * blockHeight * biasGradSeqLen +
                                 taskInfo[curTaskId].colId * blockHeight;
        int64_t attnBiasDiagonalOffset = taskInfo[curTaskId].batchId * headNum * biasGradSeqLen * biasGradSeqLen +
                                         taskInfo[curTaskId].headId * biasGradSeqLen * biasGradSeqLen +
                                         taskInfo[curTaskId].colId * blockHeight * biasGradSeqLen +
                                         taskInfo[curTaskId].rowId * blockHeight;

        int64_t maskOffset = 0;
        if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
            maskOffset = taskInfo[curTaskId].batchId * headNum * maxSeqLen * maxSeqLen +
                         taskInfo[curTaskId].headId * maxSeqLen * maxSeqLen +
                         taskInfo[curTaskId].rowId * blockHeight * maxSeqLen + taskInfo[curTaskId].colId * blockHeight;
        }

        bool useMask = false;
        if (IfMask(maskType, MaskType::MASK_TRIL)) {
            useMask = blockMaskParams[curTaskId].NeedMask();
        } else if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
            useMask = true;
        }

        VecScoreImpl(taskId, attnBiasOffset, attnBiasDiagonalOffset, maskOffset, taskInfo[curTaskId].rowLine,
                     taskInfo[curTaskId].colLine, useMask);
    }

    __aicore__ inline void CopyInPadding(LocalTensor<qType> dstTensor, GlobalTensor<qType> srcTensor, int64_t rowNum,
                                         int64_t colNum, int64_t seqLen)
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

    __aicore__ inline void CopyOutPadding(GlobalTensor<qType> dstTensor, LocalTensor<qType> srcTensor, int64_t rowNum,
                                          int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (blockHeight - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint32_t dstStride = (seqLen - colNum) * sizeof(qType);

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void ValidVecScore(int64_t thisLen, int64_t validRowNum, int64_t totalColNum, int64_t qkOffset,
                                         int64_t curMaskOffset, int64_t curAttnBiasOffset, int64_t curBiasGradOutOffset,
                                         bool useMask, BlockMaskGenerator& generator, int64_t rowInBlock)
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
                generator.GenMask(inputMask, rowInBlock, thisLen / blockHeight, blockHeight);
            }
            if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
                CopyInPadding(inputMask.template ReinterpretCast<qType>(), mask[curMaskOffset], validRowNum,
                              totalColNum, maxSeqLen);
            }
            queueVecScoreMask.EnQue(inputMask);
        }
        if (enableBias) {
            LocalTensor<float> inputBias = queueVecScoreBias.AllocTensor<float>();
            CopyInPadding(inputBias.template ReinterpretCast<qType>(), attnBias[curAttnBiasOffset], validRowNum,
                          totalColNum, biasGradSeqLen);
            queueVecScoreBias.EnQue(inputBias);
        }

        CalcuScoreWithFloat32(thisLen, useMask);

        LocalTensor<qType> outputScore = queueOutputScore.DeQue<qType>();
        LocalTensor<qType> outputBias = queueOutputBias.DeQue<qType>();
        DataCopy<qType>(scoreTemp[scoreTempOffset], outputScore, thisLen);
        if (isNormal || enableBias) {
            CopyOutPadding(attnBiasGrad[curAttnBiasOffset], outputBias, validRowNum, totalColNum, biasGradSeqLen);
        } else {
            CopyOutPadding(biasGradTemp[curBiasGradOutOffset], outputBias, validRowNum, totalColNum, blockHeight);
        }
        queueOutputScore.FreeTensor(outputScore);
        queueOutputBias.FreeTensor(outputBias);
    }

    __aicore__ inline void VecScoreImpl(int64_t taskId, int64_t attnBiasOffset, int64_t attnBiasDiagonalOffset,
                                        int64_t maskOffset, int64_t totalRowNum, int64_t totalColNum, bool useMask)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskId % MID_USE_TIMES;

        int64_t total = blockHeight * blockHeight;
        int64_t remain = total;
        int64_t thisLen = vecOnceDataNum;
        BlockMaskGenerator generator(&blockMaskParams[curTaskId]);
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
            int64_t curBiasGradOutOffset = midResultIdx * blockHeight * blockHeight + startRowNum * blockHeight;
            int64_t curMaskOffset = 0;
            if (IfMask(maskType, MaskType::MASK_TRIL)) {
                curMaskOffset = maskOffset + baseOffset;
            } else if (IfMask(maskType, MaskType::MASK_CUSTOM)) {
                curMaskOffset = maskOffset + startRowNum * maxSeqLen;
            }

            if (validRowNum > 0) {
                ValidVecScore(thisLen, validRowNum, totalColNum, qkOffset, curMaskOffset, curAttnBiasOffset,
                              curBiasGradOutOffset, useMask, generator, startRowNum);
            }

            if (enableBias && IfMask(maskType, MaskType::MASK_TRIL) &&
                blockMaskParams[curTaskId].DiagonalNoComputation()) {
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

    __aicore__ inline void DoTrans(int64_t taskId, GlobalTensor<float> from, GlobalTensor<qType> to, bool isCol = true)
    {
        int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
        int64_t midResultIdx = taskInfo[curTaskId].accumId % MID_USE_TIMES;
        int64_t fromOffset = midResultIdx * blockHeight * headDim;
        int64_t toOffset = 0;
        int64_t total = 0;
        if (isCol) {
            toOffset = taskInfo[curTaskId].batchId * seqLen * headNum * headDim +
                       taskInfo[curTaskId].colId * blockHeight * headNum * headDim +
                       taskInfo[curTaskId].headId * headDim;
            total = taskInfo[curTaskId].colLine * headDim;
        } else {
            toOffset = taskInfo[curTaskId].batchId * seqLen * headNum * headDim +
                       taskInfo[curTaskId].rowId * blockHeight * headNum * headDim +
                       taskInfo[curTaskId].headId * headDim;
            total = taskInfo[curTaskId].rowLine * headDim;
        }

        DoTransImpl(from, to, fromOffset, toOffset, total);
    }

    __aicore__ inline void DoTransImpl(GlobalTensor<float> from, GlobalTensor<qType> to, int64_t fromOffset,
                                       int64_t toOffset, int64_t total = 0)
    {
        int64_t remain = total;
        int64_t copyLenEachLoopAlignHeadDim = vecOnceDataNum / headDim * headDim;
        int64_t thisLen = copyLenEachLoopAlignHeadDim;
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

        qkMatmul.WaitIterateAll();
        qkMatmul.End();
        qkMatmul.WaitIterateAll();
        qkMatmul.End();
        if (taskId > 1) {
            vGradMatmul.WaitIterateAll();
            vGradMatmul.End();
            kGradMatmul.WaitIterateAll();
            kGradMatmul.End();
            if (taskInfo[(taskId - TWO) % COMPUTE_PIPE_NUM].accumId !=
                taskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId) {
                DoTrans(taskId - TWO, vGradAccumTemp, vGrad);
                DoTrans(taskId - TWO, kGradAccumTemp, kGrad);
            }
        }
    }

    __aicore__ inline void FirstStageEnding(int64_t taskId)
    {
        if (taskId > 1) {
            DoVGradMatmul(taskId - TWO);
            DoKGradMatmul(taskId - TWO);
            VecScore(taskId - 1);
            vGradMatmul.WaitIterateAll();
            vGradMatmul.End();
            kGradMatmul.WaitIterateAll();
            kGradMatmul.End();
            if (taskInfo[(taskId - TWO) % COMPUTE_PIPE_NUM].accumId !=
                taskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId) {
                DoTrans(taskId - TWO, vGradAccumTemp, vGrad);
                DoTrans(taskId - TWO, kGradAccumTemp, kGrad);
            }

            DoVGradMatmul(taskId - 1);
            DoKGradMatmul(taskId - 1);
            vGradMatmul.WaitIterateAll();
            vGradMatmul.End();
            kGradMatmul.WaitIterateAll();
            kGradMatmul.End();
            DoTrans(taskId - 1, vGradAccumTemp, vGrad);
            DoTrans(taskId - 1, kGradAccumTemp, kGrad);
        }

        if (taskId == 1) {
            VecScore(taskId - 1);

            DoVGradMatmul(taskId - 1);
            DoKGradMatmul(taskId - 1);
            vGradMatmul.WaitIterateAll();
            vGradMatmul.End();
            kGradMatmul.WaitIterateAll();
            kGradMatmul.End();
            DoTrans(taskId - 1, vGradAccumTemp, vGrad);
            DoTrans(taskId - 1, kGradAccumTemp, kGrad);
        }
    }

    __aicore__ inline int64_t GetNumContext(int64_t batchId)
    {
        if (enableContextMask) {
            return numContextGt.GetValue(batchId);
        }
        return 0;
    }

    __aicore__ inline int64_t GetNumTarget(int64_t batchId)
    {
        if (enableTargetMask) {
            return numTargetGt.GetValue(batchId);
        }
        return 0;
    }

    __aicore__ inline void ComputeFirst()
    {
        int64_t taskId = 0;
        int64_t accumId = 0;

        int64_t totalAivNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        int64_t startId = GetBlockIdx();
        int64_t nextCol = totalAivNum * TWO - GetBlockIdx() * TWO - 1;

        for (int64_t gColId = startId; gColId < totalColBlockNum;) {
            int64_t batchId = gColId / (headNum * colBlockNum);
            int64_t colIdInBatch = gColId % (headNum * colBlockNum);
            int64_t headId = colIdInBatch / colBlockNum;
            int64_t colId = colIdInBatch % colBlockNum;
            int64_t colLine = seqLen - colId * blockHeight;
            colLine = colLine > blockHeight ? blockHeight : colLine;

            for (int64_t rowId = 0; rowId < rowBlockNum; rowId++) {
                this->blockMaskParams[taskId % COMPUTE_PIPE_NUM] = {
                    static_cast<uint32_t>(rowId),  static_cast<uint32_t>(colId),
                    static_cast<uint32_t>(seqLen), this->blockHeight,
                    GetNumContext(batchId),        GetNumTarget(batchId),
                    this->targetGroupSize,         1};
                if (IfMask(maskType, MaskType::MASK_TRIL) &&
                    this->blockMaskParams[taskId % COMPUTE_PIPE_NUM].NoComputation()) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                taskInfo[curTaskId] = BlockInfo{taskId, batchId, headId, rowId, colId, accumId};
                taskInfo[curTaskId].colLine = colLine;
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
            if (taskInfo[(taskId - 1) % COMPUTE_PIPE_NUM].accumId != taskInfo[taskId % COMPUTE_PIPE_NUM].accumId) {
                DoTrans(taskId - 1, kGradAccumTemp, qGrad, 0);
            }
        }
        qGradMatmul.WaitIterateAll();
        qGradMatmul.End();
    }

    __aicore__ inline void ComputeSecond()
    {
        SyncAll();

        int64_t taskId = 0;
        int64_t accumId = 0;

        int64_t totalAivNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        int64_t startId = GetBlockIdx();
        int64_t nextRow = totalAivNum * TWO - GetBlockIdx() * TWO - 1;

        for (int64_t gRowId = startId; gRowId < totalRowBlockNum;) {
            int64_t batchId = gRowId / (headNum * rowBlockNum);
            int64_t rowIdInBatch = gRowId % (headNum * rowBlockNum);
            int64_t headId = rowIdInBatch / rowBlockNum;
            int64_t rowId = rowIdInBatch % rowBlockNum;
            int64_t rowLine = seqLen - rowId * blockHeight;
            rowLine = rowLine > blockHeight ? blockHeight : rowLine;

            for (int64_t colId = 0; colId < colBlockNum; colId++) {
                this->blockMaskParams[taskId % COMPUTE_PIPE_NUM] = {
                    static_cast<uint32_t>(rowId),  static_cast<uint32_t>(colId),
                    static_cast<uint32_t>(seqLen), this->blockHeight,
                    GetNumContext(batchId),        GetNumTarget(batchId),
                    this->targetGroupSize,         1};
                if (IfMask(maskType, MaskType::MASK_TRIL) &&
                    this->blockMaskParams[taskId % COMPUTE_PIPE_NUM].NoComputation()) {
                    continue;
                }

                int64_t curTaskId = taskId % COMPUTE_PIPE_NUM;
                taskInfo[curTaskId] = BlockInfo{taskId, batchId, headId, rowId, colId, accumId};
                taskInfo[curTaskId].rowLine = rowLine;
                CalcBaseOffsets(curTaskId, false);

                SecondStagePipeline(taskId);

                taskId++;
            }

            accumId++;
            gRowId += nextRow;
            nextRow = totalAivNum * TWO - nextRow;
        }

        if (taskId > 0) {
            DoTrans(taskId - 1, kGradAccumTemp, qGrad, 0);
        }
    }

    template <typename oType>
    __aicore__ inline void DoCopyBlockQGrad(int64_t thisBatchIdx, int64_t headIdx, int64_t curSeqLen,
                                            GlobalTensor<oType>& seqOffsets)
    {
        int64_t totalLen = curSeqLen * headDim;
        int64_t remain = totalLen;
        int64_t copyLenEachLoopAlignHeadDim = vecOnceDataNum / headDim * headDim;
        int64_t thisLen = copyLenEachLoopAlignHeadDim;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curOffset =
                (headNum * seqOffsets.GetValue(thisBatchIdx) * headDim) + (headIdx * totalLen) + (totalLen - remain);
            LocalTensor<float> input = queueVecScoreQK.AllocTensor<float>();
            DataCopy<float>(input, qGradAccumTemp[curOffset], thisLen);
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
            uint16_t dstStride = (headNum - 1) * headDim * sizeof(qType) / DATA_ALIGN_BYTES;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};

            int64_t curOutOffset = seqOffsets.GetValue(thisBatchIdx) * headNum * headDim + headIdx * headDim +
                                   (totalLen - remain) * headNum;
            DataCopy<qType>(qGrad[curOutOffset], newOutput, copyParams);
            queueOutputTemp.FreeTensor(newOutput);

            remain = remain - thisLen;
        }
    }

    template <typename oType>
    __aicore__ inline void DoCopyQGrad(GlobalTensor<oType>& seqOffsets)
    {
        int64_t batchIdx = GetBlockIdx();
        int64_t taskNum = batchSize * headNum;
        int64_t coreTask = taskNum / aivNum;
        int64_t coreSplitId = taskNum % aivNum;

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
    int32_t enableContextMask;
    int32_t enableTargetMask;
    float siluScale;

    int32_t isNormal;
    uint32_t aivNum;

    // Tiling
    int64_t rowBlockNum;
    int64_t colBlockNum;
    int64_t totalRowBlockNum;
    int64_t totalColBlockNum;
    int64_t totalBlockNum;

    // task
    BlockInfo taskInfo[COMPUTE_PIPE_NUM];

    // MaskType
    int64_t targetGroupSize;
    float alpha;
    BlockMaskParams blockMaskParams[COMPUTE_PIPE_NUM];

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
    GlobalTensor<int64_t> numContextGt;
    GlobalTensor<int64_t> numTargetGt;

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
    GlobalTensor<float> qGradAccumTemp;
    GlobalTensor<qType> biasGradTemp;

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

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true,
                   LayoutMode::NONE, false, TPosition::VECOUT>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopyVGradB1<qType>>>
        vGradMatmul;
};
}  // namespace HstuDenseBackward
#endif
