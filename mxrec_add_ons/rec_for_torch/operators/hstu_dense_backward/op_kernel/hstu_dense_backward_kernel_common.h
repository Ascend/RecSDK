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


#ifndef HSTU_DENSE_BACKWARD_KERNEL_COMMON_H
#define HSTU_DENSE_BACKWARD_KERNEL_COMMON_H

#include <cstdint>
#include <type_traits>
#include <unistd.h>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace HstuDenseBackward {
constexpr int MAX_BATCH_SIZE = 2048;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int ALIGN_16 = 16;

constexpr int DATA_ALIGN_BYTES = 32;
constexpr int VCORE_NUM_IN_ONE_AIC = 2;

constexpr int MID_USE_TIMES = 2;
constexpr int USE_BUFFER_NUM = 2;
constexpr int TWO = 2;

enum class MaskType { MASK_TRIL = 0, MASK_TRIU = 1, MASK_NONE = 2, MASK_CUSTOM = 3 };

struct Args {
    GM_ADDR grad;
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR attnBias;

    GM_ADDR qGrad;
    GM_ADDR kGrad;
    GM_ADDR vGrad;
    GM_ADDR attnBiasGrad;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

__aicore__ inline bool IfMask(const int32_t &maskType, MaskType maskTypeEnum)
{
    return static_cast<int32_t>(maskTypeEnum) == maskType;
}

// For QK and GV
template <typename qType>
__aicore__ inline void CopyQKA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm,
                                int row, int col, int useM, int useK,
                                const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(const_cast<__gm__ void *>(gm)), useM * useK);

    HstuDenseBackwardTilingData *tilingP = reinterpret_cast<HstuDenseBackwardTilingData *>(tilingPtr);
    int64_t headNum = tilingP->headNum;
    int64_t headDim = tilingP->headDim;

    int32_t baseM = tilingP->qkMatmul.baseM;
    int32_t baseN = tilingP->qkMatmul.baseN;
    int32_t baseK = tilingP->qkMatmul.baseK;

    uint16_t alignedUseM = AlignUp(useM, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useM, (uint16_t)useK, 0, (uint16_t)(headNum * headDim), alignedUseM, 1, 0};

    int64_t startIdx = row * baseM * headNum * headDim + col * baseK;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

// For QK and GV
template <typename qType>
__aicore__ inline void CopyQKB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm,
                                int row, int col, int useK, int useN,
                                const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(const_cast<__gm__ void *>(gm)), useN * useK);

    HstuDenseBackwardTilingData *tilingP = reinterpret_cast<HstuDenseBackwardTilingData *>(tilingPtr);
    int64_t headNum = tilingP->headNum;
    int64_t headDim = tilingP->headDim;

    int32_t baseM = tilingP->qkMatmul.baseM;
    int32_t baseN = tilingP->qkMatmul.baseN;
    int32_t baseK = tilingP->qkMatmul.baseK;

    uint16_t alignedUseN = AlignUp(useN, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useN, (uint16_t)useK, 0, (uint16_t)(headNum * headDim), alignedUseN, 1, 0};

    int64_t startIdx = col * baseN * headNum * headDim + row * baseK;
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

// For q_grad
template <typename qType>
__aicore__ inline void CopyQGradA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm,
                                   int row, int col, int useM, int useK,
                                   const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(const_cast<__gm__ void *>(gm)), useM * useK);

    HstuDenseBackwardTilingData *tilingP = reinterpret_cast<HstuDenseBackwardTilingData *>(tilingPtr);
    int64_t biasGradSeqLen = tilingP->biasGradSeqLen;

    int32_t baseM = tilingP->qGradMatmul.baseM;
    int32_t baseN = tilingP->qGradMatmul.baseN;
    int32_t baseK = tilingP->qGradMatmul.baseK;

    uint16_t alignedUseM = AlignUp(useM, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useM, (uint16_t)useK, 0, (uint16_t)biasGradSeqLen, alignedUseM, 1, 0};

    int64_t startIdx = row * baseM * biasGradSeqLen + col * baseK;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

// For k_grad
template <typename qType>
__aicore__ inline void CopyKGradA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm,
                                   int row, int col, int useM, int useK,
                                   const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(const_cast<__gm__ void *>(gm)), useK * useM);

    HstuDenseBackwardTilingData *tilingP = reinterpret_cast<HstuDenseBackwardTilingData *>(tilingPtr);
    int64_t biasGradSeqLen = tilingP->biasGradSeqLen;

    int32_t baseM = tilingP->kGradMatmul.baseM;
    int32_t baseN = tilingP->kGradMatmul.baseN;
    int32_t baseK = tilingP->kGradMatmul.baseK;

    uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useM, 0, (uint16_t)biasGradSeqLen, alignedUseK, 1, 0};

    int64_t startIdx = col * baseK * biasGradSeqLen + row * baseM;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};

template <typename qType>
__aicore__ inline void CopyVGradB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm,
                                   int row, int col, int useK, int useN,
                                   const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType *>(const_cast<__gm__ void *>(gm)), useN * useK);

    HstuDenseBackwardTilingData *tilingP = reinterpret_cast<HstuDenseBackwardTilingData *>(tilingPtr);
    int64_t headNum = tilingP->headNum;
    int64_t headDim = tilingP->headDim;

    int32_t baseM = tilingP->vGradMatmul.baseM;
    int32_t baseN = tilingP->vGradMatmul.baseN;
    int32_t baseK = tilingP->vGradMatmul.baseK;

    uint16_t alignedUseK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param{1, (uint16_t)useK, (uint16_t)useN, 0, (uint16_t)(headNum * headDim), alignedUseK, 1, 0};

    int64_t startIdx = row * baseK * headNum * headDim + col * baseN;
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[startIdx], param);
};



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

} // namespace HstuDenseBackward

#endif