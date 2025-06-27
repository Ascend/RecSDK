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

#ifndef HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_V200_FUXI_H
#define HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_V200_FUXI_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace HstuDenseForwardFuxi {

constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int UB_SIZE = 248 * 1024;  // 248KB
constexpr int SPLIT_CORE = 1;
constexpr int VECTOR_ELEM = 5; // half: queIn queMaskIn queTransOut float: quetransIn
constexpr int MATMUL_NUM = 2;

constexpr int VCORE_NUM_IN_ONE_AIC = 1;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 1;

constexpr int OUTPUT_DIM3_TIMES_1 = 1;
constexpr int OUTPUT_DIM3_TIMES_3 = 3;

constexpr int SV_WORKSPACE_IDX = 0;
constexpr int TV_WORKSPACE_IDX = 1;
constexpr int PV_WORKSPACE_IDX = 2;

constexpr int INVALID_TASK_ID = -1;
struct Args {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR timestampBias;
    GM_ADDR positionBias;
    GM_ADDR attnMask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename qType>
class HstuDenseKernelPattenBsndV200Fuxi {
public:
    __aicore__ inline HstuDenseKernelPattenBsndV200Fuxi() {}

    __aicore__ inline void InitGlobalBuffer()
    {
        qGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(q));
        kGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(k));
        vGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(v));

        int64_t oneBlockSize = blockHeight * dim;
        int64_t oneBlockMidElem = oneBlockSize * COMPUTE_PIPE_NUM;
        int64_t oneCoreMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

        if (enableBias) {
            timestampBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(timestampBias));
            positionBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(positionBias));
        }
        attnMaskGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnMask));

        attnOutputGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput));

        svResultGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) +
            GetBlockIdx() * oneBlockMidElem + SV_WORKSPACE_IDX * oneBlockSize, oneBlockSize);
        if (enableBias) {
            tvResultGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) +
                GetBlockIdx() * oneBlockMidElem + TV_WORKSPACE_IDX * oneBlockSize, oneBlockSize);
            pvResultGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) +
                GetBlockIdx() * oneBlockMidElem + PV_WORKSPACE_IDX * oneBlockSize, oneBlockSize);
        }
    }

    __aicore__ inline void Init(const Args &args,
                                const HstuDenseForwardFuxiTilingData *__restrict tilingDataPtr,
                                TPipe *pipePtr)
    {
        pipe = pipePtr;
        q = args.q;
        k = args.k;
        v = args.v;
        timestampBias = args.timestampBias;
        positionBias = args.positionBias;
        attnMask = args.attnMask;
        attnOutput = args.attnOutput;
        workspace = args.workspace;

        // Batch Size
        batchSize = tilingDataPtr->batchSize;
        // Seq Len
        seqLen = tilingDataPtr->seqLen;
        // Head Num
        numHead = tilingDataPtr->headNum;
        // Embedding Dim
        dim = tilingDataPtr->dim;

        // Tiling
        blockHeight = tilingDataPtr->blockHeight;
        seqBlockNumQk = seqLen / blockHeight;
        seqBlockNumSV = seqLen / blockHeight;
        enableBias = tilingDataPtr->enableBias;

        qkTotalBlock = batchSize * numHead * seqBlockNumQk;

        // DataType
        dataTypeBitNum = sizeof(qType);
        dataTypeAlign = DATA_ALIGN_BYTES / sizeof(qType);

        // Ub
        tmpUbSize = tilingDataPtr->tmpUbSize;
        int usedUb = blockHeight * dim * MATMUL_NUM * sizeof(qType) +     // queInputQ + queInputKV
                     blockHeight * blockHeight * sizeof(float) +          // queAttnScore
                     blockHeight * blockHeight * sizeof(qType) * 2 +      // 2: queTimeBiasIn + quePosBiasIn
                     tmpUbSize;
        
        int blockElem = (UB_SIZE - usedUb) / sizeof(qType) / VECTOR_ELEM / USE_QUEUE_NUM / blockHeight * blockHeight;
        vectorScoreUbBlockElem = blockElem;
        accuBlockElem = blockElem;
        
        // attr
        siluScale = tilingDataPtr->siluScale;

        InitGlobalBuffer();

        // Init pipe
        pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(qType));
        pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(qType));

        pipe->InitBuffer(queTransIn, USE_QUEUE_NUM, accuBlockElem * sizeof(float));
        pipe->InitBuffer(queTransOut, USE_QUEUE_NUM, accuBlockElem * sizeof(qType));

        pipe->InitBuffer(queInputQ, USE_QUEUE_NUM, blockHeight * dim * sizeof(qType));
        pipe->InitBuffer(queInputKV, USE_QUEUE_NUM,  blockHeight * dim * sizeof(qType));

        pipe->InitBuffer(queTimeBiasIn, USE_QUEUE_NUM, blockHeight * blockHeight *sizeof(qType));
        pipe->InitBuffer(quePosBiasIn, USE_QUEUE_NUM, blockHeight * blockHeight *sizeof(qType));

        pipe->InitBuffer(queTmp, USE_QUEUE_NUM,  tmpUbSize);
        pipe->InitBuffer(queAttnScore, USE_QUEUE_NUM, blockHeight * blockHeight *sizeof(float));
    }

    __aicore__ inline void CalcuScoreWithQType(LocalTensor<qType>& newAttnScore,
                                               LocalTensor<qType>& newInMaskLt,
                                               int64_t thisLen, int64_t thisOffset)
    {
        LocalTensor<qType> tempTensor = queIn.AllocTensor<qType>();
        Silu<qType>(tempTensor, newAttnScore[thisOffset], thisLen);
        Muls<qType>(tempTensor, tempTensor, siluScale, thisLen);
        Mul<qType>(newAttnScore[thisOffset], tempTensor, newInMaskLt, thisLen);
        queIn.FreeTensor(tempTensor);
    }

    __aicore__ inline void CopyInputGm2Ub(LocalTensor<qType>& inTensor,
                                          const GlobalTensor<qType>& inGlobalTensor,
                                          int64_t height, int64_t width)
    {
        uint16_t blockLen = width / dataTypeAlign;
        uint16_t srcStride = static_cast<uint16_t>((numHead - 1) * dim / dataTypeAlign);
        DataCopyParams params {static_cast<uint16_t>(height), blockLen, srcStride, 0};
        DataCopy(inTensor, inGlobalTensor, params);
    }

    __aicore__ inline void VecScoreImpl(int64_t maskOffset, float scale)
    {
        // Ub
        uint16_t copyLen = blockHeight * sizeof(qType) / DATA_ALIGN_BYTES;
        uint16_t srcStride = (seqLen - blockHeight) * sizeof(qType) / DATA_ALIGN_BYTES;

        int64_t total = blockHeight * blockHeight;
        int64_t remain = total;

        LocalTensor<float> attnScoreLt = queAttnScore.DeQue<float>();
        LocalTensor<qType> attnTmp = queTmp.AllocTensor<qType>();

        Cast(attnTmp, attnScoreLt, RoundMode::CAST_NONE, total);
        int64_t thisLen = vectorScoreUbBlockElem;
        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t thisOffset = total - remain;
            int64_t thisAttnMaskOffset = maskOffset + thisOffset / blockHeight * seqLen;

            LocalTensor<qType> inMaskLt = queMaskIn.AllocTensor<qType>();

            DataCopyParams copyParms = {(uint16_t)(thisLen / blockHeight), copyLen, srcStride, 0};

            DataCopy(inMaskLt, attnMaskGt[thisAttnMaskOffset], copyParms);

            queMaskIn.EnQue(inMaskLt);

            LocalTensor<qType> newInMaskLtQtype = queMaskIn.DeQue<qType>();

            CalcuScoreWithQType(attnTmp, newInMaskLtQtype, thisLen, thisOffset);

            queMaskIn.FreeTensor(newInMaskLtQtype);

            remain = remain - thisLen;
        }
        queAttnScore.FreeTensor(attnScoreLt);

        auto score = queAttnScore.AllocTensor<qType>();
        DataCopy(score, attnTmp, total);
        pipe_barrier(PIPE_ALL);
        queTmp.FreeTensor(attnTmp);
        queAttnScore.EnQue(score);
    }

    __aicore__ inline void DoCopyQImpl(int64_t qOffset)
    {
        LocalTensor<qType> inQTensor = queInputQ.AllocTensor<qType>();
        CopyInputGm2Ub(inQTensor, qGt[qOffset], blockHeight, dim);
        queInputQ.EnQue(inQTensor);
    }

    __aicore__ inline void DoFreeQImpl()
    {
        LocalTensor<qType> inQTensor = queInputQ.DeQue<qType>();
        queInputQ.FreeTensor(inQTensor);
    }

    __aicore__ inline void DoQkMatmulImpl(int64_t qOffset, int64_t kOffset)
    {
        LocalTensor<qType> inQTensor = queInputQ.DeQue<qType>();

        LocalTensor<qType> inKTensor = queInputKV.AllocTensor<qType>();
        CopyInputGm2Ub(inKTensor, kGt[kOffset], blockHeight, dim);
        queInputKV.EnQue(inKTensor);
        inKTensor = queInputKV.DeQue<qType>();

        LocalTensor<uint8_t> tmpTensor = queTmp.AllocTensor<uint8_t>();
        qkMatmul.SetLocalWorkspace(tmpTensor);

        qkMatmul.SetTensorA(inQTensor);
        qkMatmul.SetTensorB(inKTensor, true);
        
        LocalTensor<float> attnScoreTensor = queAttnScore.AllocTensor<float>();

        qkMatmul.IterateAll(attnScoreTensor);
        qkMatmul.End();
        queAttnScore.EnQue(attnScoreTensor);
        queInputKV.FreeTensor(inKTensor);
        queInputQ.EnQue(inQTensor);
        queTmp.FreeTensor(tmpTensor);
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint8_t enAtomicAdd)
    {
        LocalTensor<uint8_t> tmpTensor = queTmp.AllocTensor<uint8_t>();

        auto attnScoreTensor = queAttnScore.DeQue<qType>();
        svMatmul.SetLocalWorkspace(tmpTensor);
        svMatmul.SetTensorA(attnScoreTensor);

        LocalTensor<qType> inVTensor = queInputKV.AllocTensor<qType>();
        CopyInputGm2Ub(inVTensor, vGt[vOffset], blockHeight, dim);
        queInputKV.EnQue(inVTensor);
        inVTensor = queInputKV.DeQue<qType>();
        svMatmul.SetTensorB(inVTensor);

        svMatmul.IterateAll(svResultGt, enAtomicAdd);
        svMatmul.End();

        queTmp.FreeTensor(tmpTensor);
        queAttnScore.FreeTensor(attnScoreTensor);
        queInputKV.FreeTensor(inVTensor);
    }

    __aicore__ inline void DoTvMatmulImpl(int64_t vOffset, uint8_t enAtomicAdd)
    {
        LocalTensor<uint8_t> tmpTensor = queTmp.AllocTensor<uint8_t>();

        auto timeBiasTensor = queTimeBiasIn.DeQue<qType>();
        tvMatmul.SetLocalWorkspace(tmpTensor);
        tvMatmul.SetTensorA(timeBiasTensor);

        LocalTensor<qType> inVTensor = queInputKV.AllocTensor<qType>();
        CopyInputGm2Ub(inVTensor, vGt[vOffset], blockHeight, dim);
        queInputKV.EnQue(inVTensor);
        inVTensor = queInputKV.DeQue<qType>();
        tvMatmul.SetTensorB(inVTensor);

        tvMatmul.IterateAll(tvResultGt, enAtomicAdd);
        tvMatmul.End();

        queTmp.FreeTensor(tmpTensor);
        queTimeBiasIn.FreeTensor(timeBiasTensor);
        queInputKV.FreeTensor(inVTensor);
    }

    __aicore__ inline void DoPvMatmulImpl(int64_t vOffset, uint8_t enAtomicAdd)
    {
        LocalTensor<uint8_t> tmpTensor = queTmp.AllocTensor<uint8_t>();

        auto posBiasTensor = quePosBiasIn.DeQue<qType>();
        pvMatmul.SetLocalWorkspace(tmpTensor);
        pvMatmul.SetTensorA(posBiasTensor);

        LocalTensor<qType> inVTensor = queInputKV.AllocTensor<qType>();
        CopyInputGm2Ub(inVTensor, vGt[vOffset], blockHeight, dim);
        queInputKV.EnQue(inVTensor);
        inVTensor = queInputKV.DeQue<qType>();
        pvMatmul.SetTensorB(inVTensor);

        pvMatmul.IterateAll(pvResultGt, enAtomicAdd);
        pvMatmul.End();

        queTmp.FreeTensor(tmpTensor);
        quePosBiasIn.FreeTensor(posBiasTensor);
        queInputKV.FreeTensor(inVTensor);
    }

    __aicore__ inline void DoTransImpl(const GlobalTensor<float>& dstGt, int64_t kThisOffset, int64_t thisLen,
        int64_t outOffset)
    {
        uint16_t copyLen = dim * sizeof(qType) / DATA_ALIGN_BYTES;
        uint16_t strideLen = enableBias ? (OUTPUT_DIM3_TIMES_3 * numHead * dim) : (numHead * dim);
        uint16_t distStride = (strideLen - dim) * sizeof(qType) / DATA_ALIGN_BYTES;

        LocalTensor<float> inLt = queTransIn.AllocTensor<float>();
        DataCopy(inLt, dstGt[kThisOffset], thisLen);
        queTransIn.EnQue(inLt);
        inLt = queTransIn.DeQue<float>();

        LocalTensor<qType> outLt = queTransOut.AllocTensor<qType>();
        if (std::is_same<qType, float>::value) {
            DataCopy(outLt.template ReinterpretCast<float>(), inLt, thisLen);
        } else {
            Cast(outLt, inLt, RoundMode::CAST_NONE, thisLen);
        }
        queTransIn.FreeTensor(inLt);

        queTransOut.EnQue(outLt);
        LocalTensor<qType> newOutLt = queTransOut.DeQue<qType>();

        DataCopyParams copyParms = {(uint16_t)(thisLen / dim), copyLen, 0, distStride};
        DataCopy(attnOutputGt[outOffset], newOutLt, copyParms);
        queTransOut.FreeTensor(newOutLt);
    }

    __aicore__ inline void DoTransResultImpl(int64_t outStartOffset)
    {
        int64_t total = blockHeight * dim;
        int64_t remain = total;
        
        while (remain > 0) {
            int64_t thisLen = accuBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = (total - remain);
            int64_t thisLineOffset = kThisOffset / dim;

            int64_t svOutOffset = outStartOffset + thisLineOffset * numHead * dim;
            DoTransImpl(svResultGt, kThisOffset, thisLen, svOutOffset);

            if (enableBias) {
                int64_t tvOutOffset = svOutOffset + numHead * dim;
                DoTransImpl(tvResultGt, kThisOffset, thisLen, tvOutOffset);

                int64_t pvOutOffset = tvOutOffset + numHead * dim;
                DoTransImpl(pvResultGt, kThisOffset, thisLen, pvOutOffset);
            }

            remain = remain - thisLen;
        }
    }

    __aicore__ inline void DoBiasMaskImpl(int64_t maskOffset, int64_t timestampOffset, int64_t positionOffset)
    {
        uint16_t maskCopyLen = blockHeight * sizeof(qType) / DATA_ALIGN_BYTES;
        uint16_t maskSrcStride = (seqLen - blockHeight) * sizeof(qType) / DATA_ALIGN_BYTES;

        int64_t totalLen = blockHeight * blockHeight;
        int64_t remain = totalLen;
        int64_t thisLen = vectorScoreUbBlockElem;

        LocalTensor<qType> inTimeBiasLt = queTimeBiasIn.AllocTensor<qType>();
        LocalTensor<qType> inPosBiasLt = quePosBiasIn.AllocTensor<qType>();

        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t thisOffset = totalLen - remain;
            int64_t thisMaskOffset = maskOffset + thisOffset / blockHeight * seqLen;
            int64_t thisTimestampOffset = timestampOffset + thisOffset / blockHeight * seqLen;
            int64_t thisPositionOffset = positionOffset + thisOffset / blockHeight * seqLen;

            DataCopyParams copyParams = {static_cast<uint16_t>(thisLen / blockHeight), maskCopyLen, maskSrcStride, 0};

            LocalTensor<qType> inMaskLt = queMaskIn.AllocTensor<qType>();
            DataCopy(inMaskLt, attnMaskGt[thisMaskOffset], copyParams);
            queMaskIn.EnQue(inMaskLt);

            // timestampBias
            LocalTensor<qType> tmpTsBiasLt = queIn.AllocTensor<qType>();
            DataCopy(tmpTsBiasLt, timestampBiasGt[thisTimestampOffset], copyParams);
            queIn.EnQue(tmpTsBiasLt);

            inMaskLt = queMaskIn.DeQue<qType>();
            tmpTsBiasLt = queIn.DeQue<qType>();
            Mul<qType>(inTimeBiasLt[thisOffset], tmpTsBiasLt, inMaskLt, thisLen);
            queIn.FreeTensor(tmpTsBiasLt);

            // positionBias
            LocalTensor<qType> tmpPosBiasLt = queTransIn.AllocTensor<qType>();
            DataCopy(tmpPosBiasLt, positionBiasGt[thisPositionOffset], copyParams);
            queTransIn.EnQue(tmpPosBiasLt);

            tmpPosBiasLt = queTransIn.DeQue<qType>();
            Mul<qType>(inPosBiasLt[thisOffset], tmpPosBiasLt, inMaskLt, thisLen);
            queTransIn.FreeTensor(tmpPosBiasLt);
            queMaskIn.FreeTensor(inMaskLt);

            remain = remain - thisLen;
        }
        queTimeBiasIn.EnQue(inTimeBiasLt);
        quePosBiasIn.EnQue(inPosBiasLt);
    }

    // GM_ADDR
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR timestampBias;
    GM_ADDR positionBias;
    GM_ADDR attnMask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;

    // Shape
    int64_t batchSize;
    int64_t seqLen;
    int64_t numHead;
    int64_t dim;

    // Tiling
    int64_t blockHeight;
    int64_t seqBlockNumQk;
    int64_t seqBlockNumSV;
    bool enableBias;

    // Tiling-QK
    int64_t qkTotalBlock;

    // Ub
    int64_t ub;
    int64_t vectorScoreUbBlockElem;
    int64_t accuBlockElem;
    int32_t tmpUbSize;

    // split
    int64_t blockSplitNum;

    // Attr
    float siluScale;

    // DataType
    int dataTypeBitNum;
    int dataTypeAlign;

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn;

    TQue<TPosition::VECIN, USE_QUEUE_NUM> queTimeBiasIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> quePosBiasIn;

    TQue<TPosition::VECIN, USE_QUEUE_NUM> queTransIn;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queTransOut;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queInputQ;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queInputKV;

    TQue<TPosition::VECCALC, 1> queTmp;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queAttnScore;
    // Gt
    GlobalTensor<qType> qGt;
    GlobalTensor<qType> kGt;
    GlobalTensor<qType> vGt;
    GlobalTensor<qType> attnOutputGt;
    GlobalTensor<qType> timestampBiasGt;
    GlobalTensor<qType> positionBiasGt;
    GlobalTensor<qType> attnMaskGt;
    GlobalTensor<float> svResultGt;
    GlobalTensor<float> tvResultGt;
    GlobalTensor<float> pvResultGt;

    GlobalTensor<qType> attenMaskGt;

    // Matmul
    using MatmulTypeQ = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypeK = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, true>;
    using MatmulTypeQK = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, float, false>;
    matmul::Matmul<MatmulTypeQ, MatmulTypeK, MatmulTypeQK> qkMatmul;

    using MatmulTypeV = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;

    using MatmulTypeS = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypeSV = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    matmul::Matmul<MatmulTypeS, MatmulTypeV, MatmulTypeSV> svMatmul;

    using MatmulTypeT = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypeTV = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    matmul::Matmul<MatmulTypeT, MatmulTypeV, MatmulTypeTV> tvMatmul;

    using MatmulTypeP = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypePV = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    matmul::Matmul<MatmulTypeP, MatmulTypeV, MatmulTypePV> pvMatmul;
};
}  // namespace HstuDenseForwardFuxi
#endif
