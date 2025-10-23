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

#ifndef HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_V200_H
#define HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_V200_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "hstu_dense_causal_mask.h"

using namespace AscendC;

namespace HstuDenseForward {

constexpr uint32_t MAX_BATCH_SIZE = 2048;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;
constexpr int UB_SIZE = 248 * 1024;  // 248KB
constexpr int ACCU_BLOCK_SINGLE_ELEMENTS = 24;
constexpr int QUEUE_IN_NUM = 1;
constexpr int SPLIT_CORE = 1;
constexpr int VECTOR_ELEM = 5; // half: queIn queMaskIn queTransOut float: quetransIn
constexpr int MATMUL_NUM = 2;

constexpr int VCORE_NUM_IN_ONE_AIC = 1;
constexpr int COMPUTE_PIPE_NUM = 1;
constexpr int TRANS_PIPE_NUM = 1;

constexpr int INVALID_TASK_ID = -1;
struct Args {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR attnBias;
    GM_ADDR attnMask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename qType>
class HstuDenseForwardKernelPattenBsnd {
public:
    __aicore__ inline HstuDenseForwardKernelPattenBsnd() {}
    __aicore__ inline void Init(const Args &args,
                                const HstuDenseForwardTilingData *__restrict tilingDataPtr,
                                TPipe *pipePtr)
    {
        pipe = pipePtr;
        q = args.q;
        k = args.k;
        v = args.v;
        attnBias = args.attnBias;
        attnMask = args.attnMask;
        attnOutput = args.attnOutput;
        workspace = args.workspace;

        // Batch Size
        xDim0 = tilingDataPtr->batchSize;
        // Seq Len
        xDim1 = tilingDataPtr->seqLen;
        // Head Num
        xDim2 = tilingDataPtr->headNum;
        // Embedding Dim
        xDim3 = tilingDataPtr->dim;

        // Tiling
        blockHeight = tilingDataPtr->blockHeight;
        seqBlockNumQk = xDim1 / blockHeight;
        seqBlockNumSV = xDim1 / blockHeight;

        qkTotalBlock = xDim0 * xDim2 * seqBlockNumQk;

        // DataType
        dataTypeBitNum = sizeof(qType);
        dataTypeAlign = DATA_ALIGN_BYTES / sizeof(qType);

        // Ub
        tmpUbSize = tilingDataPtr->tmpUbSize;
        int usedUb = blockHeight * xDim3 * MATMUL_NUM * sizeof(qType) +
                     blockHeight * blockHeight * sizeof(float) + tmpUbSize;
        
        int blockElem = (UB_SIZE - usedUb) / sizeof(qType) / VECTOR_ELEM / USE_QUEUE_NUM / blockHeight * blockHeight;
        vectorScoreUbBlockElem = blockElem;
        accuBlockElem = blockElem;
        
        // attr
        siluScale = tilingDataPtr->siluScale;

        // Gt
        qGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(q));
        kGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(k));
        vGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(v));

        int64_t oneBlockMidElem = blockHeight * blockHeight * COMPUTE_PIPE_NUM;
        int64_t oneCoreMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

        int64_t oneBlockMidTransElem = blockHeight * xDim3 * TRANS_PIPE_NUM;
        int64_t oneCoreTransMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

        attnBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnBias));
        attnMaskGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnMask));

        attnOutputGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput));

        svResultGt.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem + GetBlockIdx() * oneBlockMidTransElem,
            oneCoreTransMidElem);
        // Init pipe
        pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(qType));
        pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(qType));

        pipe->InitBuffer(queTransIn, USE_QUEUE_NUM, accuBlockElem * sizeof(float));
        pipe->InitBuffer(queTransOut, USE_QUEUE_NUM, accuBlockElem * sizeof(qType));

        pipe->InitBuffer(queInputQ, 1, blockHeight * xDim3 * sizeof(qType));
        pipe->InitBuffer(queInputKV, 1,  blockHeight * xDim3 * sizeof(qType));

        pipe->InitBuffer(queTmp, 1,  tmpUbSize);
        pipe->InitBuffer(queAttnScore, 1, blockHeight * blockHeight *sizeof(float));
    }

    __aicore__ inline void CalcuScoreWithQType(LocalTensor<qType>& newInLt,
                                               LocalTensor<qType>& newAttnScore,
                                               LocalTensor<qType>& newInMaskLt,
                                               int64_t thisLen, int64_t thisOffset)
    {
        Add<qType>(newAttnScore[thisOffset], newInLt, newAttnScore[thisOffset], thisLen);
        Silu<qType>(newInLt, newAttnScore[thisOffset], thisLen);
        Muls<qType>(newInLt, newInLt, siluScale, thisLen);
        Mul<qType>(newAttnScore[thisOffset], newInLt, newInMaskLt, thisLen);
    }

    __aicore__ inline void CopyInputGm2Ub(LocalTensor<qType>& inTensor,
                                          const GlobalTensor<qType>& inGlobalTensor,
                                          int64_t height, int64_t width)
    {
        uint16_t blockLen = width / dataTypeAlign;
        uint16_t srcStride = static_cast<uint16_t>((xDim2 - 1) * xDim3 / dataTypeAlign);
        DataCopyParams params {static_cast<uint16_t>(height), blockLen, srcStride, 0};
        DataCopy(inTensor, inGlobalTensor, params);
    }

    __aicore__ inline void VecScoreImpl(int64_t taskId, int64_t attnBiasOffset,
        int64_t maskOffset, float scale, uint32_t causalMask, uint32_t m, uint32_t n)
    {
        if (taskId == INVALID_TASK_ID) {
            return;
        }

        // Ub
        uint16_t copyLen = blockHeight * sizeof(qType) / DATA_ALIGN_BYTES;
        uint16_t srcStride = (xDim1 - blockHeight) * sizeof(qType) / DATA_ALIGN_BYTES;

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
            int64_t thisAttnBaisOffset = attnBiasOffset + thisOffset / blockHeight * xDim1;
            int64_t thisAttnMaskOffset = maskOffset + thisOffset / blockHeight * xDim1;

            LocalTensor<qType> inLt = queMaskIn.AllocTensor<qType>();
            LocalTensor<qType> inMaskLt = queIn.AllocTensor<qType>();

            DataCopyParams copyParms = {(uint16_t)(thisLen / blockHeight), copyLen, srcStride, 0};

            DataCopy(inLt, attnBiasGt[thisAttnBaisOffset], copyParms);
            DataCopy(inMaskLt, attnMaskGt[thisAttnMaskOffset], copyParms);

            queIn.EnQue(inLt);
            queMaskIn.EnQue(inMaskLt);

            LocalTensor<qType> newInLtQtype = queIn.DeQue<qType>();
            LocalTensor<qType> newInMaskLtQtype = queMaskIn.DeQue<qType>();

            CalcuScoreWithQType(newInLtQtype, attnTmp, newInMaskLtQtype, thisLen, thisOffset);

            queIn.FreeTensor(newInLtQtype);
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
        CopyInputGm2Ub(inQTensor, qGt[qOffset], blockHeight, xDim3);
        queInputQ.EnQue(inQTensor);
    }

    __aicore__ inline void DoFreeQImpl()
    {
        LocalTensor<qType> inQTensor = queInputQ.DeQue<qType>();
        queInputQ.FreeTensor(inQTensor);
    }

    __aicore__ inline void DoQkMatmulImpl(int64_t qOffset, int64_t kOffset, uint32_t taskId)
    {
        if (taskId == INVALID_TASK_ID) {
            return;
        }

        LocalTensor<qType> inQTensor = queInputQ.DeQue<qType>();

        LocalTensor<qType> inKTensor = queInputKV.AllocTensor<qType>();
        CopyInputGm2Ub(inKTensor, kGt[kOffset], blockHeight, xDim3);
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

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, int isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k)
    {
        if (taskId == INVALID_TASK_ID) {
            return;
        }
        LocalTensor<uint8_t> tmpTensor = queTmp.AllocTensor<uint8_t>();

        auto attnScoreTensor = queAttnScore.DeQue<qType>();
        svMatmul.SetLocalWorkspace(tmpTensor);
        svMatmul.SetTensorA(attnScoreTensor);

        LocalTensor<qType> inVTensor = queInputKV.AllocTensor<qType>();
        CopyInputGm2Ub(inVTensor, vGt[vOffset], blockHeight, xDim3);
        queInputKV.EnQue(inVTensor);
        inVTensor = queInputKV.DeQue<qType>();
        svMatmul.SetTensorB(inVTensor);

        if (isAtomicAdd == 0) {
            // Override
            svMatmul.IterateAll(svResultGt, 0);
        } else {
            // Automic Add
            svMatmul.IterateAll(svResultGt, 1);
        }
        svMatmul.End();

        queTmp.FreeTensor(tmpTensor);
        queAttnScore.FreeTensor(attnScoreTensor);
        queInputKV.FreeTensor(inVTensor);
    }

    __aicore__ inline void DoTransSvImpl(int64_t transTaskId, int64_t outStartOffset, uint32_t m)
    {
        if (transTaskId == INVALID_TASK_ID) {
            return;
        }

        int64_t total = blockHeight * xDim3;
        int64_t remain = total;
        uint16_t copyLen = xDim3 * sizeof(qType) / DATA_ALIGN_BYTES;
        uint16_t distStride = (xDim2 * xDim3 - xDim3) * sizeof(qType) / DATA_ALIGN_BYTES;

        while (remain > 0) {
            LocalTensor<float> inLt = queTransIn.AllocTensor<float>();
            LocalTensor<qType> outLt = queTransOut.AllocTensor<qType>();
            int64_t thisLen = accuBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = (total - remain);
            DataCopy(inLt, svResultGt[kThisOffset], thisLen);

            queTransIn.EnQue(inLt);

            inLt = queTransIn.DeQue<float>();

            if (std::is_same<qType, float>::value) {
                DataCopy(outLt.template ReinterpretCast<float>(), inLt, thisLen);
            } else {
                Cast(outLt, inLt, RoundMode::CAST_NONE, thisLen);
            }

            queTransOut.EnQue(outLt);
            
            LocalTensor<qType> newOutLt = queTransOut.DeQue<qType>();

            DataCopyParams copyParms = {(uint16_t)(thisLen / xDim3), copyLen, 0, distStride};
            int64_t thisLineOffset = (total - remain) / xDim3;
            int64_t outOffset = outStartOffset + thisLineOffset * xDim2 * xDim3;
            DataCopy(attnOutputGt[outOffset], newOutLt, copyParms);

            remain = remain - thisLen;
            queTransIn.FreeTensor(inLt);
            queTransOut.FreeTensor(newOutLt);
        }
    }
    // GM_ADDR
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR attnBias;
    GM_ADDR attnMask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;

    // Shape
    int64_t xDim0;
    int64_t xDim1;
    int64_t xDim2;
    int64_t xDim3;

    // Tiling
    int64_t blockHeight;
    int64_t seqBlockNumQk;
    int64_t seqBlockNumSV;

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
    CausalMaskT maskType;

    // DataType
    int dataTypeBitNum;
    int dataTypeAlign;

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn;

    TQue<TPosition::VECIN, USE_QUEUE_NUM> queTransIn;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queTransOut;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>queInputQ;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>queInputKV;

    TQue<TPosition::VECCALC, 1> queTmp;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>queAttnScore;
    // Gt
    GlobalTensor<qType> qGt;
    GlobalTensor<qType> kGt;
    GlobalTensor<qType> vGt;
    GlobalTensor<qType> attnOutputGt;
    GlobalTensor<qType> attnBiasGt;
    GlobalTensor<qType> attnMaskGt;
    GlobalTensor<float> svResultGt;

    GlobalTensor<qType> attenMaskGt;

    // Matmul
    using MatmulTypeQ = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypeK = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, true>;
    using MatmulTypeQK = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, float, false>;
    matmul::Matmul<MatmulTypeQ, MatmulTypeK, MatmulTypeQK> qkMatmul;

    using MatmulTypeS = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypeV = matmul::MatmulType<TPosition::VECOUT, CubeFormat::ND, qType, false>;
    using MatmulTypeSV = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    matmul::Matmul<MatmulTypeS, MatmulTypeV, MatmulTypeSV> svMatmul;
};
}  // namespace HstuDenseForward
#endif
