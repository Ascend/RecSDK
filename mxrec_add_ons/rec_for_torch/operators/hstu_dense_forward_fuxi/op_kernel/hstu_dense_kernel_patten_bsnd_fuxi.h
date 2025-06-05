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

#ifndef HSTU_DENSE_KERNEL_PATTEN_BSND_FUXI_H
#define HSTU_DENSE_KERNEL_PATTEN_BSND_FUXI_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace HstuDenseForwardFuxi {

constexpr uint32_t MAX_BATCH_SIZE = 512;
constexpr int USE_QUEUE_NUM = 1;
constexpr int DATA_ALIGN_BYTES = 32;
constexpr int VEC_PER_PROCESS = 32;
constexpr int MAX_INDICS_ONE_BLOCK = 100;
constexpr int UB_SIZE = 170 * 1024;  // 170KB
constexpr int QUEUE_IN_NUM = 2;
constexpr int SPLIT_CORE = 2;
constexpr int ALIGN_16 = 16;

constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;

constexpr int SV_WORKSPACE_IDX = 0;
constexpr int TV_WORKSPACE_IDX = 1;
constexpr int PV_WORKSPACE_IDX = 2;

constexpr int ATTN_WORKSPACE_IDX = 0;
constexpr int TS_WORKSPACE_IDX = 1;
constexpr int POS_WORKSPACE_IDX = 2;

constexpr int OUTPUT_DIM2_TIMES3 = 3;

constexpr int INVALID_TASK_ID = -1;

struct Args {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR timestampBias;
    GM_ADDR positionBias;
    GM_ADDR mask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

enum class CausalMaskT {
    MASK_TRIL = 0,          // 下三角
    MASK_TRIU,              // 上三角
    MASK_NONE,              // 不使能mask
    MASK_CUSTOME,           // 用户自定义mask
};

template<typename qType, CausalMaskT maskType>
__aicore__ inline void DoCausalMask(LocalTensor<qType>& inMaskLt, int64_t maskOffset, int64_t maskLens,
    int64_t maskStride, int64_t repeatTimes, qType value)
{
    if constexpr (maskType == CausalMaskT::MASK_TRIL) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            int64_t thisIndexMask = maskOffset + i + 1;
            Duplicate<qType>(inMaskLt[i * maskStride], value, thisIndexMask);
        }
    } else if constexpr (maskType == CausalMaskT::MASK_TRIU) {
        ASCENDC_ASSERT((false), "DoCausalMask triu is unreadlized");
    } else if constexpr (maskType == CausalMaskT::MASK_NONE) {
        Duplicate<qType>(inMaskLt, 0, maskLens);
        for (int i = 0; i < repeatTimes; i++) {
            Duplicate<qType>(inMaskLt[i * maskStride], value, maskOffset);
        }
    } else {
        ASCENDC_ASSERT((false), "DoCausalMask custom is unreadlized");
    }
}


template <typename qType>
__aicore__ inline void CopyQKA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col, int useM,
                                int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);
    int blockLen = useM * useK;

    HstuDenseForwardFuxiTilingData* tilingP = reinterpret_cast<HstuDenseForwardFuxiTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseM = tilingP->qkMatmul.baseM;
    int32_t baseK = tilingP->qkMatmul.baseK;

    auto alignOfM = AlignUp(useM, ALIGN_16);
    Nd2NzParams param = {
        1, (uint16_t)useM, (uint16_t)useK, 0, static_cast<uint16_t>(dim * headNum), (uint16_t)alignOfM, 1, 0
    };

    int64_t offsetOfGt = row * dim * headNum * baseM + col * baseK;
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType>
__aicore__ inline void CopyQKB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseForwardFuxiTilingData* tilingP = reinterpret_cast<HstuDenseForwardFuxiTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseN = tilingP->qkMatmul.baseN;
    int32_t baseK = tilingP->qkMatmul.baseK;

    auto alignOfN = AlignUp(useN, ALIGN_16);
    Nd2NzParams param = {
        1, (uint16_t)useN, (uint16_t)useK, 0, static_cast<uint16_t>(dim * headNum), (uint16_t)alignOfN, 1, 0
    };

    int64_t offsetOfGt = col * dim * headNum * baseN + row * baseK;
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType>
__aicore__ inline void CopySVB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseForwardFuxiTilingData* tilingP = reinterpret_cast<HstuDenseForwardFuxiTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseN = tilingP->svMatmul.baseN;
    int32_t baseK = tilingP->svMatmul.baseK;
    auto alignOfK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param = {
        1, (uint16_t)useK, (uint16_t)useN, 0, static_cast<uint16_t>(dim * headNum), (uint16_t)alignOfK, 1, 0
    };

    int64_t offsetOfGt = row * dim * headNum * baseK + col * baseN;
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType, int ElementOfBlock = DATA_ALIGN_BYTES / sizeof(qType)>
class HstuDenseKernelPattenBsndFuxi {
public:
    __aicore__ inline HstuDenseKernelPattenBsndFuxi() {}
    __aicore__ inline void Init(const Args &args, const HstuDenseForwardFuxiTilingData *__restrict tilingDataPtr,
                                TPipe *pipePtr)
    {
        pipe = pipePtr;
        q = args.q;
        k = args.k;
        v = args.v;
        timestampBias = args.timestampBias;
        positionBias = args.positionBias;
        mask = args.mask;
        attnOutput = args.attnOutput;
        workspace = args.workspace;

        // Batch Size
        batchSize = tilingDataPtr->batchSize;
        // Seq Len
        seqLen = tilingDataPtr->seqLen;
        // Head Num
        headNum = tilingDataPtr->headNum;
        // Embedding Dim
        headDim = tilingDataPtr->dim;

        // Tiling
        blockHeight = tilingDataPtr->blockHeight;

        // Ub
        vectorScoreUbBlockElem = VEC_PER_PROCESS * blockHeight / USE_QUEUE_NUM;

        // attr
        siluScale = tilingDataPtr->siluScale;
        maskType = static_cast<CausalMaskT>(tilingDataPtr->maskType);
        enableBias = (tilingDataPtr->enableBias == 1);

        eventIdV2MTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);

        // Gt
        qGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(q));
        kGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(k));
        vGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(v));

        int64_t oneBlockLen = blockHeight * blockHeight * COMPUTE_PIPE_NUM;
        int64_t oneBlockMidElem = oneBlockLen * OUTPUT_DIM2_TIMES3;
        int64_t oneCoreMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

        int64_t oneBlockTransLen = blockHeight * headDim * TRANS_PIPE_NUM;
        int64_t oneBlockMidTransElem = oneBlockTransLen * OUTPUT_DIM2_TIMES3;
        int64_t oneCoreTransMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;

        if (enableBias) {
            timestampBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(timestampBias));
            positionBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(positionBias));
        }

        if (maskType == CausalMaskT::MASK_CUSTOME) {
            attnMaskGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(mask));
        }

        attnOutputGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput));

        attnScoreGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(workspace) +
            GetBlockIdx() * oneBlockMidElem + ATTN_WORKSPACE_IDX * oneBlockLen, oneBlockLen);
        if (enableBias) {
            tsTempGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(workspace) +
                GetBlockIdx() * oneBlockMidElem + TS_WORKSPACE_IDX * oneBlockLen, oneBlockLen);
            posTempGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(workspace) +
                GetBlockIdx() * oneBlockMidElem + POS_WORKSPACE_IDX * oneBlockLen, oneBlockLen);
        }

        svResultGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem +
            GetBlockIdx() * oneBlockMidTransElem + SV_WORKSPACE_IDX * oneBlockTransLen,
            oneBlockTransLen);
        if (enableBias) {
            tvResultGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem +
                GetBlockIdx() * oneBlockMidTransElem + TV_WORKSPACE_IDX * oneBlockTransLen,
                oneBlockTransLen);
            pvResultGt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem +
                GetBlockIdx() * oneBlockMidTransElem + PV_WORKSPACE_IDX * oneBlockTransLen,
                oneBlockTransLen);
        }

        // Init pipe total 32K * 5 = 160K
        pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(tmpBuff, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(biasIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queOut, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
    }

    __aicore__ inline void CastQtype2Float(LocalTensor<float> distTensor, LocalTensor<qType> srcTensor,
                                           LocalTensor<qType> midTensor, int64_t len)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            DataCopy<qType>(midTensor, srcTensor, len);
            Cast(distTensor, midTensor, RoundMode::CAST_NONE, len);
        }
    }

    __aicore__ inline void CastFloat2Qtype(LocalTensor<qType>& distTensor, LocalTensor<float>& srcTensor,
                                           LocalTensor<float>& midTensor, int64_t len)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            DataCopy<float>(midTensor, srcTensor, len);
            Cast(distTensor, midTensor, RoundMode::CAST_RINT, len);
        }
    }

    __aicore__ inline void WaitQkMatmul()
    {
        qkMatmul.WaitIterateAll();
        qkMatmul.End();
    }

    __aicore__ inline void WaitSvMatmul()
    {
        svMatmul.WaitIterateAll();
        svMatmul.End();
    }

    __aicore__ inline void WaitTvMatmul()
    {
        tvMatmul.WaitIterateAll();
        tvMatmul.End();
    }

    __aicore__ inline void WaitPvMatmul()
    {
        pvMatmul.WaitIterateAll();
        pvMatmul.End();
    }

    __aicore__ inline void DoMaskOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& newOutLt,
        int64_t thisLen,
        bool needMask,
        float scale)
    {
        if (maskType != CausalMaskT::MASK_NONE) {
            queMaskIn.DeQue();
        }

        if (needMask) {
            if (maskType == CausalMaskT::MASK_CUSTOME) {
                inMaskLtFp32 = inMaskLt.template ReinterpretCast<float>();
                CastQtype2Float(inMaskLtFp32, inMaskLt, tmpLt, thisLen);
                Muls<float>(inMaskLtFp32, inMaskLtFp32, scale, thisLen);
            }
            Mul<float>(newOutLt, newOutLt, inMaskLtFp32, thisLen);
        } else {
            Muls<float>(newOutLt, newOutLt, scale, thisLen);
        }

        if (maskType != CausalMaskT::MASK_NONE) {
            queMaskIn.FreeTensor(inMaskLtFp32);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& tmpLtFp32,
        bool needMask,
        int64_t thisLen,
        float scale)
    {
        queIn.DeQue();
        auto newInLt = inLt.template ReinterpretCast<float>();
        CastQtype2Float(newInLt, inLt, tmpLt, thisLen);

        auto outLt = queOut.AllocTensor<qType>();
        auto newOutLt = outLt.template ReinterpretCast<float>();
        Silu<float>(newOutLt, newInLt, thisLen);

        queIn.FreeTensor(inLt);
        DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, newOutLt, thisLen, needMask, scale);
        CastFloat2Qtype(outLt, newOutLt, tmpLtFp32, thisLen);
        queOut.EnQue(outLt);
    }

    __aicore__ inline void DataCopyMayPad(
        const LocalTensor<qType>& lt, GlobalTensor<qType>& gt, uint16_t copyBlock, uint32_t blockLen,
        int64_t offset)
    {
        bool align = false;
        uint16_t alignOfN = AlignUp(blockLen, ElementOfBlock);
        align = (seqLen % ElementOfBlock == 0) && (alignOfN == blockLen);

        uint16_t dstStride = (blockHeight - alignOfN) * sizeof(qType) / DATA_ALIGN_BYTES;

        if (align) {
            uint16_t copyLen = alignOfN * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t srcStride = (seqLen - blockLen) * sizeof(qType) / DATA_ALIGN_BYTES;

            DataCopyParams copyParms = { copyBlock, copyLen, srcStride, dstStride };
            DataCopy(lt, gt[offset], copyParms);
        } else {
            uint16_t copyLenBytes = blockLen * sizeof(qType);
            uint16_t srcStrideBytes = (seqLen - blockLen) * sizeof(qType);

            uint8_t padLens = alignOfN - blockLen;
            DataCopyParams copyParms = { copyBlock, copyLenBytes, srcStrideBytes, dstStride };
            DataCopyPadParams padParms = { true, 0, padLens, 0 };
            DataCopyPad(lt, gt[offset], copyParms, padParms);
        }
    }

    __aicore__ inline bool GenMask(
        LocalTensor<float>& inMaskLt, int causalMask, int64_t maskLen, int64_t maskOffset, float sclae)
    {
        bool needMask = false;
        if (causalMask == 1) {
            DoCausalMask<float, CausalMaskT::MASK_TRIL>(inMaskLt, maskOffset, maskLen, this->blockHeight,
                                                        maskLen / this->blockHeight, sclae);
            needMask = true;
        }

        return needMask;
    }

    __aicore__ inline bool DoMaskInitOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        uint32_t causalMask,
        int64_t maskOffset,
        int64_t thisLen,
        int64_t blockOffset,
        float scale,
        uint32_t n)
    {
        bool needMask = false;
        if (maskType == CausalMaskT::MASK_TRIL) {
            inMaskLtFp32 = queMaskIn.AllocTensor<float>();
            needMask = GenMask(
                inMaskLtFp32, causalMask, thisLen,
                ((causalMask == 1) ? (blockOffset) : n), scale);
            queMaskIn.EnQue(inMaskLtFp32);
        } else if (maskType == CausalMaskT::MASK_CUSTOME) {
            int64_t thisMaskOffset = maskOffset + blockOffset * seqLen;
            inMaskLt = queMaskIn.AllocTensor<qType>();
            DataCopyMayPad(inMaskLt, attnMaskGt,
                (uint16_t)(thisLen / blockHeight), n, thisMaskOffset);
            queMaskIn.EnQue(inMaskLt);

            needMask = true;
        }

        return needMask;
    }

    __aicore__ inline void VecScoreImpl(
        int64_t taskId,
        int64_t biasOffset,
        int64_t maskOffset,
        float scale,
        uint32_t causalMask,
        uint32_t m,
        uint32_t n)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t total = m * blockHeight;
        int64_t offset = midResultIdx * blockHeight * blockHeight;

        auto tmpLt = tmpBuff.AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;

        int64_t remain = total;
        while (remain > 0) {
            int64_t thisLen = vectorScoreUbBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t thisOffset = offset + (total - remain);
            auto inLt = queIn.AllocTensor<qType>();

            DataCopy(inLt, attnScoreGt[thisOffset], thisLen);

            queIn.EnQue(inLt);

            int64_t blockOffset = (total - remain) / blockHeight;

            bool needMask = DoMaskInitOptional(inMaskLt, inMaskLtFp32, causalMask,
                maskOffset, thisLen, blockOffset, scale, n);

            CalcuScoreWithFloat32(inLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32,
                needMask, thisLen, scale);
            
            auto outLt = queOut.DeQue<qType>();
            DataCopy(attnScoreGt[thisOffset], outLt, thisLen);
            queOut.FreeTensor(outLt);

            remain = remain - thisLen;
        }

        tmpBuff.FreeTensor(tmpLt);
    }

    __aicore__ inline void BiasMaskImpl(int64_t taskId, int64_t tsBiasOffset, int64_t posBiasOffset, int64_t maskOffset,
        uint32_t causalMask, uint32_t m, uint32_t n)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t total = m * blockHeight;
        int64_t offset = midResultIdx * blockHeight * blockHeight;

        auto tmpLt = tmpBuff.AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;

        int64_t remain = total;
        while (remain > 0) {
            int64_t thisLen = vectorScoreUbBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t thisOffset = offset + (total - remain);
            int64_t blockOffset = (total - remain) / blockHeight;

            bool needMask = DoMaskInitOptional(inMaskLt, inMaskLtFp32, causalMask,
                maskOffset, thisLen, blockOffset, 1.0, n);

            if (maskType != CausalMaskT::MASK_NONE) {
                queMaskIn.DeQue();
            }

            if (needMask && (maskType == CausalMaskT::MASK_CUSTOME)) {
                inMaskLtFp32 = inMaskLt.template ReinterpretCast<float>();
                CastQtype2Float(inMaskLtFp32, inMaskLt, tmpLt, thisLen);
            }

            // timestampBias
            int64_t thisTsBiasOffset = tsBiasOffset + blockOffset * seqLen;
            LocalTensor<qType> tsBiasLt = queIn.AllocTensor<qType>();
            DataCopyMayPad(tsBiasLt, timestampBiasGt, (uint16_t)(thisLen / blockHeight), n, thisTsBiasOffset);
            queIn.EnQue(tsBiasLt);
            tsBiasLt = queIn.DeQue<qType>();
            LocalTensor<float> tsBiasLtFp32 = tsBiasLt.template ReinterpretCast<float>();
            CastQtype2Float(tsBiasLtFp32, tsBiasLt, tmpLt, thisLen);
            if (needMask) {
                Mul<float>(tsBiasLtFp32, tsBiasLtFp32, inMaskLtFp32, thisLen);
            }
            CastFloat2Qtype(tsBiasLt, tsBiasLtFp32, tmpLtFp32, thisLen);

            // positionBias
            int64_t thisPosBiasOffset = posBiasOffset + blockOffset * seqLen;
            LocalTensor<qType> posBiasLt = biasIn.AllocTensor<qType>();
            DataCopyMayPad(posBiasLt, positionBiasGt, (uint16_t)(thisLen / blockHeight), n, thisPosBiasOffset);
            biasIn.EnQue(posBiasLt);
            posBiasLt = biasIn.DeQue<qType>();
            LocalTensor<float> posBiasLtFp32 = posBiasLt.template ReinterpretCast<float>();
            CastQtype2Float(posBiasLtFp32, posBiasLt, tmpLt, thisLen);
            if (needMask) {
                Mul<float>(posBiasLtFp32, posBiasLtFp32, inMaskLtFp32, thisLen);
            }
            CastFloat2Qtype(posBiasLt, posBiasLtFp32, tmpLtFp32, thisLen);

            SetFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdV2MTE3);

            DataCopy(tsTempGt[thisOffset], tsBiasLt, thisLen);
            DataCopy(posTempGt[thisOffset], posBiasLt, thisLen);

            pipe_barrier(PIPE_ALL);
            queIn.FreeTensor(tsBiasLt);
            biasIn.FreeTensor(posBiasLt);
            if (maskType != CausalMaskT::MASK_NONE) {
                queMaskIn.FreeTensor(inMaskLtFp32);
            }

            remain = remain - thisLen;
        }

        tmpBuff.FreeTensor(tmpLt);
    }

    __aicore__ inline void DoQkMatmulImpl(int64_t qOffset, int64_t kOffset, uint32_t taskId, uint32_t m, uint32_t n,
                                          uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outOffset = midResultIdx * blockHeight * blockHeight;

        qkMatmul.SetTensorA(qGt[qOffset]);
        qkMatmul.SetTensorB(kGt[kOffset], true);
        qkMatmul.SetTail(m, n, k);

        qkMatmul.template IterateAll<false>(attnScoreGt[outOffset], 0, false, true);
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, uint8_t isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * blockHeight * headDim;
        int64_t sOffset = midResultIdx * blockHeight * blockHeight;

        svMatmul.SetTensorA(attnScoreGt[sOffset]);
        svMatmul.SetTensorB(vGt[vOffset]);
        svMatmul.SetTail(m, n, k);

        svMatmul.template IterateAll<false>(svResultGt[outOffset], isAtomicAdd, false, true);
    }

    __aicore__ inline void DoTvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, uint8_t isAtomicAdd,
        uint32_t m, uint32_t n, uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * blockHeight * headDim;
        int64_t sOffset = midResultIdx * blockHeight * blockHeight;

        tvMatmul.SetTensorA(tsTempGt[sOffset]);
        tvMatmul.SetTensorB(vGt[vOffset]);
        tvMatmul.SetTail(m, n, k);

        tvMatmul.template IterateAll<false>(tvResultGt[outOffset], isAtomicAdd, false, true);
    }

    __aicore__ inline void DoPvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, uint8_t isAtomicAdd,
        uint32_t m, uint32_t n, uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * blockHeight * headDim;
        int64_t sOffset = midResultIdx * blockHeight * blockHeight;

        pvMatmul.SetTensorA(posTempGt[sOffset]);
        pvMatmul.SetTensorB(vGt[vOffset]);
        pvMatmul.SetTail(m, n, k);

        pvMatmul.template IterateAll<false>(pvResultGt[outOffset], isAtomicAdd, false, true);
    }

    __aicore__ inline void DoTransImpl(const GlobalTensor<float>& srcGt, int64_t kThisOffset, int64_t thisLen,
        int64_t outOffset)
    {
        uint16_t copyLen = headDim * sizeof(qType) / DATA_ALIGN_BYTES;
        uint16_t strideLen = enableBias ? (OUTPUT_DIM2_TIMES3 * headNum * headDim) : (headNum * headDim);
        uint16_t distStride = (strideLen - headDim) * sizeof(qType) / DATA_ALIGN_BYTES;

        LocalTensor<float> inLt = queIn.AllocTensor<float>();
        DataCopy(inLt, srcGt[kThisOffset], thisLen);
        queIn.EnQue(inLt);

        LocalTensor<float> newInLt = queIn.DeQue<float>();
        LocalTensor<qType> outLt = queOut.AllocTensor<qType>();
        if (std::is_same<qType, float>::value) {
            DataCopy(outLt.template ReinterpretCast<float>(), newInLt, thisLen);
        } else {
            Cast(outLt, newInLt, RoundMode::CAST_RINT, thisLen);
        }
        queIn.FreeTensor(inLt);

        queOut.EnQue(outLt);
        LocalTensor<qType> newOutLt = queOut.DeQue<qType>();
        DataCopyParams copyParms = {(uint16_t)(thisLen / headDim), copyLen, 0, distStride};
        DataCopy(attnOutputGt[outOffset], newOutLt, copyParms);
        queOut.FreeTensor(newOutLt);
    }

    __aicore__ inline void DoTransResultImpl(int64_t transTaskId, int64_t outStartOffset, uint32_t m)
    {
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t inOffset = outMidIndex * blockHeight * headDim;

        int64_t total = m * headDim;
        int64_t remain = total;
        while (remain > 0) {
            int64_t thisLen = vectorScoreUbBlockElem;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = inOffset + (total - remain);
            int64_t thisLineOffset = (total - remain) / headDim;

            int64_t thisLineOffsetLen = enableBias ? (OUTPUT_DIM2_TIMES3 * thisLineOffset * headNum * headDim) :
                (thisLineOffset * headNum * headDim);
            int64_t svOutOffset = outStartOffset + thisLineOffsetLen;
            DoTransImpl(svResultGt, kThisOffset, thisLen, svOutOffset);

            if (enableBias) {
                int64_t tvOutOffset = svOutOffset + headNum * headDim;
                DoTransImpl(tvResultGt, kThisOffset, thisLen, tvOutOffset);

                int64_t pvOutOffset = tvOutOffset + headNum * headDim;
                DoTransImpl(pvResultGt, kThisOffset, thisLen, pvOutOffset);
            }

            remain = remain - thisLen;
        }
    }

    // GM_ADDR
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR timestampBias;
    GM_ADDR positionBias;
    GM_ADDR mask;
    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;

    // Shape
    int64_t batchSize;
    int64_t seqLen;
    int64_t headNum;
    int64_t headDim;

    // Tiling
    int64_t blockHeight;
    int64_t seqBlockNumQk;

    // Tiling-QK
    int64_t qkTotalBlock;

    // Ub
    int64_t vectorScoreUbBlockElem;

    // split
    int64_t blockSplitNum;

    // Attr
    float siluScale;
    CausalMaskT maskType;
    bool enableBias;

    TEventID eventIdV2MTE3;

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> biasIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn;
    TQue<TPosition::VECCALC, USE_QUEUE_NUM> tmpBuff;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut;

    // Gt
    GlobalTensor<qType> qGt;
    GlobalTensor<qType> kGt;
    GlobalTensor<qType> vGt;
    GlobalTensor<qType> attnOutputGt;
    GlobalTensor<qType> attnScoreGt;
    GlobalTensor<qType> tsTempGt;
    GlobalTensor<qType> posTempGt;
    GlobalTensor<qType> timestampBiasGt;
    GlobalTensor<qType> positionBiasGt;
    GlobalTensor<qType> attnMaskGt;
    GlobalTensor<float> svResultGt;
    GlobalTensor<float> tvResultGt;
    GlobalTensor<float> pvResultGt;

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
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopySVB1<qType>>>
        svMatmul;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopySVB1<qType>>>
        tvMatmul;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>, CFG_NORM,
                   matmul::MatmulCallBackFunc<nullptr, nullptr, CopySVB1<qType>>>
        pvMatmul;
};
}  // namespace HstuDenseForwardFuxi
#endif
