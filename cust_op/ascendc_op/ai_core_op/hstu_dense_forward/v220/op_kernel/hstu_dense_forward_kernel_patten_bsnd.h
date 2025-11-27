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

#ifndef HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_H
#define HSTU_DENSE_FORWARD_KERNEL_PATTEN_BSND_H

#include <unistd.h>

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "hstu_dense_causal_mask.h"
#include "hstu_common_const.h"
#include "matmul_constexpr.h"

using namespace AscendC;

namespace HstuDenseForward {

constexpr int VEC_PER_PROCESS = 32;
constexpr int UB_SIZE = 170 * 1024;  // 170KB
constexpr int QUEUE_IN_NUM = 2;
constexpr int SPLIT_CORE = 2;
constexpr int ALIGN_16 = 16;

constexpr int VCORE_NUM_IN_ONE_AIC = 2;
constexpr int COMPUTE_PIPE_NUM = 3;
constexpr int TRANS_PIPE_NUM = 4;
constexpr int INT_ALIGN_NUM = 8;

struct Args {
    // hstu normal
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR attnBias;
    // jagged
    GM_ADDR seqOffsetQ;
    GM_ADDR seqOffsetK;
    // page
    GM_ADDR seqOffsetT;
    GM_ADDR kvCache;
    GM_ADDR pageOffsets;
    GM_ADDR pageIds;
    GM_ADDR lastPageLen;
    // mask
    GM_ADDR numContext;
    GM_ADDR numTarget;

    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename qType>
__aicore__ inline void CopyQKA1(const LocalTensor<int8_t>& aMatrix, const __gm__ void* gm, int row, int col, int useM,
                                int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useM * useK);
    int blockLen = useM * useK;

    HstuDenseForwardTilingData* tilingP = reinterpret_cast<HstuDenseForwardTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int64_t headNum = tilingP->headNum;
    int32_t baseM = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicM : mmStaticConfigQKFp16.basicM;
    int32_t baseK = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicK : mmStaticConfigQKFp16.basicK;

    auto alignOfM = AlignUp(useM, ALIGN_16);
    Nd2NzParams param = {
        1, static_cast<uint16_t>(useM), static_cast<uint16_t>(useK), 0,
        static_cast<uint16_t>(dim * headNum), static_cast<uint16_t>(alignOfM), 1, 0
    };

    int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNum * static_cast<int64_t>(baseM) +
                         static_cast<int64_t>(col) * static_cast<int64_t>(baseK);
    DataCopy(aMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType>
__aicore__ inline void CopyQKB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseForwardTilingData* tilingP = reinterpret_cast<HstuDenseForwardTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int32_t headNumK = static_cast<int32_t>(dataPtr);
    int32_t baseN = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicN : mmStaticConfigQKFp16.basicN;
    int32_t baseK = std::is_same<qType, float>::value ? mmStaticConfigQKFp32.basicK : mmStaticConfigQKFp16.basicK;

    auto alignOfN = AlignUp(useN, ALIGN_16);
    Nd2NzParams param = {
        1, static_cast<uint16_t>(useN), static_cast<uint16_t>(useK), 0,
        static_cast<uint16_t>(dim * headNumK), static_cast<uint16_t>(alignOfN), 1, 0
    };

    int64_t offsetOfGt = static_cast<int64_t>(col) * dim * headNumK * static_cast<int64_t>(baseN) +
                         static_cast<int64_t>(row) * static_cast<int64_t>(baseK);
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType>
__aicore__ inline void CopySVB1(const LocalTensor<int8_t>& bMatrix, const __gm__ void* gm, int row, int col, int useK,
                                int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    GlobalTensor<qType> globalGt;
    globalGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(const_cast<__gm__ void*>(gm)), useN * useK);

    HstuDenseForwardTilingData* tilingP = reinterpret_cast<HstuDenseForwardTilingData*>(tilingPtr);
    int64_t dim = tilingP->dim;
    int32_t headNumK = static_cast<int32_t>(dataPtr);
    int32_t baseN = std::is_same<qType, float>::value ? mmStaticConfigSVFp32.basicN : mmStaticConfigSVFp16.basicN;
    int32_t baseK = std::is_same<qType, float>::value ? mmStaticConfigSVFp32.basicK : mmStaticConfigSVFp16.basicK;
    auto alignOfK = AlignUp(useK, ALIGN_16);

    Nd2NzParams param = {
        1, static_cast<uint16_t>(useK), static_cast<uint16_t>(useN), 0,
        static_cast<uint16_t>(dim * headNumK), static_cast<uint16_t>(alignOfK), 1, 0
    };

    int64_t offsetOfGt = static_cast<int64_t>(row) * dim * headNumK * static_cast<int64_t>(baseK) +
                         static_cast<int64_t>(col) * static_cast<int64_t>(baseN);
    DataCopy(bMatrix.ReinterpretCast<qType>(), globalGt[offsetOfGt], param);
};

template <typename qType, bool enableBias, bool isQkUseUb, CausalMaskT maskType>
class HstuDenseForwardKernelPattenBsnd {
public:
    static constexpr int ElementOfBlock = DATA_ALIGN_BYTES / sizeof(qType);
    static constexpr int blockHeight = isQkUseUb ? BLOCK_HEIGHT_128 : BLOCK_HEIGHT_256;
    static constexpr int vectorScoreUbBlockElem =
        (isQkUseUb ? (blockHeight * blockHeight) : (VEC_PER_PROCESS * blockHeight)) / USE_QUEUE_NUM;
    static constexpr auto qkMMCPos = isQkUseUb ? TPosition::VECIN : TPosition::GM;
    static constexpr MatmulConfig qkMMConfig = std::is_same<qType, float>::value ?
        mmStaticConfigQKFp32 : mmStaticConfigQKFp16;
    static constexpr MatmulConfig svMMConfig = std::is_same<qType, float>::value ?
        mmStaticConfigSVFp32 : mmStaticConfigSVFp16;

    __aicore__ inline HstuDenseForwardKernelPattenBsnd() {}
    __aicore__ inline void Init(const Args& args, const HstuDenseForwardTilingData* __restrict tilingDataPtr,
                                TPipe* pipePtr)
    {
        InitArgs(args, tilingDataPtr);
        InitPipe(pipePtr);
    }

    __aicore__ inline void InitArgs(const Args& args, const HstuDenseForwardTilingData* __restrict tilingDataPtr)
    {
        q = args.q;
        k = args.k;
        v = args.v;
        attnBias = args.attnBias;
        mask = args.mask;
        seqOffsetQ = args.seqOffsetQ;
        seqOffsetK = args.seqOffsetK;

        attnOutput = args.attnOutput;
        workspace = args.workspace;

        numContext = args.numContext;
        numTarget = args.numTarget;
        // Batch Size
        xDim0 = tilingDataPtr->batchSize;
        // Seq Len
        xDim1 = tilingDataPtr->seqLen;
        this->maxSeqLenQ = tilingDataPtr->maxSeqLenq;
        this->maxSeqLenK = tilingDataPtr->maxSeqLenk;
        // Head Num
        xDim2 = tilingDataPtr->headNum;
        // Embedding Dim
        xDim3 = tilingDataPtr->dim;

        // attr
        siluScale = tilingDataPtr->siluScale;
        alpha = tilingDataPtr->alpha;
        targetGroupSize = tilingDataPtr->targetGroupSize;
        enableNumContext = tilingDataPtr->enableNumContext;
        enableNumTarget = tilingDataPtr->enableNumTarget;

        // copyKV
        copyHeadNum = tilingDataPtr->headNumK;

        // GQA
        headNumK = tilingDataPtr->headNumK;
        headRatio = tilingDataPtr->headRatio;
    }

    __aicore__ inline void InitPipe(TPipe* pipePtr)
    {
        pipe = pipePtr;

        // Gt
        qGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(q));
        kGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(k));
        vGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(v));

        if constexpr (enableBias) {
            attnBiasGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnBias));
        }

        if constexpr (maskType == CausalMaskT::MASK_CUSTOM) {
            attnMaskGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(mask));
        }

        attnOutputGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(attnOutput));

        int64_t oneBlockMidElem = blockHeight * blockHeight * COMPUTE_PIPE_NUM;
        int64_t oneCoreMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidElem;

        int64_t oneBlockMidTransElem = blockHeight * MAX_BLOCK_DIM * TRANS_PIPE_NUM;
        int64_t oneCoreTransMidElem = GetBlockNum() * VCORE_NUM_IN_ONE_AIC * oneBlockMidTransElem;
        int64_t kvOffset = oneCoreMidElem + oneCoreTransMidElem * 3; // svResultGt midkGt midvGt

        attnScoreGt.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(workspace) + GetBlockIdx() * oneBlockMidElem);
        svResultGt.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + oneCoreMidElem + GetBlockIdx() * oneBlockMidTransElem,
            oneBlockMidTransElem);

        if constexpr (!isQkUseUb) {
            // Init pipe total 32K * 5 = 160K
            transUbBlockElem = vectorScoreUbBlockElem;
            pipe->InitBuffer(queIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(queOut, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(tmpBuff, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(biasIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        } else {
            transUbBlockElem = vectorScoreUbBlockElem / 2;
            pipe->InitBuffer(queIn, USE_QUEUE_NUM, transUbBlockElem * sizeof(float));
            pipe->InitBuffer(queOut, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(tmpBuff, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(biasIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(queMaskIn, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(qkQueInA, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
            pipe->InitBuffer(qkQueInB, USE_QUEUE_NUM, vectorScoreUbBlockElem * sizeof(float));
        }
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

    __aicore__ inline void AllocQkUbTensor()
    {
        if constexpr (isQkUseUb) {
            this->qkUbA = this->qkQueInA.template AllocTensor<qType>();
            this->qkUbB = this->qkQueInB.template AllocTensor<qType>();
        }
    }

    __aicore__ inline void FreeQkUbTensor()
    {
        if constexpr (isQkUseUb) {
            this->qkQueInA.template FreeTensor<qType>(this->qkUbA);
            this->qkQueInB.template FreeTensor<qType>(this->qkUbB);
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

    __aicore__ inline void DoMaskOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& newOutLt,
        int64_t thisLen,
        bool needMask,
        float scale)
    {
        if constexpr (maskType != CausalMaskT::MASK_NONE) {
            queMaskIn.DeQue();
        }

        if (needMask) {
            if constexpr (maskType == CausalMaskT::MASK_CUSTOM) {
                inMaskLtFp32 = inMaskLt.template ReinterpretCast<float>();
                CastQtype2Float(inMaskLtFp32, inMaskLt, tmpLt, thisLen);
                Muls<float>(inMaskLtFp32, inMaskLtFp32, scale, thisLen);
            }
            Mul<float>(newOutLt, newOutLt, inMaskLtFp32, thisLen);
        } else {
            Muls<float>(newOutLt, newOutLt, scale, thisLen);
        }

        if constexpr (maskType != CausalMaskT::MASK_NONE) {
            queMaskIn.FreeTensor(inMaskLtFp32);
        }
    }

    __aicore__ inline void DoBiasOptional(
        LocalTensor<float>& newInLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& tmpLt,
        int64_t thisLen
    )
    {
        if constexpr (enableBias) {
            biasIn.DeQue();
            auto newBiasLt = biasLt.template ReinterpretCast<float>();
            CastQtype2Float(newBiasLt, biasLt, tmpLt, thisLen);
            Add<float>(newInLt, newInLt, newBiasLt, thisLen);
            biasIn.FreeTensor(biasLt);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32NoRab(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& tmpLtFp32,
        bool needMask,
        int64_t thisLen,
        float scale)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            queIn.DeQue();
            Cast(tmpLtFp32, inLt, RoundMode::CAST_NONE, thisLen);
            queIn.FreeTensor(inLt);

            auto biasLtFp32 = biasLt.template ReinterpretCast<float>();
            Muls<float>(tmpLtFp32, tmpLtFp32, alpha, thisLen);
            Silu<float>(biasLtFp32, tmpLtFp32, thisLen);
            DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, biasLtFp32, thisLen, needMask, scale);

            auto outLt = queOut.AllocTensor<qType>();
            Cast(outLt, biasLtFp32, RoundMode::CAST_RINT, thisLen);
            queOut.EnQue(outLt);
        } else {
            queIn.DeQue();

            auto outLt = queOut.AllocTensor<qType>();
            Muls<float>(inLt, inLt, alpha, thisLen);
            Silu<float>(outLt, inLt, thisLen);
            queIn.FreeTensor(inLt);

            DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, outLt, thisLen, needMask, scale);
            queOut.EnQue(outLt);
        }
    }

    __aicore__ inline void CalcuScoreWithFloat32(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& biasLt,
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
        DoBiasOptional(newInLt, biasLt, tmpLt, thisLen);

        auto outLt = queOut.AllocTensor<qType>();
        auto newOutLt = outLt.template ReinterpretCast<float>();
        Muls<float>(newInLt, newInLt, alpha, thisLen);
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
        align = (maxSeqLenK % ElementOfBlock == 0) && (alignOfN == blockLen);

        uint16_t dstStride = (blockHeight - alignOfN) * sizeof(qType) / DATA_ALIGN_BYTES;

        if (align) {
            uint16_t copyLen = alignOfN * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t srcStride = (maxSeqLenK - blockLen) * sizeof(qType) / DATA_ALIGN_BYTES;

            DataCopyParams copyParms = { copyBlock, copyLen, srcStride, dstStride };
            DataCopy(lt, gt[offset], copyParms);
        } else {
            uint16_t copyLenBytes = blockLen * sizeof(qType);
            uint16_t srcStrideBytes = (maxSeqLenK - blockLen) * sizeof(qType);

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

    template<typename MaskInfoType>
    __aicore__ inline bool DoMaskInitOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        MaskInfoType& maskinfo,
        int64_t maskOffset,
        int64_t thisLen,
        int64_t blockOffset,
        float scale,
        uint32_t n)
    {
        bool needMask = false;
        if constexpr (maskType == CausalMaskT::MASK_TRIL) {
            inMaskLtFp32 = queMaskIn.AllocTensor<float>();
            if constexpr (std::is_same<MaskInfoType, uint32_t>::value) {
                // 处理 uint32_t 类型
                needMask = GenMask(
                    inMaskLtFp32,
                    maskinfo,
                    thisLen,
                    ((maskinfo > 0) ? (blockOffset) : n),  // blockOffset为行号
                    scale);
            } else {
                // 处理 BlockMaskParams 类型
                BlockMaskGenerator blkMaskGen(maskinfo);
                needMask =
                    blkMaskGen.GenMask(inMaskLtFp32, blockOffset, thisLen / this->blockHeight, this->blockHeight);
            }

            queMaskIn.EnQue(inMaskLtFp32);
        } else if constexpr (maskType == CausalMaskT::MASK_CUSTOM) {
            int64_t thisMaskOffset = maskOffset + blockOffset * maxSeqLenK;
            inMaskLt = queMaskIn.AllocTensor<qType>();
            DataCopyMayPad(inMaskLt, attnMaskGt,
                (uint16_t)(thisLen / blockHeight), n, thisMaskOffset);
            queMaskIn.EnQue(inMaskLt);

            needMask = true;
        }

        return needMask;
    }

    __aicore__ inline void DoBiasCopyOptional(
        LocalTensor<qType>& biasLt,
        int64_t biasOffset,
        int64_t thisLen,
        int64_t blockOffset,
        uint32_t n)
    {
        if constexpr (enableBias) {
            int64_t thisBiasOffset = biasOffset + blockOffset * maxSeqLenK;
            biasLt = biasIn.AllocTensor<qType>();
            DataCopyMayPad(biasLt, attnBiasGt,
                (uint16_t)(thisLen / blockHeight), n, thisBiasOffset);
            biasIn.EnQue(biasLt);
        }
    }

    template<typename MaskInfoType>
    __aicore__ inline void VecScoreImpl(
        int64_t taskId,
        int64_t biasOffset,
        int64_t maskOffset,
        float scale,
        MaskInfoType& maskinfo,
        uint32_t m,
        uint32_t n)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t total = m * blockHeight;
        int64_t offset = midResultIdx * blockHeight * blockHeight;

        auto tmpLt = tmpBuff.AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> biasLt;
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;

        if constexpr (!enableBias) {
            biasLt = biasIn.AllocTensor<qType>();
        }

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
            DoBiasCopyOptional(biasLt, biasOffset, thisLen, blockOffset, n);

            bool needMask =
                DoMaskInitOptional(inMaskLt, inMaskLtFp32, maskinfo, maskOffset, thisLen, blockOffset, scale, n);

            if constexpr (enableBias) {
                CalcuScoreWithFloat32(inLt, biasLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32,
                    needMask, thisLen, scale);
            } else {
                CalcuScoreWithFloat32NoRab(inLt, biasLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32,
                    needMask, thisLen, scale);
            }
            
            auto outLt = queOut.DeQue<qType>();
            DataCopy(attnScoreGt[thisOffset], outLt, thisLen);
            queOut.FreeTensor(outLt);

            remain = remain - thisLen;
        }

        if constexpr (!enableBias) {
            biasIn.FreeTensor(biasLt);
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
        qkMatmul.SetSelfDefineData(copyHeadNum); // 设置CopyQK的自定义headNum数据

        qkMatmul.template IterateAll<false>(attnScoreGt[outOffset], 0, false, true);
    }

    __aicore__ inline void DoQkMatmulImpl(int64_t qOffset, int64_t kOffset, uint32_t taskId, uint32_t m, uint32_t n,
                                          uint32_t k, const GlobalTensor<qType>& midkGt)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outOffset = midResultIdx * blockHeight * blockHeight;

        qkMatmul.SetTensorA(qGt[qOffset]);
        qkMatmul.SetTensorB(midkGt[kOffset], true);
        qkMatmul.SetTail(m, n, k);
        qkMatmul.SetSelfDefineData(copyHeadNum); // 设置CopyQK的自定义headNum数据

        qkMatmul.template IterateAll<false>(attnScoreGt[outOffset], 0, false, true);
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, int isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * blockHeight * MAX_BLOCK_DIM;
        int64_t sOffset = midResultIdx * blockHeight * blockHeight;

        svMatmul.SetTensorA(attnScoreGt[sOffset]);
        svMatmul.SetTensorB(vGt[vOffset]);
        svMatmul.SetTail(m, n, k);
        svMatmul.SetSelfDefineData(copyHeadNum); // 设置CopyQK的自定义headNum数据

        if (isAtomicAdd == 0) {
            // Override
            svMatmul.template IterateAll<false>(svResultGt[outOffset], 0, false, true);
        } else {
            // Automic Add
            svMatmul.template IterateAll<false>(svResultGt[outOffset], 1, false, true);
        }
    }

    __aicore__ inline void DoSvMatmulImpl(int64_t vOffset, uint32_t taskId, uint32_t transTaskId, int isAtomicAdd,
                                          uint32_t m, uint32_t n, uint32_t k, const GlobalTensor<qType>& midvGt)
    {
        int64_t midResultIdx = taskId % COMPUTE_PIPE_NUM;
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t outOffset = outMidIndex * blockHeight * MAX_BLOCK_DIM;
        int64_t sOffset = midResultIdx * blockHeight * blockHeight;

        svMatmul.SetTensorA(attnScoreGt[sOffset]);
        svMatmul.SetTensorB(midvGt[vOffset]);
        svMatmul.SetTail(m, n, k);
        svMatmul.SetSelfDefineData(copyHeadNum); // 设置拷贝v矩阵的自定义headNum数据

        if (isAtomicAdd == 0) {
            // Override
            svMatmul.template IterateAll<false>(svResultGt[outOffset], 0, false, true);
        } else {
            // Automic Add
            svMatmul.template IterateAll<false>(svResultGt[outOffset], 1, false, true);
        }
    }

    __aicore__ inline void DoTransSvImpl(int64_t transTaskId, int64_t outStartOffset, uint32_t m)
    {
        int64_t outMidIndex = transTaskId % TRANS_PIPE_NUM;
        int64_t inOffset = outMidIndex * blockHeight * MAX_BLOCK_DIM;

        int64_t total = m * xDim3;
        int64_t remain = total;
        
        DataCopyParams srcCopyParams;
        srcCopyParams.blockLen = xDim3 * sizeof(float) / DATA_ALIGN_BYTES;
        srcCopyParams.srcStride = (MAX_BLOCK_DIM - xDim3) * sizeof(float) / DATA_ALIGN_BYTES;
        srcCopyParams.dstStride = 0;

        DataCopyParams dstCopyParams;
        dstCopyParams.blockLen = xDim3 * sizeof(qType) / DATA_ALIGN_BYTES;
        dstCopyParams.srcStride = 0;
        dstCopyParams.dstStride = (xDim2 * xDim3 - xDim3) * sizeof(qType) / DATA_ALIGN_BYTES;

        int64_t copyLenEachLoopAlignHeadDim = transUbBlockElem / xDim3 * xDim3;

        while (remain > 0) {
            int64_t thisLen = copyLenEachLoopAlignHeadDim;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = inOffset + (total - remain) / xDim3 * MAX_BLOCK_DIM;

            srcCopyParams.blockCount = static_cast<uint16_t>(thisLen / xDim3);
            LocalTensor<float> inLt = queIn.AllocTensor<float>();
            DataCopy(inLt, svResultGt[kThisOffset], srcCopyParams);

            queIn.EnQue(inLt);

            LocalTensor<float> newInLt = queIn.DeQue<float>();
            LocalTensor<qType> outLt = queOut.AllocTensor<qType>();
            if constexpr (std::is_same<qType, float>::value) {
                DataCopy(outLt.template ReinterpretCast<float>(), newInLt, thisLen);
            } else {
                Cast(outLt, newInLt, RoundMode::CAST_RINT, thisLen);
            }

            queOut.EnQue(outLt);
            queIn.FreeTensor(newInLt);

            LocalTensor<qType> newOutLt = queOut.DeQue<qType>();

            dstCopyParams.blockCount = static_cast<uint16_t>(thisLen / xDim3);
            int64_t thisLineOffset = (total - remain) / xDim3;
            int64_t outOffset = outStartOffset + thisLineOffset * xDim2 * xDim3;
            DataCopy(attnOutputGt[outOffset], newOutLt, dstCopyParams);

            queOut.FreeTensor(newOutLt);
            remain = remain - thisLen;
        }
    }

    // GM_ADDR
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR attnBias;
    GM_ADDR mask;
    GM_ADDR seqOffsetQ;
    GM_ADDR seqOffsetK;

    GM_ADDR numContext;
    GM_ADDR numTarget;

    GM_ADDR attnOutput;
    GM_ADDR workspace;
    GM_ADDR tiling;

    // Shape
    int64_t xDim0;
    int64_t xDim1;
    int64_t xDim2;
    int64_t xDim3;
    int64_t maxSeqLenQ;
    int64_t maxSeqLenK;
    bool enableNumContext;
    bool enableNumTarget;

    // Tiling
    int64_t seqBlockNumQk;

    // Tiling-QK
    int64_t qkTotalBlock;

    // Ub
    int64_t transUbBlockElem;

    // split
    int64_t blockSplitNum;

    // Attr
    float siluScale;
    float alpha;
    int64_t targetGroupSize;

    // copyQKV
    uint64_t copyHeadNum;

    // GQA
    uint64_t headNumK;
    uint64_t headRatio;

    // Tpipe
    TPipe *pipe;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> biasIn;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn;
    TQue<TPosition::VECCALC, USE_QUEUE_NUM> tmpBuff;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> qkQueInA;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> qkQueInB;

    // Gt
    GlobalTensor<qType> qGt;
    GlobalTensor<qType> kGt;
    GlobalTensor<qType> vGt;
    GlobalTensor<qType> attnOutputGt;
    GlobalTensor<qType> attnScoreGt;
    GlobalTensor<qType> attnBiasGt;
    GlobalTensor<qType> attnMaskGt;
    GlobalTensor<float> svResultGt;

    LocalTensor<qType> qkUbA;
    LocalTensor<qType> qkUbB;

    // Matmul
    using QK_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using QK_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, true>;
    using QK_MM_C_T = matmul::MatmulType<qkMMCPos, CubeFormat::ND, qType, false>;
    using QK_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using QK_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, CopyQKA1<qType>, CopyQKB1<qType>>;

    static constexpr auto staticQkTilingCfg = GetMatmulApiTiling<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T>(
        qkMMConfig, MATMUL_L1_SIZE);
    matmul::Matmul<QK_MM_A_T, QK_MM_B_T, QK_MM_C_T, QK_MM_BIAS_T, staticQkTilingCfg, QK_MM_CB_T> qkMatmul;

    using SV_MM_A_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using SV_MM_B_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType, false>;
    using SV_MM_C_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using SV_MM_BIAS_T = matmul::MatmulType<TPosition::GM, CubeFormat::ND, qType>;
    using SV_MM_CB_T = matmul::MatmulCallBackFunc<nullptr, nullptr, CopySVB1<qType>>;

    static constexpr auto staticSvTilingCfg = GetMatmulApiTiling<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T>(
        svMMConfig, MATMUL_L1_SIZE);
    matmul::Matmul<SV_MM_A_T, SV_MM_B_T, SV_MM_C_T, SV_MM_BIAS_T, staticSvTilingCfg, SV_MM_CB_T> svMatmul;
};
}  // namespace HstuDenseForward
#endif
