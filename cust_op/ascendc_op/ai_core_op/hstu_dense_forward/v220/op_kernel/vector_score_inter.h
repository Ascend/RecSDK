/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef VECTOR_SCORE_INTER_H
#define VECTOR_SCORE_INTER_H

#include <type_traits>
#include "kernel_operator.h"
#include "hstu_common_const.h"
#include "hstu_dense_causal_mask.h"

namespace HstuForward {

template <typename MaskInfoType>
struct VecScoreRabParam {
    int64_t srcOffset;
    int64_t biasOffset;
    int64_t maskOffset;
    uint32_t m;
    uint32_t n;
    MaskInfoType& maskinfo;
};

template <typename qType>
struct VecScoreRabGtInfo {
    GlobalTensor<qType> attnBiasGt;
    GlobalTensor<qType> attnMaskGt;
};

template <typename qType, CausalMaskT maskType, typename MaskInfoType, typename VectorScoreImpl>
class VectorScoreInter {
public:
    static constexpr int ElementOfBlock = DATA_ALIGN_BYTES / sizeof(qType);
    __aicore__ inline VectorScoreInter() {}

    __aicore__ inline void InitParams(TPipe* pipePtr, VecScoreRabGtInfo<qType>& gtInfo, float siluScale, float alpha,
        int blockM, int blockN, int64_t maxSeqLenK)
    {
        pipe_ = pipePtr;
        attnMaskGt_ = gtInfo.attnMaskGt;

        siluScale_ = siluScale;
        alpha_ = alpha;
        blockM_ = blockM;
        blockN_ = blockN;
        maxSeqLenK_ = maxSeqLenK;

        pipe_->InitBuffer(queIn_, USE_QUEUE_NUM, VECTOR_SCORE_UB_BLOCK_ELEM * sizeof(float));
        pipe_->InitBuffer(queOut_, USE_QUEUE_NUM, VECTOR_SCORE_UB_BLOCK_ELEM * sizeof(float));
        pipe_->InitBuffer(tmpBuff_, USE_QUEUE_NUM, VECTOR_SCORE_UB_BLOCK_ELEM * sizeof(float));
        pipe_->InitBuffer(biasIn_, USE_QUEUE_NUM, VECTOR_SCORE_UB_BLOCK_ELEM * sizeof(float));
        pipe_->InitBuffer(queMaskIn_, USE_QUEUE_NUM, VECTOR_SCORE_UB_BLOCK_ELEM * sizeof(float));
    }

    __aicore__ inline void Init(TPipe* pipePtr, VecScoreRabGtInfo<qType>& gtInfo, float siluScale, float alpha,
        int blockM, int blockN, int64_t maxSeqLenK)
    {
        static_cast<VectorScoreImpl*>(this)->Init(pipePtr, gtInfo, siluScale, alpha, blockM, blockN, maxSeqLenK);
    }

    __aicore__ inline void VecScoreImpl(const VecScoreRabParam<MaskInfoType>& params, GlobalTensor<qType>& attnScoreGt)
    {
        static_cast<VectorScoreImpl*>(this)->VecScoreImpl(params, attnScoreGt);
    }

    __aicore__ inline void InitSyncGm(GlobalTensor<int32_t>& syncGm)
    {
        SyncAll<true>();
        if (GetBlockIdx() == 0) {
            const uint32_t coreNum = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
            uint32_t zeroNumber = coreNum * DATA_ALIGN_BYTES / sizeof(int32_t);
            auto zeroBuff = queOut_.template AllocTensor<int32_t>();
            Duplicate<int32_t>(zeroBuff, 0, zeroNumber);
            queOut_.EnQue(zeroBuff);
            auto overBuff = queOut_.DeQue<int32_t>();
            pipe_barrier(PIPE_ALL);
            DataCopy(syncGm, overBuff, zeroNumber);
            queOut_.template FreeTensor<int32_t>(overBuff);
        }
        SyncAll<true>();
    }

    __aicore__ inline void DataCopyMayPad(const LocalTensor<qType>& lt, GlobalTensor<qType>& gt, uint16_t copyBlock,
        uint32_t blockLen, int64_t offset)
    {
        bool align = false;
        uint16_t alignOfN = AlignUp(blockLen, ElementOfBlock);
        align = (maxSeqLenK_ % ElementOfBlock == 0) && (alignOfN == blockLen);

        uint16_t dstStride = (blockN_ - alignOfN) * sizeof(qType) / DATA_ALIGN_BYTES;

        if (align) {
            uint16_t copyLen = alignOfN * sizeof(qType) / DATA_ALIGN_BYTES;
            uint16_t srcStride = (maxSeqLenK_ - blockLen) * sizeof(qType) / DATA_ALIGN_BYTES;

            DataCopyParams copyParms = { copyBlock, copyLen, srcStride, dstStride };
            DataCopy(lt, gt[offset], copyParms);
        } else {
            uint16_t copyLenBytes = blockLen * sizeof(qType);
            uint16_t srcStrideBytes = (maxSeqLenK_ - blockLen) * sizeof(qType);

            uint8_t padLens = alignOfN - blockLen;
            DataCopyParams copyParms = { copyBlock, copyLenBytes, srcStrideBytes, dstStride };
            DataCopyPadParams padParms = { true, 0, padLens, 0 };
            DataCopyPad(lt, gt[offset], copyParms, padParms);
        }
    }

    __aicore__ inline bool DoMaskInitOptional(LocalTensor<qType>& inMaskLt, LocalTensor<float>& inMaskLtFp32,
        MaskInfoType& maskinfo, int64_t maskOffset, int64_t thisLen, int64_t blockOffset, uint32_t n)
    {
        bool needMask = false;
        if constexpr (maskType == CausalMaskT::MASK_TRIL) {
            inMaskLtFp32 = queMaskIn_.template AllocTensor<float>();
            if constexpr (std::is_same<MaskInfoType, uint32_t>::value) {
                // 处理 uint32_t 类型
                needMask = GenMask(
                    inMaskLtFp32,
                    maskinfo,
                    thisLen,
                    ((maskinfo > 0) ? (blockOffset) : n),  // blockOffset为行号
                    siluScale_);
            } else {
                // 处理 BlockMaskParams 类型
                BlockMaskGenerator blkMaskGen(maskinfo);
                needMask = blkMaskGen.GenMask(inMaskLtFp32, blockOffset, thisLen / blockM_, blockM_);
            }

            queMaskIn_.EnQue(inMaskLtFp32);
        } else if constexpr (maskType == CausalMaskT::MASK_CUSTOM) {
            int64_t thisMaskOffset = maskOffset + blockOffset * maxSeqLenK_;

            inMaskLt = queMaskIn_.template AllocTensor<qType>();
            DataCopyMayPad(inMaskLt, attnMaskGt_, (uint16_t)(thisLen / blockM_), n, thisMaskOffset);
            queMaskIn_.EnQue(inMaskLt);

            needMask = true;
        }

        return needMask;
    }

    __aicore__ inline bool GenMask(LocalTensor<float>& inMaskLt, int causalMask, int64_t maskLen,
        int64_t maskOffset, float sclae)
    {
        bool needMask = false;
        if (causalMask == 1) {
            DoCausalMask<float, CausalMaskT::MASK_TRIL>(inMaskLt, maskOffset, maskLen, blockM_, maskLen / blockM_,
                sclae);
            needMask = true;
        }

        return needMask;
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

    __aicore__ inline void DoMaskOptional(
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& newOutLt,
        int64_t thisLen,
        bool needMask)
    {
        if constexpr (maskType != CausalMaskT::MASK_NONE) {
            queMaskIn_.DeQue();
        }

        if (needMask) {
            if constexpr (maskType == CausalMaskT::MASK_CUSTOM) {
                inMaskLtFp32 = inMaskLt.template ReinterpretCast<float>();
                CastQtype2Float(inMaskLtFp32, inMaskLt, tmpLt, thisLen);
                Muls<float>(inMaskLtFp32, inMaskLtFp32, siluScale_, thisLen);
            }
            Mul<float>(newOutLt, newOutLt, inMaskLtFp32, thisLen);
        } else {
            Muls<float>(newOutLt, newOutLt, siluScale_, thisLen);
        }

        if constexpr (maskType != CausalMaskT::MASK_NONE) {
            queMaskIn_.FreeTensor(inMaskLtFp32);
        }
    }

    float siluScale_;
    float alpha_;

    CausalMaskT maskType_;
    int blockM_;
    int blockN_;
    int64_t maxSeqLenK_;

    TPipe *pipe_;

    TQue<TPosition::VECCALC, USE_QUEUE_NUM> tmpBuff_;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queIn_;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> biasIn_;
    TQue<TPosition::VECIN, USE_QUEUE_NUM> queMaskIn_;
    TQue<TPosition::VECOUT, USE_QUEUE_NUM> queOut_;

    GlobalTensor<qType> attnMaskGt_;
};


// 带bias的vectorScore计算
template <typename qType, CausalMaskT maskType, typename MaskInfoType>
class VectorScoreWithRab :
    public VectorScoreInter<qType, maskType, MaskInfoType, VectorScoreWithRab<qType, maskType, MaskInfoType>> {
public:
    __aicore__ inline VectorScoreWithRab() {}

    __aicore__ inline void Init(TPipe* pipePtr, VecScoreRabGtInfo<qType>& gtInfo, float siluScale, float alpha,
        int blockM, int blockN, int64_t maxSeqLenK)
    {
        this->InitParams(pipePtr, gtInfo, siluScale, alpha, blockM, blockN, maxSeqLenK);
        attnBiasGt_ = gtInfo.attnBiasGt;
    }

    __aicore__ inline void VecScoreImpl(const VecScoreRabParam<MaskInfoType>& params, GlobalTensor<qType>& attnScoreGt)
    {
        int64_t offset = params.srcOffset;
        int64_t total = params.m * this->blockM_;

        auto tmpLt = this->tmpBuff_.template AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> biasLt;
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;

        int64_t remain = total;
        while (remain > 0) {
            int64_t thisLen = VECTOR_SCORE_UB_BLOCK_ELEM;
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t thisOffset = offset + (total - remain);
            auto inLt = this->queIn_.template AllocTensor<qType>();
            DataCopy(inLt, attnScoreGt[thisOffset], thisLen);

            this->queIn_.EnQue(inLt);

            int64_t blockOffset = (total - remain) / this->blockM_;
            DoBiasCopy(biasLt, params.biasOffset, thisLen, blockOffset, params.n);

            bool needMask = this->DoMaskInitOptional(inMaskLt, inMaskLtFp32, params.maskinfo,
                params.maskOffset, thisLen, blockOffset, params.n);

            CalcuVecScore(inLt, biasLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32, needMask, thisLen);

            auto outLt = this->queOut_.template DeQue<qType>();
            DataCopy(attnScoreGt[thisOffset], outLt, thisLen);
            this->queOut_.FreeTensor(outLt);

            remain = remain - thisLen;
        }

        this->tmpBuff_.FreeTensor(tmpLt);
    }

    __aicore__ inline void DoBiasCopy(LocalTensor<qType>& biasLt, int64_t biasOffset, int64_t thisLen,
        int64_t blockOffset, uint32_t n)
    {
        int64_t thisBiasOffset = biasOffset + blockOffset * this->maxSeqLenK_;
        biasLt = this->biasIn_.template AllocTensor<qType>();
        this->DataCopyMayPad(biasLt, attnBiasGt_, (uint16_t)(thisLen / this->blockM_), n, thisBiasOffset);
        this->biasIn_.EnQue(biasLt);
    }

    __aicore__ inline void CalcuVecScore(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& tmpLtFp32,
        bool needMask,
        int64_t thisLen)
    {
        this->queIn_.DeQue();
        auto newInLt = inLt.template ReinterpretCast<float>();
        this->CastQtype2Float(newInLt, inLt, tmpLt, thisLen);
        DoBias(newInLt, biasLt, tmpLt, thisLen);

        auto outLt = this->queOut_.template AllocTensor<qType>();
        auto newOutLt = outLt.template ReinterpretCast<float>();
        Muls<float>(newInLt, newInLt, this->alpha_, thisLen);
        Silu<float>(newOutLt, newInLt, thisLen);

        this->queIn_.FreeTensor(inLt);
        this->DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, newOutLt, thisLen, needMask);
        this->CastFloat2Qtype(outLt, newOutLt, tmpLtFp32, thisLen);
        this->queOut_.EnQue(outLt);
    }

    __aicore__ inline void DoBias(
        LocalTensor<float>& newInLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& tmpLt,
        int64_t thisLen)
    {
        this->biasIn_.DeQue();
        auto newBiasLt = biasLt.template ReinterpretCast<float>();
        this->CastQtype2Float(newBiasLt, biasLt, tmpLt, thisLen);
        Add<float>(newInLt, newInLt, newBiasLt, thisLen);
        this->biasIn_.FreeTensor(biasLt);
    }

    GlobalTensor<qType> attnBiasGt_;
};

// 不带bias的vectorScore计算
template <typename qType, CausalMaskT maskType, typename MaskInfoType>
class VectorScoreWithoutRab :
    public VectorScoreInter<qType, maskType, MaskInfoType, VectorScoreWithoutRab<qType, maskType, MaskInfoType>> {
public:
    __aicore__ inline VectorScoreWithoutRab() {}

    __aicore__ inline void Init(TPipe* pipePtr, VecScoreRabGtInfo<qType>& gtInfo, float siluScale, float alpha,
        int blockM, int blockN, int64_t maxSeqLenK)
    {
        this->InitParams(pipePtr, gtInfo, siluScale, alpha, blockM, blockN, maxSeqLenK);
    }

    __aicore__ inline void VecScoreImpl(const VecScoreRabParam<MaskInfoType>& params, GlobalTensor<qType>& attnScoreGt)
    {
        int64_t total = params.m * this->blockM_;
        int64_t offset = params.srcOffset;

        auto tmpLt = this->tmpBuff_.template AllocTensor<qType>();
        auto tmpLtFp32 = tmpLt.template ReinterpretCast<float>();
        LocalTensor<qType> biasLt;
        LocalTensor<qType> inMaskLt;
        LocalTensor<float> inMaskLtFp32;

        biasLt = this->biasIn_.template AllocTensor<qType>();

        int64_t remain = total;
        while (remain > 0) {
            int64_t thisLen = VECTOR_SCORE_UB_BLOCK_ELEM;
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t thisOffset = offset + (total - remain);
            auto inLt = this->queIn_.template AllocTensor<qType>();

            DataCopy(inLt, attnScoreGt[thisOffset], thisLen);

            this->queIn_.EnQue(inLt);

            int64_t blockOffset = (total - remain) / this->blockM_;

            bool needMask = this->DoMaskInitOptional(inMaskLt, inMaskLtFp32, params.maskinfo,
                params.maskOffset, thisLen, blockOffset, params.n);

            CalcuVecScore(inLt, biasLt, inMaskLt, inMaskLtFp32, tmpLt, tmpLtFp32, needMask, thisLen);

            auto outLt = this->queOut_.template DeQue<qType>();
            DataCopy(attnScoreGt[thisOffset], outLt, thisLen);
            this->queOut_.FreeTensor(outLt);

            remain = remain - thisLen;
        }

        this->biasIn_.FreeTensor(biasLt);
        this->tmpBuff_.FreeTensor(tmpLt);
    }

    __aicore__ inline void CalcuVecScore(
        LocalTensor<qType>& inLt,
        LocalTensor<qType>& biasLt,
        LocalTensor<qType>& inMaskLt,
        LocalTensor<float>& inMaskLtFp32,
        LocalTensor<qType>& tmpLt,
        LocalTensor<float>& tmpLtFp32,
        bool needMask,
        int64_t thisLen)
    {
        if constexpr (!std::is_same<qType, float>::value) {
            this->queIn_.DeQue();
            Cast(tmpLtFp32, inLt, RoundMode::CAST_NONE, thisLen);
            this->queIn_.FreeTensor(inLt);

            auto biasLtFp32 = biasLt.template ReinterpretCast<float>();
            Muls<float>(tmpLtFp32, tmpLtFp32, this->alpha_, thisLen);
            Silu<float>(biasLtFp32, tmpLtFp32, thisLen);
            this->DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, biasLtFp32, thisLen, needMask);

            auto outLt = this->queOut_.template AllocTensor<qType>();
            Cast(outLt, biasLtFp32, RoundMode::CAST_RINT, thisLen);
            this->queOut_.EnQue(outLt);
        } else {
            this->queIn_.DeQue();

            auto outLt = this->queOut_.template AllocTensor<qType>();
            Muls<float>(inLt, inLt, this->alpha_, thisLen);
            Silu<float>(outLt, inLt, thisLen);
            this->queIn_.FreeTensor(inLt);

            this->DoMaskOptional(inMaskLt, inMaskLtFp32, tmpLt, outLt, thisLen, needMask);
            this->queOut_.EnQue(outLt);
        }
    }
};

} // HstuForward

#endif // VECTOR_SCORE_INTER_H