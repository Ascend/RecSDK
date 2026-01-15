/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#ifndef HSTU_VECTOR_SCORE_H
#define HSTU_VECTOR_SCORE_H

#include <cstdint>
#include "hstu_mask.h"
#include "hstu_common_const.h"

using BNSSShape = Shape<uint32_t, uint32_t, uint32_t, uint32_t>;
using BNSSStride = AscendC::Stride<uint32_t, uint32_t, uint32_t, uint32_t>;
using BNSSLayout = Layout<BNSSShape, BNSSStride>;

using HstuDenseBackward::BlockMaskGenerator;
using HstuDenseBackward::BlockMaskParams;

namespace HstuDenseBackward {

struct VectorScoreAttrs {
    float siluScale;
    float alpha;
    bool enableBias;
    MaskType maskType;
};

template <typename qType>
struct VectorScoreGtInfo {
    GlobalTensor<qType> qkTemp;
    GlobalTensor<qType> gvTemp;
    GlobalTensor<qType> maskGt;
    GlobalTensor<qType> biasGt;
    GlobalTensor<qType> biasGradGt;
};

// 静态接口实现多态的办法
template <typename qType, class VectorScoreStrategy>
class VectorScoreInterface {
public:
    __aicore__ inline VectorScoreInterface() {}

    __aicore__ inline void Init(TPipe* pipePtr, BNSSLayout bnssLayout, VectorScoreAttrs& attrs,
                                VectorScoreGtInfo<qType>& gtInfo)
    {
        static_cast<VectorScoreStrategy*>(this)->Init(pipePtr, bnssLayout, attrs, gtInfo);
    }

    __aicore__ inline void VecScoreJagged(int64_t tempOffset, int64_t batchId, int64_t headId, int64_t rowId,
                                          int64_t colId, int64_t totalRowNum, int64_t totalColNum,
                                          BlockMaskParams& blockMaskParam)
    {
        static_cast<VectorScoreStrategy*>(this)->VecScoreJagged(tempOffset, batchId, headId, rowId, colId, totalRowNum,
                                                                totalColNum, blockMaskParam);
    }
};

template <typename qType, uint32_t blockHeightQ, uint32_t blockHeightK>
class HstuF16R0VectorScore
    : public VectorScoreInterface<qType, HstuF16R0VectorScore<qType, blockHeightQ, blockHeightK>> {
public:
    __aicore__ inline HstuF16R0VectorScore() {}

    __aicore__ inline void Init(TPipe* pipePtr, BNSSLayout bnssLayout, VectorScoreAttrs& attrs,
                                VectorScoreGtInfo<qType>& gtInfo)
    {
        pipePtr_ = pipePtr;
        siluScale_ = attrs.siluScale;
        alpha_ = attrs.alpha;
        aivNum_ = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        qkTemp_ = gtInfo.qkTemp;
        gvTemp_ = gtInfo.gvTemp;
        // 20通过ub的大小计算得到
        vecOnceDataNum_ = blockHeightQ * 20;

        pipePtr_->InitBuffer(queueVecScoreQK_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(queueVecScoreGV_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(queueVecScoreMask_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(queueVecScoreBias_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMid_, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMidQk_, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMidGV_, vecOnceDataNum_ * sizeof(float));

        pipePtr_->InitBuffer(queueOutputScore_, 1, vecOnceDataNum_ * sizeof(qType));
        pipePtr_->InitBuffer(queueOutputBias_, 1, vecOnceDataNum_ * sizeof(qType));
    }

    __aicore__ inline void VecScoreJagged(int64_t tempOffset, int64_t batchId, int64_t headId, int64_t rowId,
                                          int64_t colId, int64_t totalRowNum, int64_t totalColNum,
                                          BlockMaskParams& blockMaskParam)
    {
        int64_t total = blockHeightQ * blockHeightK;
        int64_t remain = total;
        int64_t thisLen = vecOnceDataNum_;
        BlockMaskGenerator generator(&blockMaskParam);

        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t baseOffset = total - remain;
            int64_t startRowNum = baseOffset / blockHeightQ;
            int64_t thisRowNum = thisLen / blockHeightQ;
            int64_t validRowNum = totalRowNum - startRowNum;
            validRowNum = validRowNum > thisRowNum ? thisRowNum : validRowNum;
            validRowNum = validRowNum < 0 ? 0 : validRowNum;

            int64_t qkOffset = tempOffset + baseOffset;

            if (validRowNum > 0) {
                VecScoreWithValidRowNum(thisLen, validRowNum, totalColNum, startRowNum, qkOffset, generator);
            }

            remain = remain - thisLen;
        }
    }

    __aicore__ inline void VecScoreWithValidRowNum(int64_t thisLen, int64_t validRowNum, int64_t totalColNum,
                                                   int64_t rowInBlock, int64_t qkOffset, BlockMaskGenerator& generator)
    {
        const bool thisBlockNeedProcessMask = generator.NeedMask();
        int64_t gvOffset = qkOffset;
        int64_t scoreTempOffset = qkOffset;
        LocalTensor<float> inputQK = queueVecScoreQK_.AllocTensor<float>();
        DataCopy<qType>(inputQK.template ReinterpretCast<qType>(), qkTemp_[qkOffset], thisLen);
        queueVecScoreQK_.EnQue(inputQK);

        LocalTensor<float> inputGV = queueVecScoreGV_.AllocTensor<float>();
        DataCopy<qType>(inputGV.template ReinterpretCast<qType>(), gvTemp_[gvOffset], thisLen);
        queueVecScoreGV_.EnQue(inputGV);
        LocalTensor<float> inputMask = queueVecScoreMask_.AllocTensor<float>();
        if (thisBlockNeedProcessMask) {
            generator.GenMask(inputMask, rowInBlock, thisLen / blockHeightQ, blockHeightQ);
        }
        queueVecScoreMask_.EnQue(inputMask);

        CalcuScoreWithFloat32(thisLen, thisBlockNeedProcessMask);

        LocalTensor<qType> outputScore = queueOutputScore_.DeQue<qType>();
        LocalTensor<qType> outputBias = queueOutputBias_.DeQue<qType>();
        DataCopy<qType>(qkTemp_[scoreTempOffset], outputScore, thisLen);
        DataCopy<qType>(gvTemp_[scoreTempOffset], outputBias, thisLen);

        queueOutputScore_.FreeTensor(outputScore);
        queueOutputBias_.FreeTensor(outputBias);
    }

    __aicore__ inline void CalcuScoreWithFloat32(int64_t thisLen, bool thisBlockNeedProcessMask)
    {
        LocalTensor<float> inputQK = queueVecScoreQK_.template DeQue<float>();
        LocalTensor<float> inputGV = queueVecScoreGV_.template DeQue<float>();

        LocalTensor<float> midQK = tbufMidQk_.Get<float>();
        LocalTensor<float> midGV = tbufMidGV_.Get<float>();
        Cast(midQK, inputQK.template ReinterpretCast<qType>(), RoundMode::CAST_NONE, thisLen);
        Cast(midGV, inputGV.template ReinterpretCast<qType>(), RoundMode::CAST_NONE, thisLen);
        queueVecScoreGV_.FreeTensor(inputGV);
        queueVecScoreQK_.FreeTensor(inputQK);

        LocalTensor<float> inputMask = queueVecScoreMask_.DeQue<float>();
        LocalTensor<float> inputBias = queueVecScoreBias_.AllocTensor<float>();
        LocalTensor<float> dsiluTemp = tbufMid_.Get<float>();
        // 计算公式: inputQK inputGV inputMask inputBias
        // 计算公式:  v = qk_input * alpha
        Muls<float>(midQK, midQK, alpha_, thisLen);
        // 计算公式:  inputBias = sigmoid_fast(v);   sigmoid_v = sigmoid_fast(v);
        Sigmoid<float>(inputBias, midQK, thisLen);
        // 计算公式:  inputBias = inputBias * inputMask; sigmoid_v = sigmoid_v * mask
        if (thisBlockNeedProcessMask) {
            Mul<float>(inputBias, inputBias, inputMask, thisLen);
        }
        // 计算公式:  silu_out =  inputBias * inputQK  silu_out = v * sigmoid_v
        Mul<float>(inputMask, inputBias, midQK, thisLen);

        // 计算公式:  dsilu_temp = sigmoid_v * (1 + v * (1 - sigmoid_v))
        // 计算公式:  dsilu_temp = v - v*sigmoid_v
        Sub<float>(dsiluTemp, midQK, inputMask, thisLen);
        // 计算公式:  dsilu_temp = 1 + dsilu_temp
        Adds<float>(dsiluTemp, dsiluTemp, 1, thisLen);
        // 计算公式:  dsilu_temp = sigmoid_v * dsilu_temp
        Mul<float>(dsiluTemp, dsiluTemp, inputBias, thisLen);
        // 计算公式:  scoreTemp = silu_out * silu_scale
        Muls(inputMask, inputMask, siluScale_, thisLen);

        // 计算公式:  scoreGradTemp = gv_input * silu_scale * dsilu_temp * alpha
        Muls<float>(midGV, midGV, siluScale_ * alpha_, thisLen);
        Mul<float>(inputBias, midGV, dsiluTemp, thisLen);

        LocalTensor<qType> outputScore = queueOutputScore_.AllocTensor<qType>();
        LocalTensor<qType> outputBias = queueOutputBias_.AllocTensor<qType>();
        Cast(outputScore, inputMask, RoundMode::CAST_RINT, thisLen);
        Cast(outputBias, inputBias, RoundMode::CAST_RINT, thisLen);
        queueVecScoreMask_.FreeTensor(inputMask);
        queueVecScoreBias_.FreeTensor(inputBias);
        queueOutputScore_.EnQue(outputScore);
        queueOutputBias_.EnQue(outputBias);
    }

    // Attr
    float siluScale_ = 1.0f;
    float alpha_ = 1.0f;
    uint32_t aivNum_ = 0;

    // Tpipe
    TPipe* pipePtr_ = nullptr;

    // vec score
    int64_t vecOnceDataNum_ = 0;
    TQue<TPosition::VECIN, 1> queueVecScoreQK_;
    TQue<TPosition::VECIN, 1> queueVecScoreGV_;
    TQue<TPosition::VECIN, 1> queueVecScoreMask_;
    TQue<TPosition::VECIN, 1> queueVecScoreBias_;

    TQue<TPosition::VECOUT, 1> queueOutputScore_;
    TQue<TPosition::VECOUT, 1> queueOutputBias_;
    TBuf<TPosition::VECCALC> tbufMid_;
    TBuf<TPosition::VECCALC> tbufMidQk_;
    TBuf<TPosition::VECCALC> tbufMidGV_;

    // Gt
    GlobalTensor<qType> qkTemp_;
    GlobalTensor<qType> gvTemp_;
};

template <typename qType, uint32_t blockHeightQ, uint32_t blockHeightK>
class HstuVectorScoreCommon
    : public VectorScoreInterface<qType, HstuVectorScoreCommon<qType, blockHeightQ, blockHeightK>> {
public:
    __aicore__ inline HstuVectorScoreCommon() {}

    __aicore__ inline void Init(TPipe* pipePtr, BNSSLayout bnssLayout, VectorScoreAttrs& attrs,
                                VectorScoreGtInfo<qType>& gtInfo)
    {
        pipePtr_ = pipePtr;
        siluScale_ = attrs.siluScale;
        alpha_ = attrs.alpha;
        enableBias_ = attrs.enableBias;
        maskType_ = attrs.maskType;

        aivNum_ = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        qkTemp_ = gtInfo.qkTemp;
        gvTemp_ = gtInfo.gvTemp;
        maskGt_ = gtInfo.maskGt;
        biasGt_ = gtInfo.biasGt;
        biasGradGt_ = gtInfo.biasGradGt;
        bnssLayout_ = bnssLayout;
        maxSeqLen_ = AscendC::Std::get<2>(bnssLayout.GetShape());
        // 20通过ub的大小计算得到
        vecOnceDataNum_ = blockHeightQ * 15;

        pipePtr_->InitBuffer(queueVecScoreQK_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(queueVecScoreGV_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(queueVecScoreMask_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(queueVecScoreBias_, 1, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMid_, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMidQk_, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMidGV_, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMidBias_, vecOnceDataNum_ * sizeof(float));
        pipePtr_->InitBuffer(tbufMidMask_, vecOnceDataNum_ * sizeof(float));

        pipePtr_->InitBuffer(queueOutputScore_, 1, vecOnceDataNum_ * sizeof(qType));
        pipePtr_->InitBuffer(queueOutputBias_, 1, vecOnceDataNum_ * sizeof(qType));
    }

    __aicore__ inline void CopyInPadding(LocalTensor<qType> dstTensor, GlobalTensor<qType> srcTensor, int64_t rowNum,
                                         int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (seqLen - colNum) * sizeof(qType);
        uint32_t dstStride = (blockHeightQ - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint8_t rightPadding = (blockHeightQ - colNum) % (DATA_ALIGN_BYTES / sizeof(qType));

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPadExtParams<qType> padParams{true, 0, rightPadding, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    __aicore__ inline void CopyOutPadding(GlobalTensor<qType> dstTensor, LocalTensor<qType> srcTensor, int64_t rowNum,
                                          int64_t colNum, int64_t seqLen)
    {
        uint16_t blockCount = rowNum;
        uint32_t blockLen = colNum * sizeof(qType);
        uint32_t srcStride = (blockHeightQ - colNum) / (DATA_ALIGN_BYTES / sizeof(qType));
        uint32_t dstStride = (seqLen - colNum) * sizeof(qType);

        DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void VecScoreJagged(int64_t tempOffset, int64_t batchId, int64_t headId, int64_t rowId,
                                          int64_t colId, int64_t totalRowNum, int64_t totalColNum,
                                          BlockMaskParams& blockMaskParam)
    {
        int64_t total = blockHeightQ * blockHeightK;
        int64_t remain = total;
        int64_t thisLen = vecOnceDataNum_;
        BlockMaskGenerator generator(&blockMaskParam);
        uint32_t rowOffsetInSeq = rowId * blockHeightQ;
        uint32_t colOffsetInSeq = colId * blockHeightK;
        while (remain > 0) {
            if (remain < thisLen) {
                thisLen = remain;
            }

            int64_t baseOffset = total - remain;
            int64_t startRowNum = baseOffset / blockHeightQ;
            int64_t thisRowNum = thisLen / blockHeightQ;
            int64_t validRowNum = totalRowNum - startRowNum;
            validRowNum = validRowNum > thisRowNum ? thisRowNum : validRowNum;
            validRowNum = validRowNum < 0 ? 0 : validRowNum;

            int64_t qkOffset = tempOffset + baseOffset;
            int64_t bnssOffset = bnssLayout_(MakeCoord(batchId, headId, rowOffsetInSeq + startRowNum, colOffsetInSeq));
            if (validRowNum > 0) {
                VecScoreWithValidRowNum(thisLen, validRowNum, totalColNum, startRowNum, qkOffset, bnssOffset,
                                        generator);
            }
            if (enableBias_ && maskType_ == MaskType::MASK_TRIL && blockMaskParam.DiagonalNoComputation()) {
                LocalTensor<qType> outputTempTensor = queueOutputBias_.AllocTensor<qType>();
                Duplicate<qType>(outputTempTensor, 0, thisLen);
                queueOutputBias_.EnQue(outputTempTensor);
                int64_t curAttnBiasDiagonalOffset =
                    bnssLayout_(MakeCoord(batchId, headId, colOffsetInSeq + startRowNum, rowOffsetInSeq));
                outputTempTensor = queueOutputBias_.DeQue<qType>();
                CopyOutPadding(biasGradGt_[curAttnBiasDiagonalOffset], outputTempTensor, thisRowNum, totalRowNum,
                               maxSeqLen_);
                queueOutputBias_.FreeTensor(outputTempTensor);
            }
            remain = remain - thisLen;
        }
    }

    __aicore__ inline void VecScoreWithValidRowNum(int64_t thisLen, int64_t validRowNum, int64_t totalColNum,
                                                   int64_t rowInBlock, int64_t qkOffset, int64_t bnssOffset,
                                                   BlockMaskGenerator& generator)
    {
        bool thisBlockNeedProcessMask = false;
        if (maskType_ == MaskType::MASK_TRIL) {
            thisBlockNeedProcessMask = generator.NeedMask();
        } else if (maskType_ == MaskType::MASK_CUSTOM) {
            thisBlockNeedProcessMask = true;
        }
        int64_t gvOffset = qkOffset;
        int64_t scoreTempOffset = qkOffset;
        LocalTensor<float> inputQK = queueVecScoreQK_.AllocTensor<float>();
        DataCopy<qType>(inputQK.template ReinterpretCast<qType>(), qkTemp_[qkOffset], thisLen);
        queueVecScoreQK_.EnQue(inputQK);

        LocalTensor<float> inputGV = queueVecScoreGV_.AllocTensor<float>();
        DataCopy<qType>(inputGV.template ReinterpretCast<qType>(), gvTemp_[gvOffset], thisLen);
        queueVecScoreGV_.EnQue(inputGV);

        LocalTensor<float> inputMask = queueVecScoreMask_.AllocTensor<float>();
        if (maskType_ == MaskType::MASK_CUSTOM) {
            // DataCopy<qType>(inputMask.template ReinterpretCast<qType>(), maskGt_[bnssOffset], thisLen);
            CopyInPadding(inputMask.template ReinterpretCast<qType>(), maskGt_[bnssOffset], validRowNum, totalColNum,
                          maxSeqLen_);
        } else if (thisBlockNeedProcessMask) {
            generator.GenMask(inputMask, rowInBlock, thisLen / blockHeightQ, blockHeightQ);
        }
        queueVecScoreMask_.EnQue(inputMask);
        LocalTensor<float> inputBias = queueVecScoreBias_.AllocTensor<float>();
        if (enableBias_) {
            CopyInPadding(inputBias.template ReinterpretCast<qType>(), biasGt_[bnssOffset], validRowNum, totalColNum,
                          maxSeqLen_);
        }
        queueVecScoreBias_.EnQue(inputBias);

        CalcuScoreWithFloat32(thisLen, thisBlockNeedProcessMask);

        LocalTensor<qType> outputScore = queueOutputScore_.DeQue<qType>();
        LocalTensor<qType> outputBias = queueOutputBias_.DeQue<qType>();
        DataCopy<qType>(qkTemp_[scoreTempOffset], outputScore, thisLen);
        DataCopy<qType>(gvTemp_[scoreTempOffset], outputBias, thisLen);
        if (enableBias_) {
            CopyOutPadding(biasGradGt_[bnssOffset], outputBias, validRowNum, totalColNum, maxSeqLen_);
        }

        queueOutputScore_.FreeTensor(outputScore);
        queueOutputBias_.FreeTensor(outputBias);
    }

    __aicore__ inline void Convert2Float(LocalTensor<float>& dstTensor, LocalTensor<float>& srcTensor, int64_t thisLen)
    {
        if constexpr (std::is_same<qType, float>::value) {
            DataCopy<float>(dstTensor, srcTensor, thisLen);
        } else {
            Cast(dstTensor, srcTensor.ReinterpretCast<qType>(), RoundMode::CAST_NONE, thisLen);
        }
    }
    __aicore__ inline void CalcuScoreWithFloat32(int64_t thisLen, bool thisBlockNeedProcessMask)
    {
        LocalTensor<float> inputQK = queueVecScoreQK_.template DeQue<float>();
        LocalTensor<float> inputGV = queueVecScoreGV_.template DeQue<float>();
        LocalTensor<float> inputMask = queueVecScoreMask_.DeQue<float>();
        LocalTensor<float> inputBias = queueVecScoreBias_.DeQue<float>();

        LocalTensor<float> midQK = tbufMidQk_.Get<float>();
        LocalTensor<float> midGV = tbufMidGV_.Get<float>();
        LocalTensor<float> midBias = tbufMidBias_.Get<float>();
        LocalTensor<float> midMask = tbufMidMask_.Get<float>();

        Convert2Float(midQK, inputQK, thisLen);
        Convert2Float(midGV, inputGV, thisLen);
        Convert2Float(midBias, inputBias, thisLen);
        Convert2Float(midMask, inputMask, thisLen);

        if (maskType_ == MaskType::MASK_CUSTOM) {
            Convert2Float(midMask, inputMask, thisLen);
        } else if (maskType_ == MaskType::MASK_TRIL && thisBlockNeedProcessMask) {
            DataCopy<float>(midMask, inputMask, thisLen);
        }

        queueVecScoreGV_.FreeTensor(inputGV);
        queueVecScoreQK_.FreeTensor(inputQK);
        queueVecScoreMask_.FreeTensor(inputMask);
        queueVecScoreBias_.FreeTensor(inputBias);

        LocalTensor<float> dsiluTemp = tbufMid_.Get<float>();
        if (enableBias_) {
            // 计算公式: qkb = qk + attn_bias
            Add<float>(midQK, midQK, midBias, thisLen);
        }
        // 计算公式: inputQK inputGV inputMask inputBias
        // 计算公式:  v = qk_input * alpha
        Muls<float>(midQK, midQK, alpha_, thisLen);
        // 计算公式:  inputBias = sigmoid_fast(v)表示sigmoid_v = sigmoid_fast(v);
        Sigmoid<float>(midBias, midQK, thisLen);
        // 计算公式:  inputBias = inputBias * inputMask; sigmoid_v = sigmoid_v * mask
        if (thisBlockNeedProcessMask) {
            Mul<float>(midBias, midBias, midMask, thisLen);
        }
        // 计算公式:  silu_out =  inputBias * inputQK  silu_out = v * sigmoid_v
        Mul<float>(midMask, midBias, midQK, thisLen);

        // 计算公式: dsilu_temp = sigmoid_v * (1 + v * (1 - sigmoid_v))
        // 计算公式:  dsilu_temp = v - v*sigmoid_v
        Sub<float>(dsiluTemp, midQK, midMask, thisLen);
        // 计算公式: dsilu_temp = 1 + dsilu_temp
        Adds<float>(dsiluTemp, dsiluTemp, 1, thisLen);
        // 计算公式: dsilu_temp = sigmoid_v * dsilu_temp
        Mul<float>(dsiluTemp, dsiluTemp, midBias, thisLen);
        // 计算公式: scoreTemp = silu_out * silu_scale
        Muls(midMask, midMask, siluScale_, thisLen);

        // 计算公式: scoreGradTemp = gv_input * silu_scale * dsilu_temp * alpha
        Muls<float>(midGV, midGV, siluScale_ * alpha_, thisLen);
        Mul<float>(midBias, midGV, dsiluTemp, thisLen);

        LocalTensor<qType> outputScore = queueOutputScore_.AllocTensor<qType>();
        LocalTensor<qType> outputBias = queueOutputBias_.AllocTensor<qType>();
        if constexpr (std::is_same<qType, float>::value) {
            DataCopy<qType>(outputScore, midMask, thisLen);
            DataCopy<qType>(outputBias, midBias, thisLen);
        } else {
            Cast(outputScore, midMask, RoundMode::CAST_RINT, thisLen);
            Cast(outputBias, midBias, RoundMode::CAST_RINT, thisLen);
        }
        queueOutputScore_.EnQue(outputScore);
        queueOutputBias_.EnQue(outputBias);
    }
    // Mode
    bool enableBias_ = false;
    MaskType maskType_ = MaskType::MASK_NONE;

    // Attr
    float siluScale_ = 1.0f;
    float alpha_ = 1.0f;
    uint32_t aivNum_ = 0;
    uint32_t maxSeqLen_ = 0;

    // Tpipe
    TPipe* pipePtr_ = nullptr;

    // vec score
    int64_t vecOnceDataNum_ = 0;
    TQue<TPosition::VECIN, 1> queueVecScoreQK_;
    TQue<TPosition::VECIN, 1> queueVecScoreGV_;
    TQue<TPosition::VECIN, 1> queueVecScoreMask_;
    TQue<TPosition::VECIN, 1> queueVecScoreBias_;

    TQue<TPosition::VECOUT, 1> queueOutputScore_;
    TQue<TPosition::VECOUT, 1> queueOutputBias_;
    TBuf<TPosition::VECCALC> tbufMid_;
    TBuf<TPosition::VECCALC> tbufMidQk_;
    TBuf<TPosition::VECCALC> tbufMidGV_;
    TBuf<TPosition::VECCALC> tbufMidBias_;
    TBuf<TPosition::VECCALC> tbufMidMask_;
    // Gt
    GlobalTensor<qType> qkTemp_;
    GlobalTensor<qType> gvTemp_;
    GlobalTensor<qType> maskGt_;
    GlobalTensor<qType> biasGt_;
    GlobalTensor<qType> biasGradGt_;

    // Layout
    BNSSLayout bnssLayout_;
};
}  // namespace HstuDenseBackward
#endif