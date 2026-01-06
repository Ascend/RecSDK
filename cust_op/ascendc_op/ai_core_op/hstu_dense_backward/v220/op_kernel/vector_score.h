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

#include "hstu_mask.h"
#include "hstu_common_const.h"

using HstuDenseBackward::BlockMaskGenerator;
using HstuDenseBackward::BlockMaskParams;

namespace HstuDenseBackward {

struct VectorScoreAttrs {
    float siluScale;
    float alpha;
};

struct VectorScoreGtInfo {
    GM_ADDR qkTemp;
    GM_ADDR gvTemp;
    GM_ADDR maskTemp;
    GM_ADDR biasTemp;
};

// 静态接口实现多态的办法
template <class VectorScoreStrategy>
class VectorScoreInterface {
public:
    __aicore__ inline VectorScoreInterface() {}

    __aicore__ inline void Init(TPipe* pipePtr, VectorScoreAttrs* attrs, VectorScoreGtInfo* gtInfo)
    {
        static_cast<VectorScoreStrategy*>(this)->Init(pipePtr, attrs, gtInfo);
    }

    __aicore__ inline void VecScoreJagged(int64_t taskId, int64_t batchId, int64_t headId, int64_t rowId, int64_t colId,
                                          int64_t totalRowNum, int64_t totalColNum, BlockMaskParams& blockMaskParam)
    {
        static_cast<VectorScoreStrategy*>(this)->VecScoreJagged(taskId, batchId, headId, rowId, colId, totalRowNum,
                                                                totalColNum, blockMaskParam);
    }
};

template <typename qType, uint32_t blockHeightQ, uint32_t blockHeightK, uint32_t headDim>
class HstuF16R0VectorScore
    : public VectorScoreInterface<HstuF16R0VectorScore<qType, blockHeightQ, blockHeightK, headDim>> {
public:
    __aicore__ inline HstuF16R0VectorScore() {}

    __aicore__ inline void Init(TPipe* pipePtr, VectorScoreAttrs* attrs, VectorScoreGtInfo* gtInfo)
    {
        pipePtr_ = pipePtr;
        siluScale_ = attrs->siluScale;
        alpha_ = attrs->alpha;
        aivNum_ = GetBlockNum() * VCORE_NUM_IN_ONE_AIC;
        qkTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(gtInfo->qkTemp));
        gvTemp_.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(gtInfo->gvTemp));
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
        // 变量 inputQK inputGV inputMask inputBias
        // v = qk_input * alpha
        Muls<float>(midQK, midQK, alpha_, thisLen);
        // inputBias = sigmoid_fast(v);   sigmoid_v = sigmoid_fast(v);
        Sigmoid<float>(inputBias, midQK, thisLen);
        // inputBias = inputBias * inputMask; sigmoid_v = sigmoid_v * mask
        if (thisBlockNeedProcessMask) {
            Mul<float>(inputBias, inputBias, inputMask, thisLen);
        }
        // silu_out =  inputBias * inputQK  silu_out = v * sigmoid_v
        Mul<float>(inputMask, inputBias, midQK, thisLen);

        // dsilu_temp = sigmoid_v * (1 + v * (1 - sigmoid_v))
        // dsilu_temp = v - v*sigmoid_v
        Sub<float>(dsiluTemp, midQK, inputMask, thisLen);
        // dsilu_temp = 1 + dsilu_temp
        Adds<float>(dsiluTemp, dsiluTemp, 1, thisLen);
        // dsilu_temp = sigmoid_v * dsilu_temp
        Mul<float>(dsiluTemp, dsiluTemp, inputBias, thisLen);
        // scoreTemp = silu_out * silu_scale
        Muls(inputMask, inputMask, siluScale_, thisLen);

        // scoreGradTemp = gv_input * silu_scale * dsilu_temp * alpha
        Muls<float>(midGV, midGV, siluScale_ * alpha_, thisLen);
        Mul<float>(inputBias, midGV, dsiluTemp, thisLen);
        // Muls<float>(inputGV, inputGV, alpha_, thisLen);

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
}  // namespace HstuDenseBackward
#endif