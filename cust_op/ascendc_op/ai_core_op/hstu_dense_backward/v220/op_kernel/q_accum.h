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

#ifndef HSTU_Q_ACCUM_H
#define HSTU_Q_ACCUM_H

#include "hstu_common_const.h"

namespace HstuDenseBackward {

template <typename fromType, typename toType, typename seqType>
class qBlockAccumKernel {
public:
    __aicore__ inline qBlockAccumKernel() {}

    __aicore__ inline void Init(TPipe* pipePtr, BaseShapeArgs* baseShapeArgs, GlobalTensor<fromType>& qGradAccumTempGt,
                                GlobalTensor<toType>& qGradGt, GlobalTensor<seqType>& seqOffsetsGt, uint32_t aivNum)
    {
        qGradAccumTempGt_ = qGradAccumTempGt;
        qGradGt_ = qGradGt;
        seqOffsetsQGt_ = seqOffsetsGt;
        pipe_ = pipePtr;
        aivNum_ = aivNum;
        batchSize_ = baseShapeArgs->batchSize;
        headNum_ = baseShapeArgs->headNum;
        headDimQK_ = baseShapeArgs->headDimQK;
        headDimQKAlign32_ = baseShapeArgs->headDimQKAlign32;
        vecOnceDataNum_ = UB_SIZE / (sizeof(fromType) + sizeof(toType));
        vecOnceDataNum_ = vecOnceDataNum_ / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        vecOnceDataNum_ = vecOnceDataNum_ / headDimQKAlign32_ * headDimQKAlign32_;

        const uint32_t inputUbLen = vecOnceDataNum_ * sizeof(fromType);
        const uint32_t outputUbLen = vecOnceDataNum_ * sizeof(toType);

        inputLt_ = {TPosition::VECIN, 0, vecOnceDataNum_};
        outputLt_ = {TPosition::VECOUT, inputUbLen, vecOnceDataNum_};
    }

    __aicore__ inline void DoCopyQGrad()
    {
        uint32_t batchIdx = GetBlockIdx();
        uint32_t taskNum = batchSize_ * headNum_;
        uint32_t coreTask = taskNum / aivNum_;
        uint32_t coreSplitId = taskNum % aivNum_;

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
            uint32_t thisBatchIdx = (offsetOfThisCore + taskId) / headNum_;
            uint32_t headIdx = (offsetOfThisCore + taskId) % headNum_;

            int64_t curSeqLen =
                static_cast<int64_t>(seqOffsetsQGt_.GetValue(thisBatchIdx + 1) - seqOffsetsQGt_.GetValue(thisBatchIdx));
            DoCopyBlockQGrad(thisBatchIdx, headIdx, curSeqLen);
        }
    }

    __aicore__ inline void DoCopyBlockQGrad(int64_t thisBatchIdx, int64_t headIdx, int64_t curSeqLen)
    {
        int64_t totalLen = curSeqLen * headDimQKAlign32_;
        int64_t remain = totalLen;
        int64_t thisLen = vecOnceDataNum_;
        int64_t thisBatchInOffset = seqOffsetsQGt_.GetValue(thisBatchIdx) * headDimQKAlign32_ * headNum_;
        int64_t BasicInOffset = thisBatchInOffset + (headIdx * totalLen);
        int64_t thisBatchOutOffset = seqOffsetsQGt_.GetValue(thisBatchIdx) * headDimQK_ * headNum_;
        int64_t BasicOutOffset = thisBatchOutOffset + headIdx * headDimQK_;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curOffset = BasicInOffset + (totalLen - remain);

            DataCopy<fromType>(inputLt_, qGradAccumTempGt_[curOffset], thisLen);
            PipeBarrier<PIPE_ALL>();
            if constexpr (std::is_same<toType, fromType>::value) {
                DataCopy(outputLt_, inputLt_, thisLen);
            } else {
                Cast(outputLt_, inputLt_, RoundMode::CAST_RINT, thisLen);
            }
            PipeBarrier<PIPE_ALL>();

            int64_t curOutOffset = BasicOutOffset + (totalLen - remain) / headDimQKAlign32_ * headDimQK_ * headNum_;
            uint16_t blockCount = thisLen / headDimQKAlign32_;

            if ((headDimQK_ * sizeof(toType)) % DATA_ALIGN_BYTES == 0) {
                uint16_t blockLen = headDimQK_ * sizeof(toType) / DATA_ALIGN_BYTES;
                uint16_t dstStride = (headNum_ - 1) * headDimQK_ * sizeof(toType) / DATA_ALIGN_BYTES;
                DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};
                DataCopy<toType>(qGradGt_[curOutOffset], outputLt_, copyParams);
            } else {
                uint16_t blockLen = headDimQK_ * sizeof(toType);
                uint16_t dstStride = (headNum_ - 1) * headDimQK_ * sizeof(toType);
                DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};
                DataCopyPad<toType>(qGradGt_[curOutOffset], outputLt_, copyParams);
            }

            PipeBarrier<PIPE_ALL>();
            remain = remain - thisLen;
        }
    }

    uint32_t batchSize_ = 0;
    uint32_t headNum_ = 0;
    uint32_t headDimQK_ = 0;
    uint32_t headDimQKAlign32_ = 0;
    uint32_t vecOnceDataNum_ = 0;
    uint32_t aivNum_ = 0;
    LocalTensor<fromType> inputLt_;
    LocalTensor<toType> outputLt_;
    GlobalTensor<fromType> qGradAccumTempGt_;
    GlobalTensor<toType> qGradGt_;
    GlobalTensor<seqType> seqOffsetsQGt_;
    TPipe* pipe_ = nullptr;
};
}  // namespace HstuDenseBackward
#endif