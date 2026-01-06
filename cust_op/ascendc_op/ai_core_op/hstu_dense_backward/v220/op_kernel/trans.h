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

#ifndef HSTU_TRANS_H
#define HSTU_TRANS_H

#include <cstdint>
#include "hstu_mask.h"
#include "hstu_common_const.h"

using HstuDenseBackward::BlockMaskGenerator;
using HstuDenseBackward::BlockMaskParams;

namespace HstuDenseBackward {

template <typename fromType, typename toType>
class TransStrideHdDKernel {
public:
    __aicore__ inline TransStrideHdDKernel() {}

    __aicore__ inline void Init(TPipe* pipePtr, uint32_t headNum, uint32_t headDim)
    {
        pipe_ = pipePtr;
        headNum_ = headNum;
        headDim_ = headDim;
        vecOnceDataNum_ = UB_SIZE / (sizeof(fromType) + sizeof(toType));
        vecOnceDataNum_ = vecOnceDataNum_ / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        vecOnceDataNum_ = vecOnceDataNum_ * headDim / headDim;

        const uint32_t inputUbLen = vecOnceDataNum_ * sizeof(fromType);
        const uint32_t outputUbLen = vecOnceDataNum_ * sizeof(toType);

        inputLt_ = {TPosition::VECIN, 0, vecOnceDataNum_};
        outputLt_ = {TPosition::VECOUT, inputUbLen, vecOnceDataNum_};
    }

    __aicore__ inline void DoTransOfStrideHeadDim(GlobalTensor<fromType>& from, GlobalTensor<toType>& to,
                                                  int64_t fromOffset, int64_t toOffset, int64_t total = 0)
    {
        int64_t remain = total;
        int64_t copyLenEachLoopAlignHeadDim = vecOnceDataNum_ / headDim_ * headDim_;
        int64_t thisLen = copyLenEachLoopAlignHeadDim;
        while (remain > 0) {
            if (thisLen > remain) {
                thisLen = remain;
            }

            int64_t curFromOffset = total - remain;
            int64_t curToOffset = curFromOffset * headNum_;

            DataCopy(inputLt_, from[fromOffset + curFromOffset], thisLen);
            // 1.做UB Copy需要等待次轮的MTE2完成
            DoVWhenMte2Finish(pipe_);

            // 2.做UB Copy需要等待上一轮的MTE3完成
            DoVWhenMte3Finish(pipe_);

            if constexpr (std::is_same<toType, fromType>::value) {
                DataCopy(outputLt_, inputLt_, thisLen);
            } else {
                Cast(outputLt_, inputLt_, RoundMode::CAST_RINT, thisLen);
            }

            // 3.下一轮的MTE2需要等待当前UB Copy完成
            DoMte2WhenVFinish(pipe_);

            uint16_t blockCount = thisLen / headDim_;
            uint16_t blockLen = headDim_ * sizeof(toType) / DATA_ALIGN_BYTES;
            uint16_t dstStride = (headNum_ * headDim_ - headDim_) * sizeof(toType) / DATA_ALIGN_BYTES;
            DataCopyParams copyParams{blockCount, blockLen, 0, dstStride};

            // 4.下一轮的MTE3需要等待当前UB Copy完成
            DoMte3WhenVFinish(pipe_);

            DataCopy(to[toOffset + curToOffset], outputLt_, copyParams);

            remain = remain - thisLen;
        }
        // 5.Trans需要等待MTE3完成
        DoSWhenMte3Finish(pipe_);
    }

    uint32_t headNum_ = 0;
    uint32_t headDim_ = 0;
    uint32_t vecOnceDataNum_ = 0;
    LocalTensor<fromType> inputLt_;
    LocalTensor<toType> outputLt_;
    TPipe* pipe_ = nullptr;
};
}  // namespace HstuDenseBackward
#endif