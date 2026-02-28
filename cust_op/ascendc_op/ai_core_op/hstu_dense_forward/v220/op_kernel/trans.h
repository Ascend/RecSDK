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

#ifndef TRANS_H
#define TRANS_H

#include "kernel_operator.h"
#include "hstu_common_const.h"

using namespace HstuForward;

template <typename fromType, typename toType>
class Trans {
public:
    __aicore__ inline Trans() {}

    __aicore__ inline void Init(TPipe* pipe, int64_t headNum, int64_t dim, int64_t blockK)
    {
        pipe_ = pipe;

        headNum_ = headNum;
        dim_ = dim;
        blockK_ = blockK;

        vecOnceDataNum_ = UB_SIZE / (sizeof(fromType) + sizeof(toType));
        vecOnceDataNum_ = vecOnceDataNum_ / DATA_ALIGN_BYTES * DATA_ALIGN_BYTES;
        vecOnceDataNum_ = vecOnceDataNum_ / dim_ * dim_;

        const uint32_t inputUbLen = vecOnceDataNum_ * sizeof(fromType);
        const uint32_t outputUbLen = vecOnceDataNum_ * sizeof(toType);

        inputLt_ = { TPosition::VECIN, 0, vecOnceDataNum_ };
        outputLt_ = { TPosition::VECOUT, inputUbLen, vecOnceDataNum_ };
    }

    template <bool needAtomic = false>
    __aicore__ inline void TransResult(GlobalTensor<fromType>& fromGt, GlobalTensor<toType>& toGt,
        int64_t fromOffset, int64_t toOffset, int64_t total)
    {
        int64_t remain = total;

        DataCopyParams srcCopyParams;
        srcCopyParams.blockLen = dim_ * sizeof(fromType) / DATA_ALIGN_BYTES;
        srcCopyParams.srcStride = (blockK_ - dim_) * sizeof(fromType) / DATA_ALIGN_BYTES;
        srcCopyParams.dstStride = 0;

        DataCopyParams dstCopyParams;
        dstCopyParams.blockLen = dim_ * sizeof(toType) / DATA_ALIGN_BYTES;
        dstCopyParams.srcStride = 0;
        dstCopyParams.dstStride = (headNum_ * dim_ - dim_) * sizeof(toType) / DATA_ALIGN_BYTES;

        if constexpr (needAtomic) {
            AscendC::SetAtomicNone();
        }

        PipeBarrier<PIPE_ALL>();

        while (remain > 0) {
            int64_t thisLen = vecOnceDataNum_;
            if (remain < thisLen) {
                thisLen = remain;
            }
            int64_t kThisOffset = fromOffset + (total - remain) / dim_ * MAX_BLOCK_DIM;

            srcCopyParams.blockCount = static_cast<uint16_t>(thisLen / dim_);
            DataCopy(inputLt_, fromGt[kThisOffset], srcCopyParams);
            DoVWhenMte2Finish(pipe_);

            DoVWhenMte3Finish(pipe_);
            if constexpr (std::is_same<toType, fromType>::value) {
                DataCopy(outputLt_, inputLt_, thisLen);
            } else {
                Cast(outputLt_, inputLt_, RoundMode::CAST_RINT, thisLen);
            }

            DoMte2WhenVFinish(pipe_);

            dstCopyParams.blockCount = static_cast<uint16_t>(thisLen / dim_);
            int64_t thisLineOffset = (total - remain) / dim_;
            int64_t outOffset = toOffset + thisLineOffset * headNum_ * dim_;

            DoMte3WhenVFinish(pipe_);
            if constexpr (needAtomic) {
                AscendC::SetAtomicAdd<toType>();
                AscendC::SetAtomicType<toType>();
                DataCopy(toGt[outOffset], outputLt_, dstCopyParams);
                AscendC::SetAtomicNone();
            } else {
                DataCopy(toGt[outOffset], outputLt_, dstCopyParams);
            }

            remain = remain - thisLen;
        }

        PipeBarrier<PIPE_ALL>();
    }

private:
    int64_t blockK_;
    int64_t dim_;
    int64_t headNum_;
    uint32_t vecOnceDataNum_;

    LocalTensor<fromType> inputLt_;
    LocalTensor<toType> outputLt_;
    TPipe* pipe_ = nullptr;
};

#endif // TRANS_H