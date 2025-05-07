/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */

#ifndef MXREC_INDEX_SELECT_FOR_RANK1_BACKWARD_KERNEL_H
#define MXREC_INDEX_SELECT_FOR_RANK1_BACKWARD_KERNEL_H
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr float ZERO = 0;
constexpr uint32_t UB_SIZE = 172 * 1024;
constexpr uint32_t UB_BUF = 1024;
constexpr uint32_t ALIGN_TARGET = 32;

class IndexSelectForRank1BackwardKernel {
public:
    __aicore__ inline IndexSelectForRank1BackwardKernel(const GM_ADDR tiling)
    {
        GET_TILING_DATA(tiling_data, tiling);
        xDim0 = tiling_data.xDim0;
        int64_t baseLen = tiling_data.baseLen;
        int64_t tailSplitIndex = tiling_data.tailSplitIndex;

        if (GetBlockIdx() >= tailSplitIndex) {
            indexWindowSize = baseLen;
            indexOffset = tailSplitIndex * (baseLen + 1) + (GetBlockIdx() - tailSplitIndex) * baseLen;
        } else {
            indexWindowSize = baseLen + 1;
            indexOffset = GetBlockIdx() * (baseLen + 1);
        }
        indexProcessWindowSize = UB_SIZE - xDim0 * sizeof(float) - UB_BUF;
        indexProcessWindowSize = indexProcessWindowSize / (sizeof(float) + sizeof(int64_t));
        indexProcessWindowSize = indexProcessWindowSize / ALIGN_TARGET * ALIGN_TARGET;
    }

    __aicore__ void Init(GM_ADDR gradY, GM_ADDR index, GM_ADDR gradX)
    {
        gradYGm.SetGlobalBuffer((__gm__ float*)gradY, indexProcessWindowSize);
        indexGm.SetGlobalBuffer((__gm__ int64_t*)index, indexProcessWindowSize);
        gradXGm.SetGlobalBuffer((__gm__ float*)gradX, xDim0);

        pipe.InitBuffer(inQueGradY, 1, Align32(indexProcessWindowSize * sizeof(float)));
        pipe.InitBuffer(inQueIndex, 1, Align32(indexProcessWindowSize * sizeof(int64_t)));
        pipe.InitBuffer(outQueGradX, 1, Align32(xDim0 * sizeof(float)));

        gradYUb = inQueGradY.AllocTensor<float>();
        indexUb = inQueIndex.AllocTensor<int64_t>();
        gradXUb = outQueGradX.AllocTensor<float>();

        Duplicate(gradXUb, ZERO, xDim0);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        gradYAddr = reinterpret_cast<__ubuf__ float*>(gradYUb.GetPhyAddr());
        indexAddr = reinterpret_cast<__ubuf__ int64_t*>(indexUb.GetPhyAddr());
        gradXAddr = reinterpret_cast<__ubuf__ float*>(gradXUb.GetPhyAddr());
    }

    __aicore__ void Process()
    {
        int64_t remain = indexWindowSize;
        for (int64_t offset = indexOffset; offset < indexOffset + indexWindowSize; offset += indexProcessWindowSize) {
            const int64_t process_len = remain > indexProcessWindowSize ? indexProcessWindowSize : remain;

            CopyIn(offset);
            Compute(process_len);

            remain -= process_len;
        }
        CopyOut();
    }

    __aicore__ void CopyIn(const int offset)
    {
        DataCopy(indexUb, indexGm[offset], indexProcessWindowSize);
        DataCopy(gradYUb, gradYGm[offset], indexProcessWindowSize);
    }

    __aicore__ void Compute(const int64_t process_len)
    {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        for (int i = 0; i < process_len; i++) {
            const auto index = indexAddr[i];
            const auto gradY = gradYAddr[i];
            gradXAddr[index] += gradY;
        }
    }

    __aicore__ void CopyOut()
    {
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

        SetAtomicAdd<float>();
        DataCopy(gradXGm, gradXUb, Align32(xDim0 * sizeof(float)) / sizeof(float));
        SetAtomicNone();

        inQueGradY.FreeTensor<float>(gradYUb);
        inQueIndex.FreeTensor<int64_t>(indexUb);
        outQueGradX.FreeTensor<float>(gradXUb);
    }

    template <typename T>
    __aicore__ T Align32(T n)
    {
        return (n + 31) / 32 * 32;
    }

private:
    LocalTensor<float> gradXUb;
    LocalTensor<float> gradYUb;
    LocalTensor<int64_t> indexUb;

    __ubuf__ float* gradXAddr;
    __ubuf__ float* gradYAddr;
    __ubuf__ int64_t* indexAddr;

    int64_t xDim0;

    int64_t indexWindowSize;
    int64_t indexProcessWindowSize;
    int64_t indexOffset;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueIndex;
    TQue<TPosition::VECIN, 1> inQueGradY;
    TQue<TPosition::VECOUT, 1> outQueGradX;

    GlobalTensor<float> gradXGm, gradYGm;
    GlobalTensor<int64_t> indexGm;
};

#endif  // MXREC_INDEX_SELECT_FOR_RANK1_BACKWARD_KERNEL_H
