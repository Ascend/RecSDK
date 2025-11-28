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

#ifndef MXREC_SPLIT_EMBEDDING_NOBAG_H
#define MXREC_SPLIT_EMBEDDING_NOBAG_H
#include "split_embedding_kernel_pattern.h"

namespace SplitEmbeddingCodegenForwardUnweighted {

class SplitEmbeddingNobagKernel : public SplitEmbeddingKernelPattern {
public:
    __aicore__ inline SplitEmbeddingNobagKernel(Args& args, TPipe* pipe) : SplitEmbeddingKernelPattern(args, pipe) {}

    __aicore__ inline void Compute()
    {
        indicesNumOneBlock = blockLen / alignMaxD;
        if (indicesNumOneBlock >= MAX_INDICES_ONE_BLOCK) {
            indicesNumOneBlock = MAX_INDICES_ONE_BLOCK;
        }

        int64_t lastIndices = 0;
        int64_t thisTableLen = 0;
        for (int64_t i = 1; i <= weightsOffsetsDim0; i++) {
            auto thisOffsetPerKey = offsetPerKeyGT.GetValue(i);
            if (thisOffsetPerKey == lastIndices) {
                continue;
            }

            Scheduler(thisOffsetPerKey - lastIndices, offsetOfThisCore, thisTableLen);
            if (thisTableLen > 0) {
                int64_t thisTableOffset = offsetOfThisCore + lastIndices;
                int64_t thisWeightOffset = weightOffsetGT.GetValue(i - 1);
                Process(thisTableLen, thisTableOffset, thisWeightOffset);
            }
            lastIndices = thisOffsetPerKey;
        }
    }

private:
    __aicore__ inline void Scheduler(const int64_t& totalLen, int64_t& offsetLen, int64_t& calcLen)
    {
        splitBaseLen = totalLen / GetBlockNum();
        tailSplitIndex = totalLen % GetBlockNum();
        if (GetBlockIdx() >= tailSplitIndex) {
            calcLen = splitBaseLen;
            offsetLen = tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
        } else {
            calcLen = splitBaseLen + 1;
            offsetLen = GetBlockIdx() * (splitBaseLen + 1);
        }
    }

    __aicore__ inline void Process(int64_t remain, int64_t startIndices, int64_t thisWeightOffset)
    {
        int64_t thisLen = remain;
        while (remain > 0) {
            if (thisLen > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }
            remain -= thisLen;
            CopyInNormal(startIndices, thisLen, maxD, thisWeightOffset);
            if (alignMaxD == maxD) {
                CopyOut<false>(thisLen, startIndices);
            } else {
                CopyOut<true>(thisLen, startIndices);
            }

            startIndices = startIndices + thisLen;
            thisLen = remain;
        }
    }

    template <bool isPad>
    __aicore__ inline void CopyOut(int64_t thisLen, int64_t startIndices)
    {
        LocalTensor<float> inputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();
        int64_t allLen;
        if constexpr (isPad) {
            allLen = thisLen * alignMaxD;
        } else {
            allLen = thisLen * maxD;
        }
        DataCopy(outLt, inputLt, allLen);

        queOut.EnQue(outLt);
        outLt = queOut.DeQue<float>();

        if constexpr (isPad) {
            for (int i = 0; i < thisLen; i++) {
                CpLocal2Gm(outGT[(startIndices + i) * maxD], outLt[i * alignMaxD], maxD);
            }
        } else {
            CpLocal2Gm(outGT[startIndices * maxD], outLt, allLen);
        }

        queIn.FreeTensor(inputLt);
        queOut.FreeTensor(outLt);
    }
};
} // namespace SplitEmbeddingCodegenForwardUnweighted
#endif  // MXREC_SPLIT_EMBEDDING_NOBAG_H
