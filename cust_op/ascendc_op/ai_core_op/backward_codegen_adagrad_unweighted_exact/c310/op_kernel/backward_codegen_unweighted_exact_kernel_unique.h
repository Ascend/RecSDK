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

#ifndef BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel.h"
using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;
namespace BackwardCodegenUnweightedExactUnique {

constexpr int M1_INDEX = 1;
constexpr int M2_INDEX = 2;

struct ComputeUniqueArgs {
    int64_t tableIndex;
    int64_t embedDim;
    int64_t inOffset;
    int64_t thisLen;
    int64_t startInd;
    int64_t weightsAddr;
    int64_t m1Addr;
    int64_t m2Addr;
};

struct DynamicArgs {
    int64_t weightsAddr;
    int64_t m1Addr;
    int64_t m2Addr;
};

__aicore__ inline void Scheduler(const int64_t &totalLen, int64_t &offsetLen, int64_t &calcLen)
{
    int64_t splitBaseLen = totalLen / GetBlockNum();
    int64_t tailSplitIndex = totalLen % GetBlockNum();
    if (GetBlockIdx() >= tailSplitIndex) {
        calcLen = splitBaseLen;
        offsetLen =
            tailSplitIndex * (splitBaseLen + 1) + (GetBlockIdx() - tailSplitIndex) * splitBaseLen;
    } else {
        calcLen = splitBaseLen + 1;
        offsetLen = GetBlockIdx() * (splitBaseLen + 1);
    }
}

class BackwardCodegenUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenUnweightedExactKernelUnique() {}

    __aicore__ inline void InitUnique(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        uniqueId = args.uniqueId;
        uniqueInverse = args.uniqueInverse;
        uniqueHashSize = args.uniqueHashSize;
        indiceSizeCumsum = args.indiceSizeCumsum;

        uniqueHashDim0 = tilingData.uniqueHashDim0;

        uniqueHashSizeGT.SetGlobalBuffer((__gm__ int64_t*)uniqueHashSize, uniqueHashDim0);
        uniqueInverseGT.SetGlobalBuffer((__gm__ int64_t*)uniqueInverse, indicesDim0);

        offsetsGT.SetGlobalBuffer((__gm__ int64_t*)offsets, offsetsDim0);
        dOffsetsGT.SetGlobalBuffer((__gm__ int32_t*)dOffsets, dOffsetsDim0);

        uniqueIdDim0 = uniqueHashSizeGT.GetValue(uniqueHashDim0 - 1);
        uniqueIdGT.SetGlobalBuffer((__gm__ int64_t*)uniqueId, uniqueIdDim0);

        indiceSizeCumsumGT.SetGlobalBuffer((__gm__ int64_t*)indiceSizeCumsum, weightsOffsetsDim0 + 1);

        pipe.InitBuffer(queIndices, 1, MAX_ARGS_PIPE_LEN * sizeof(int64_t));
    }

    __aicore__ inline void ClearGrad()
    {
        int64_t total = 0;
        int64_t offsetLen = 0;

        Scheduler(uniqueIdDim0, offsetLen, total);

        int64_t loopLen = blockLen / maxD;
        int64_t loops = total / loopLen;
        int64_t tailLen = total % loopLen;
        LocalTensor<float> outLt = queOut.template AllocTensor<float>();
        Duplicate<float>(outLt, 0.0, blockLen);
        queOut.template EnQue(outLt);
        LocalTensor<float> newOutLt = queOut.template DeQue<float>();
        for (int64_t i = 0; i < loops; i++) {
            int64_t outOffset = (offsetLen + i * loopLen) * maxD;
            CpLocal2Gm(outGT[outOffset], newOutLt, blockLen);
        }
        if (tailLen > 0) {
            int64_t outOffset = (offsetLen + loops * loopLen) * maxD;
            CpLocal2Gm(outGT[outOffset], newOutLt, tailLen * maxD);
        }
        queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void ComputeGradBag(ComputeUniqueArgs &args, float meanLen)
    {
        LocalTensor<float> inputLt = queIn.template AllocTensor<float>();
        LocalTensor<float> outputLt = queOut.template AllocTensor<float>();
        LocalTensor<int64_t> indicesLt = queIndices.AllocTensor<int64_t>();

        CpGm2Local(indicesLt, uniqueInverseGT[args.startInd], args.thisLen);
        int64_t inverseOffset = uniqueHashSizeGT.GetValue(args.tableIndex);
        CpGm2Local(inputLt, gradOutputGT[args.inOffset], args.embedDim);

        queIndices.EnQue(indicesLt);
        queIn.template EnQue(inputLt);

        inputLt = queIn.template DeQue<float>();
        indicesLt = queIndices.DeQue<int64_t>();

        if (poolMode == MEAN_POOL) {
            Muls(outputLt, inputLt, meanLen, args.embedDim);
        } else {
            DataCopy(outputLt, inputLt, args.embedDim);
        }

        queOut.template EnQue(outputLt);
        LocalTensor<float> newOutLt = queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < args.thisLen; i++) {
            int64_t outOffset = (indicesLt.GetValue(i) + inverseOffset) * maxD;
            CpLocal2Gm(outGT[outOffset], newOutLt, args.embedDim);
        }
        SetAtomicNone();
        queIn.template FreeTensor(inputLt);
        queOut.template FreeTensor(newOutLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void ComputeGradNoBag(ComputeUniqueArgs &args)
    {
        LocalTensor<float> inputLt = queIn.template AllocTensor<float>();
        LocalTensor<float> outputLt = queOut.template AllocTensor<float>();
        LocalTensor<int64_t> indicesLt = queIndices.AllocTensor<int64_t>();

        CpGm2Local(indicesLt, uniqueInverseGT[args.startInd], args.thisLen);
        int64_t inverseOffset = uniqueHashSizeGT.GetValue(args.tableIndex) * maxD;
        CpGm2Local(inputLt, gradOutputGT[args.inOffset], maxD * args.thisLen);

        queIndices.EnQue(indicesLt);
        queIn.template EnQue(inputLt);
        inputLt = queIn.template DeQue<float>();
        indicesLt = queIndices.DeQue<int64_t>();

        DataCopy(outputLt, inputLt, maxD * args.thisLen);
        queOut.template EnQue(outputLt);
        LocalTensor<float> newOutLt = queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < args.thisLen; i++) {
            int64_t outOffset = indicesLt.GetValue(i) * maxD  + inverseOffset;
            CpLocal2Gm(outGT[outOffset], newOutLt[i * maxD], args.embedDim);
        }
        SetAtomicNone();
        queIn.template FreeTensor(inputLt);
        queOut.template FreeTensor(newOutLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void ComputeGrad()
    {
        if (poolMode == NONE_POOL) {
            ComputeGradEC();
        } else {
            ComputeGradEBC();
        }
    }

    __aicore__ inline void ComputeGradEC()
    {
        int64_t indicesNumOneBlock = blockLen / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        int64_t lastIndices = 0;
        int64_t thisLen = 0;
        int64_t offsetOfThisTable = 0;
        int64_t cachedEmbedDim = 0;
        for (int64_t i = 1; i <= weightsOffsetsDim0; i++) {
            Scheduler(indiceSizeCumsumGT.GetValue(i) - lastIndices, offsetOfThisTable, thisLen);
            int64_t startIndices = offsetOfThisTable + lastIndices; // 上一张表的偏移+table_i的偏移
            lastIndices = indiceSizeCumsumGT.GetValue(i);
            if (thisLen <= 0) {
                continue;
            }
            int32_t remain = thisLen;
            int64_t thisOffsetIndex = startIndices;

            // datacopy In params
            int64_t tableIndex = i - 1;
            cachedEmbedDim = dOffsetsGT.GetValue(tableIndex + 1) - dOffsetsGT.GetValue(tableIndex);
            int64_t inputOffset = startIndices * gradOutputDim1;
            while (remain > 0) {
                if (thisLen > indicesNumOneBlock) {
                    thisLen = indicesNumOneBlock;
                }
                remain -= thisLen;
                ComputeUniqueArgs args{tableIndex, cachedEmbedDim, inputOffset, thisLen, startIndices};
                ComputeGradNoBag(args);
                inputOffset += thisLen * gradOutputDim1;
                startIndices += thisLen;
                thisLen = remain;
            }
        }
    }

    __aicore__ inline void ComputeGradEBC()
    {
        Scheduler(offsetsDim0 - 1, offsetOfThisCore, lenOfThisCore);
        if (lenOfThisCore == 0) {
            return;
        }
        int64_t indicesNumOneBlock = blockLen / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        int64_t batchs = (offsetsDim0 - 1) / weightsOffsetsDim0;
        int64_t cachedTableIndex = -1;
        int64_t cachedEmbedDim = 0;
        int64_t cachedInputEmbedOffset = 0;
        for (int64_t loop = 0; loop < lenOfThisCore; loop++) {
            int64_t i = (offsetOfThisCore + loop) / weightsOffsetsDim0;
            int64_t j = (offsetOfThisCore + loop) % weightsOffsetsDim0;
            int64_t thisOffsetIndex = j * batchs + i;
            int64_t startIndices = offsetsGT.GetValue(thisOffsetIndex);
            int64_t endIndices = offsetsGT.GetValue(thisOffsetIndex + 1);
            int32_t thisLen = endIndices - startIndices;

            if (thisLen <= 0) {
                continue;
            }

            int32_t remain = thisLen;
            float meanLen = 1 / static_cast<float>(thisLen);

            // dataCopy In params
            int64_t tableIndex = thisOffsetIndex / batchs;
            if (tableIndex != cachedTableIndex) {
                cachedTableIndex = tableIndex;
                cachedEmbedDim = dOffsetsGT.GetValue(tableIndex + 1) - dOffsetsGT.GetValue(tableIndex);
                cachedInputEmbedOffset = dOffsetsGT.GetValue(tableIndex);
            }
            int64_t embedDim = cachedEmbedDim;
            int64_t inputBatchInd = thisOffsetIndex % batchs;
            int64_t inputOffset = inputBatchInd * gradOutputDim1 + cachedInputEmbedOffset;
            while (remain > 0) {
                if (thisLen > indicesNumOneBlock) {
                    thisLen = indicesNumOneBlock;
                }
                remain -= thisLen;
                ComputeUniqueArgs args{tableIndex, embedDim, inputOffset, thisLen, startIndices};
                ComputeGradBag(args, meanLen);
                startIndices += thisLen;
                thisLen = remain;
            }
        }
    }

    GM_ADDR uniqueId;
    GM_ADDR uniqueHashSize;
    GM_ADDR uniqueInverse;
    GM_ADDR indiceSizeCumsum;
    int64_t uniqueIdDim0;
    int64_t uniqueHashDim0;

    TQue<TPosition::VECIN, 1> queIndices;

    GlobalTensor<int64_t> uniqueIdGT;
    GlobalTensor<int64_t> uniqueHashSizeGT;
    GlobalTensor<int64_t> uniqueInverseGT;
    GlobalTensor<int64_t> indiceSizeCumsumGT;
    GlobalTensor<int64_t> indicesGT;
    GlobalTensor<int64_t> offsetsGT;
    GlobalTensor<int32_t> dOffsetsGT;
};
}  // namespace BackwardCodegenUnweightedExactUnique
#endif
