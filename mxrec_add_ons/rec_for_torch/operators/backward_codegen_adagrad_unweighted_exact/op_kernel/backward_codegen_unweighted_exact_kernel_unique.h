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

template<typename wType>
class BackwardCodegenUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernel<wType> {
public:
    __aicore__ inline BackwardCodegenUnweightedExactKernelUnique() {}

    __aicore__ inline void InitUnique(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        uniqueId = args.uniqueId;
        uniqueInverse = args.uniqueInverse;
        uniqueHashSize = args.uniqueHashSize;
        
        uniqueHashDim0 = tilingData.uniqueHashDim0;
        
        uniqueHashSizeGT.SetGlobalBuffer((__gm__ int64_t*)uniqueHashSize, this->uniqueHashDim0);
        uniqueInverseGT.SetGlobalBuffer((__gm__ int64_t*)uniqueInverse, this->indicesDim0);

        offsetsGT.SetGlobalBuffer((__gm__ int64_t*)this->offsets, this->offsetsDim0);
        dOffsetsGT.SetGlobalBuffer((__gm__ int32_t*)this->dOffsets, this->dOffsetsDim0);

        // len(uniqueId) = uniqueHash[-1]
        uniqueIdDim0 = uniqueHashSizeGT.GetValue(uniqueHashDim0 - 1);
        uniqueIdGT.SetGlobalBuffer((__gm__ int64_t*)uniqueId, uniqueIdDim0);

        this->pipe.InitBuffer(queIndices, 1, MAX_ARGS_PIPE_LEN * sizeof(int64_t));
    }

    __aicore__ inline void ClearGrad()
    {
        int64_t total = 0;
        int64_t offsetLen = 0;

        Scheduler(uniqueIdDim0, offsetLen, total);

        int64_t loopLen = this->blockLen / this->maxD;
        int64_t loops = total / loopLen;
        int64_t tailLen = total % loopLen;
        LocalTensor<float> outLt = this->queOut.template AllocTensor<float>();
        Duplicate<float>(outLt, 0.0, this->blockLen);
        this->queOut.template EnQue(outLt);
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        for (int64_t i = 0; i < loops; i++) {
            int64_t outOffset = (offsetLen + i * loopLen) * this->maxD;
            CpLocal2Gm(this->outGT[outOffset], newOutLt, this->blockLen);
        }
        if (tailLen > 0) {
            int64_t outOffset = (offsetLen + loops * loopLen) * this->maxD;
            CpLocal2Gm(this->outGT[outOffset], newOutLt, tailLen * this->maxD);
        }
        this->queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void ComputeGradBag(ComputeUniqueArgs &args, float meanLen)
    {
        LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
        LocalTensor<float> outputLt = this->queOut.template AllocTensor<float>();
        LocalTensor<int64_t> indicesLt = queIndices.AllocTensor<int64_t>();

        CpGm2Local(indicesLt, uniqueInverseGT[args.startInd], args.thisLen);
        int64_t inverseOffset = uniqueHashSizeGT.GetValue(args.tableIndex);
        CpGm2Local(inputLt, this->gradOutputGT[args.inOffset], args.embedDim);

        queIndices.EnQue(indicesLt);
        this->queIn.template EnQue(inputLt);

        inputLt = this->queIn.template DeQue<float>();
        indicesLt = queIndices.DeQue<int64_t>();

        if (this->poolMode == MEAN_POOL) {
            Muls(outputLt, inputLt, meanLen, args.embedDim);
        } else {
            DataCopy(outputLt, inputLt, args.embedDim);
        }
    
        this->queOut.template EnQue(outputLt);
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < args.thisLen; i++) {
            int64_t outOffset = (indicesLt.GetValue(i) + inverseOffset) * this->maxD;
            CpLocal2Gm(this->outGT[outOffset], newOutLt, args.embedDim);
        }
        SetAtomicNone();
        this->queIn.template FreeTensor(inputLt);
        this->queOut.template FreeTensor(newOutLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void ComputeGradNoBag(ComputeUniqueArgs &args)
    {
        LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
        LocalTensor<float> outputLt = this->queOut.template AllocTensor<float>();
        LocalTensor<int64_t> indicesLt = queIndices.AllocTensor<int64_t>();

        CpGm2Local(indicesLt, uniqueInverseGT[args.startInd], args.thisLen);
        int64_t inverseOffset = uniqueHashSizeGT.GetValue(args.tableIndex) * this->maxD;
        CpGm2Local(inputLt, this->gradOutputGT[args.inOffset], this->maxD * args.thisLen);

        queIndices.EnQue(indicesLt);
        this->queIn.template EnQue(inputLt);
        inputLt = this->queIn.template DeQue<float>();
        indicesLt = queIndices.DeQue<int64_t>();

        DataCopy(outputLt, inputLt, this->maxD * args.thisLen);
        this->queOut.template EnQue(outputLt);
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < args.thisLen; i++) {
            int64_t outOffset = indicesLt.GetValue(i) * this->maxD  + inverseOffset;
            CpLocal2Gm(this->outGT[outOffset], newOutLt[i * this->maxD], args.embedDim);
        }
        SetAtomicNone();
        this->queIn.template FreeTensor(inputLt);
        this->queOut.template FreeTensor(newOutLt);
        queIndices.FreeTensor(indicesLt);
    }

    __aicore__ inline void ComputeGrad()
    {
        Scheduler(this->offsetsDim0 - 1, this->offsetOfThisCore, this->lenOfThisCore);
        if (this->lenOfThisCore == 0) {
            return;
        }
        int64_t indicesNumOneBlock = this->blockLen / this->maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        int64_t batchs = (this->offsetsDim0 - 1) / this->weightsOffsetsDim0;
        for (int64_t loop = 0; loop < this->lenOfThisCore; loop++) {
            int64_t i = (this->offsetOfThisCore + loop) / this->weightsOffsetsDim0;
            int64_t j = (this->offsetOfThisCore + loop) % this->weightsOffsetsDim0;
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
            int64_t embedDim = dOffsetsGT.GetValue(tableIndex + 1) - dOffsetsGT.GetValue(tableIndex);
            int64_t inputBatchInd = thisOffsetIndex % batchs;
            int64_t inputEmbedOffset = dOffsetsGT.GetValue(tableIndex);
            int64_t inputOffset;
            if (this->poolMode == NONE_POOL) {
                inputOffset = startIndices * this->gradOutputDim1;
            } else {
                inputOffset = inputBatchInd * this->gradOutputDim1 + inputEmbedOffset;
            }
            while (remain > 0) {
                if (thisLen > indicesNumOneBlock) {
                    thisLen = indicesNumOneBlock;
                }
                remain -= thisLen;
                ComputeUniqueArgs args{tableIndex, embedDim, inputOffset, thisLen, startIndices};
                if (this->poolMode == NONE_POOL) {
                    ComputeGradNoBag(args);
                    inputOffset += thisLen * this->gradOutputDim1;
                } else {
                    ComputeGradBag(args, meanLen);
                }
                startIndices += thisLen;
                thisLen = remain;
            }
        }
    }

    GM_ADDR uniqueId;
    GM_ADDR uniqueHashSize;
    GM_ADDR uniqueInverse;
    int64_t uniqueIdDim0;
    int64_t uniqueHashDim0;
 
    TQue<TPosition::VECIN, 1> queIndices;
 
    GlobalTensor<int64_t> uniqueIdGT;
    GlobalTensor<int64_t> uniqueHashSizeGT;
    GlobalTensor<int64_t> uniqueInverseGT;
    GlobalTensor<int64_t> indicesGT;
    GlobalTensor<int64_t> offsetsGT;
    GlobalTensor<int32_t> dOffsetsGT;
};
}  // namespace BackwardCodegenUnweightedExactUnique
#endif