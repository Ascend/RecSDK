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

#ifndef BACKWARD_CODEGEN_ADAM_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H
#define BACKWARD_CODEGEN_ADAM_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenAdamUnweightedExact {
template<typename wType>
class BackwardCodegenAdamUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel<wType> {
public:
    __aicore__ inline BackwardCodegenAdamUnweightedExactKernel() {}

    __aicore__ inline void InitAdam(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        momentum2Dev = args.momentum2Dev;
        momentum2DevOut = args.momentum2DevOut;
        momentum2DevGT.SetGlobalBuffer((__gm__ float*)momentum2Dev, this->outDim0);
        momentum2DevOutGT.SetGlobalBuffer((__gm__ float*)momentum2DevOut, this->outDim0);
        
        beta1 = tilingData.beta1;
        beta2 = tilingData.beta2;
        iter = tilingData.iter;
        beta1pow = tilingData.beta1pow;
        beta2pow = tilingData.beta2pow;
        beta2sqrt = tilingData.beta2sqrt;

        numOfOut = 3;  // 输出个数为3：grad, momentum1, momentum2
        indicesNumOneBlock = this->blockLen / numOfOut / this->maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        outIndex = 0 * this->maxD;  // grad偏移
        outIndex1 = 1 * this->maxD;  // momentum1偏移
        outIndex2 = 2 * this->maxD;  // momentum2偏移
    }

    __aicore__ inline void Tilling()
    {
        int64_t allLen = this->totalHashSize;
        int64_t totalTableSizeSplit = allLen % GetBlockNum();
        int64_t aCoreTableLen = allLen / GetBlockNum();

        if (GetBlockIdx() >= totalTableSizeSplit) {
            thisTableLen = aCoreTableLen;
            thisTableOffset =
                totalTableSizeSplit * (aCoreTableLen + 1) + (GetBlockIdx() - totalTableSizeSplit) * aCoreTableLen;
        } else {
            thisTableLen = aCoreTableLen + 1;
            thisTableOffset = GetBlockIdx() * (aCoreTableLen + 1);
        }

        for (int64_t i = this->weightsOffsetsDim0; i >= 0; i--) {
            if (thisTableOffset >= this->hashSizeCumsumGT.GetValue(i)) {
                tableIndex = i;
                break;
            }
        }
    }

    __aicore__ inline int64_t FillUpdateArgs(UpdateArgs* updateArgs, int64_t& remain)
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)this->dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)this->weightsOffsets;

        int64_t cnt = 0;
        while (cnt < indicesNumOneBlock && remain > 0) {
            int64_t thisIndForTotalTable = thisTableOffset + thisTableLen - remain;
            remain = remain - 1;
            if (thisIndForTotalTable >= this->hashSizeCumsumGT.GetValue(tableIndex + 1)) {
                tableIndex = tableIndex + 1;
            }

            if (this->workspaceGT.GetValue(thisIndForTotalTable) != NEED_UPDATE) {
                continue;
            }

            int64_t thisIndForThisTable = thisIndForTotalTable - this->hashSizeCumsumGT.GetValue(tableIndex);
            int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
            int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
            int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;

            updateArgs[cnt].embedDim = embedDim;
            updateArgs[cnt].thisOutOffset = thisOutOffset;

            cnt += 1;
        }
        return cnt;
    }

    __aicore__ inline void DataCopyIn(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            DataCopy(inputLt[i * this->maxD * numOfOut + outIndex], this->outGT[theArgs.thisOutOffset],
                     theArgs.embedDim);
            DataCopy(inputLt[i * this->maxD * numOfOut + outIndex1], this->momentum1DevGT[theArgs.thisOutOffset],
                     theArgs.embedDim);
            DataCopy(inputLt[i * this->maxD * numOfOut + outIndex2], momentum2DevGT[theArgs.thisOutOffset],
                     theArgs.embedDim);
        }
        this->queIn.template EnQue(inputLt);
    }

    __aicore__ inline void ComputeAdam(UpdateArgs* updateArgs, int64_t cnt)
    {
        float oneMinusBeta1 = (1 - beta1);
        float oneMinusBeta2 = (1 - beta2);
        float minusLearningRate = -this->learning_rate;
        float stepSize = minusLearningRate * beta2sqrt;

        LocalTensor<float> inputLt = this->queIn.template DeQue<float>();
        LocalTensor<float> outLt = this->queOut.template AllocTensor<float>();

        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisGradIndex = i * this->maxD * numOfOut + outIndex;
            int64_t thisMoment1Index = i * this->maxD * numOfOut + outIndex1;
            int64_t thisMoment2Index = i * this->maxD * numOfOut + outIndex2;

            // v[:] = beta1 * v + (1 - beta1) * p.grad
            Muls<float>(outLt[thisMoment1Index], inputLt[thisMoment1Index], beta1, theArgs.embedDim);
            Muls<float>(outLt[thisGradIndex], inputLt[thisGradIndex], oneMinusBeta1, theArgs.embedDim);
            Add<float>(outLt[thisMoment1Index], outLt[thisMoment1Index], outLt[thisGradIndex], theArgs.embedDim);

            // s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            Muls<float>(outLt[thisMoment2Index], inputLt[thisMoment2Index], beta2, theArgs.embedDim);
            Mul<float>(outLt[thisGradIndex], inputLt[thisGradIndex], inputLt[thisGradIndex], theArgs.embedDim);
            Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], oneMinusBeta2, theArgs.embedDim);
            Add<float>(outLt[thisMoment2Index], outLt[thisMoment2Index], outLt[thisGradIndex], theArgs.embedDim);

            // p[:] -= stepSize * v / (torch.sqrt(s) + eps)
            Sqrt<float>(inputLt[thisMoment2Index], outLt[thisMoment2Index], theArgs.embedDim);
            Adds<float>(inputLt[thisMoment2Index], inputLt[thisMoment2Index], this->eps, theArgs.embedDim);
            Div<float>(outLt[thisGradIndex], outLt[thisMoment1Index], inputLt[thisMoment2Index], theArgs.embedDim);
            Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], stepSize, theArgs.embedDim);
        }

        this->queOut.template EnQue(outLt);
        this->queIn.template FreeTensor(inputLt);
    }

    __aicore__ inline void DataCopyOut(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> outLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisGradIndex = i * this->maxD * numOfOut + outIndex;
            DataCopy(this->weightsDevOutGT[theArgs.thisOutOffset], outLt[thisGradIndex], theArgs.embedDim);
        }
        SetAtomicNone();
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisMoment1Index = i * this->maxD * numOfOut + outIndex1;
            int64_t thisMoment2Index = i * this->maxD * numOfOut + outIndex2;
            DataCopy(this->momentum1DevOutGT[theArgs.thisOutOffset], outLt[thisMoment1Index], theArgs.embedDim);
            DataCopy(momentum2DevOutGT[theArgs.thisOutOffset], outLt[thisMoment2Index], theArgs.embedDim);
        }
        this->queOut.template FreeTensor(outLt);
    }

    __aicore__ inline void UpdateEmbedAdam(Args args)
    {
        InitAdam(args);
        Tilling();

        UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
        int64_t remain = thisTableLen;
        while (remain > 0) {
            auto cnt = FillUpdateArgs(updateArgs, remain);
            DataCopyIn(updateArgs, cnt);
            ComputeAdam(updateArgs, cnt);
            DataCopyOut(updateArgs, cnt);
        }
    }

    __aicore__ inline void Compute(Args args)
    {
        this->Init(args);
        this->ClearGT(this->workspaceGT, this->totalHashSize);
        this->ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();

        this->ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
    }

private:
    GM_ADDR momentum2Dev;
    GM_ADDR momentum2DevOut;

    GlobalTensor<float> momentum2DevGT;
    GlobalTensor<float> momentum2DevOutGT;

    float beta1;
    float beta2;
    float stepSize;
    float beta2sqrt;
    int64_t iter;
    float beta1pow;
    float beta2pow;

    int numOfOut;
    int indicesNumOneBlock;
    int outIndex;
    int outIndex1;
    int outIndex2;

    int64_t thisTableLen;
    int64_t thisTableOffset;
    int64_t tableIndex;
};
}  // namespace BackwardCodegenAdamUnweightedExact
#endif