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

#ifndef BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel_unique.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;
using namespace BackwardCodegenUnweightedExactUnique;

namespace BackwardCodegenUnweightedExactAdagradUnique {

class BackwardCodegenAdagradUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique {
public:
    __aicore__ inline BackwardCodegenAdagradUnweightedExactKernelUnique() {}

    __aicore__ inline void AdagradScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < uniqueHashDim0; i++) {
            if (uniqueHashSizeGT.GetValue(i) != lastIndices) { // 每张表上的indices尽量均分到每张卡上
                Scheduler(uniqueHashSizeGT.GetValue(i) - lastIndices, offsetOfThisCore, thisTableLen);
                if (thisTableLen > 0) {
                    tableIndex = i - 1;
                    thisTableOffset = offsetOfThisCore + lastIndices;
                    UpdateEmbedAdagrad();
                }
                lastIndices = uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeAdagrad(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t totalLen)
    {
        int64_t momentum1Offset = totalLen;
        Mul<float>(outLt, inputLt, inputLt, momentum1Offset);
        Add<float>(outLt, inputLt[momentum1Offset], outLt, momentum1Offset);
        Sqrt<float>(outLt, outLt, momentum1Offset);
        Adds<float>(outLt, outLt, eps, momentum1Offset);
        Duplicate<float>(outLt[momentum1Offset], learning_rate, momentum1Offset);
        Div<float>(outLt, outLt[momentum1Offset], outLt, momentum1Offset);
        Mul<float>(outLt, outLt, inputLt, momentum1Offset);
        Muls<float>(outLt, outLt, -1, momentum1Offset);
        Mul<float>(outLt[momentum1Offset], inputLt, inputLt, momentum1Offset);
    }

    __aicore__ inline void CopyInNormal(int64_t *updateArgs, int thisLen, int embedDim)
    {
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;
        LocalTensor<float> inputLt = queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int64_t thisIndForThisTable = uniqueIdGT.GetValue(thisTableOffset + i);
            int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
            updateArgs[i] = thisWeightOffset + thisIndForThisTable * embedDim;
            DataCopy(inputLt[i * maxD + thisMoment1Index], momentum1DevGT[updateArgs[i]], embedDim);
        }
        queIn.template EnQue(inputLt);
    }

    __aicore__ inline void CopyOutNormal(int64_t *outOffset, int thisLen, int embedDim)
    {
        LocalTensor<float> newOutLt = queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * maxD;
            DataCopy(weightsDevOutGT[outOffset[i]], newOutLt[thisGradIndex], embedDim);
            DataCopy(momentum1DevOutGT[outOffset[i]], newOutLt[thisMoment1Index + thisGradIndex], embedDim);
        }
        SetAtomicNone();
        queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void UpdateEmbedAdagrad()
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;

        indicesNumOneBlock = blockLen / numOfOut / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        int64_t thisLen = thisTableLen;
        int64_t remain = thisTableLen;
        int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);

        while (remain > 0) {
            if (remain > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }

            int calcLen = thisLen * maxD;
            thisMoment1Index = calcLen * M1_INDEX;
            remain -= thisLen;
            LocalTensor<float> inputLt = queIn.template AllocTensor<float>();
            LocalTensor<float> outputLt = queOut.template AllocTensor<float>();
            
            // copyIn
            CpGm2Local(inputLt, outGT[thisTableOffset * maxD], calcLen);
            queIn.template EnQue(inputLt);
            // CopyIn
            int64_t updateArgs[MAX_ARGS_PIPE_LEN];
            CopyInNormal(updateArgs, thisLen, embedDim);
            // compute
            inputLt = queIn.template DeQue<float>();
            
            ComputeAdagrad(inputLt, outputLt, calcLen);
            queOut.template EnQue(outputLt);

            // copyOut
            CopyOutNormal(updateArgs, thisLen, embedDim);
            queIn.template FreeTensor(inputLt);
            thisTableOffset += thisLen;
            thisLen = remain;
        }
    }
    __aicore__ inline void Compute(Args args)
    {
        Init(args);
        InitUnique(args);
        ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        AdagradScheduler();
    }
private:
    
    GM_ADDR momentum2Dev;
    GlobalTensor<float> dynamicWeightsGT;
    GlobalTensor<float> dynamicM1GT;

    int numOfOut = 3;
    int indicesNumOneBlock;

    int64_t thisMoment1Index;
    int64_t thisTableLen;
    int64_t thisTableOffset;
    int64_t tableIndex;
};
}  // namespace BackwardCodegenAdagradUnweightedExactUnique
#endif