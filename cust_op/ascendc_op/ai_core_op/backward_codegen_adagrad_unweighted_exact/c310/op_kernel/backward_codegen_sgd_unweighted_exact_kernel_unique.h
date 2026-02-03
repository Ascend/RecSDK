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

#ifndef BACKWARD_CODEGEN_SGD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_SGD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel_unique.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;
using namespace BackwardCodegenUnweightedExactUnique;

namespace BackwardCodegenUnweightedSgdExactUnique {

class BackwardCodegenSgdUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique {
public:
    __aicore__ inline BackwardCodegenSgdUnweightedExactKernelUnique() {}
    
    __aicore__ inline void SgdScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < this->uniqueHashDim0; i++) {
            if (this->uniqueHashSizeGT.GetValue(i) != lastIndices) { // 每张表上的indices尽量均分到每张卡上
                Scheduler(this->uniqueHashSizeGT.GetValue(i) - lastIndices, this->offsetOfThisCore, thisTableLen);
                if (thisTableLen > 0) {
                    tableIndex = i - 1;
                    thisTableOffset = this->offsetOfThisCore + lastIndices;
                    UpdateEmbedSgd();
                }
                lastIndices = this->uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeSgd(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t totalLen)
    {
        int64_t thisGradIndex = 0;

        float minusLearningRate = -this->learning_rate;

        // p[:] -= learning_rate * g[:]
        Muls<float>(outLt[thisGradIndex], inputLt[thisGradIndex], minusLearningRate, totalLen);
    }

    __aicore__ inline void CopyInNormal(int64_t *updateArgs, int thisLen, int embedDim)
    {
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)this->weightsOffsets;
        LocalTensor<float> inputLt = this->queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int64_t thisIndForThisTable = this->uniqueIdGT.GetValue(thisTableOffset + i);
            int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
            updateArgs[i] = thisWeightOffset + thisIndForThisTable * embedDim;
        }
        this->queIn.template EnQue(inputLt);
    }
    
    __aicore__ inline void CopyOutNormal(int64_t *outOffset, int thisLen, int embedDim)
    {
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * this->maxD;
            DataCopy(this->weightsDevOutGT[outOffset[i]], newOutLt[thisGradIndex], embedDim);
        }
        SetAtomicNone();
        this->queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void UpdateEmbedSgd()
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)this->dOffsets;

        indicesNumOneBlock = this->blockLen / numOfOut / this->maxD;
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

            int calcLen = thisLen * this->maxD;
            remain -= thisLen;
            LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
            LocalTensor<float> outputLt = this->queOut.template AllocTensor<float>();
            
            // copyIn
            CpGm2Local(inputLt, this->outGT[thisTableOffset * this->maxD], calcLen);
            this->queIn.template EnQue(inputLt);
            
            // CopyIn
            int64_t updateArgs[MAX_ARGS_PIPE_LEN];
            CopyInNormal(updateArgs, thisLen, embedDim);
            // compute
            inputLt = this->queIn.template DeQue<float>();
            
            ComputeSgd(inputLt, outputLt, calcLen);
            this->queOut.template EnQue(outputLt);

            // copyOut
            CopyOutNormal(updateArgs, thisLen, embedDim);

            this->queIn.template FreeTensor(inputLt);
            thisTableOffset += thisLen;
            thisLen = remain;
        }
    }
    __aicore__ inline void Compute(Args args)
    {
        this->Init(args);
        this->InitUnique(args);
        this->ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        this->ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
    }
private:
    int numOfOut = 3;
    int indicesNumOneBlock;

    int64_t thisTableLen;
    int64_t thisTableOffset;
    int64_t tableIndex;
};
}  // namespace BackwardCodegenUnweightedSgdExactUnique
#endif