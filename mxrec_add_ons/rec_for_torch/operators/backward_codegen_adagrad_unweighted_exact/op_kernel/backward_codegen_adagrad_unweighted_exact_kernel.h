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

#ifndef BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H
#define BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenAdagradUnweightedExact {
template <typename wType>
class BackwardCodegenAdagradUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel<wType> {
public:
    __aicore__ inline BackwardCodegenAdagradUnweightedExactKernel() {}

    __aicore__ inline void UpdateEmbedAda()
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)this->dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)this->weightsOffsets;
        __gm__ int64_t* offsetsPtr = (__gm__ int64_t*)this->offsets;

        int64_t allLen = this->totalHashSize;
        int64_t totalTableSizeSplit = allLen % GetBlockNum();
        int64_t aCoreTableLen = allLen / GetBlockNum();

        int64_t thisTableLen = 0;
        int64_t thisTableOffset = 0;

        if (GetBlockIdx() >= totalTableSizeSplit) {
            thisTableLen = aCoreTableLen;
            thisTableOffset =
                totalTableSizeSplit * (aCoreTableLen + 1) + (GetBlockIdx() - totalTableSizeSplit) * aCoreTableLen;
        } else {
            thisTableLen = aCoreTableLen + 1;
            thisTableOffset = GetBlockIdx() * (aCoreTableLen + 1);
        }

        int64_t tableIndex = 0;
        for (int64_t i = this->weightsOffsetsDim0; i >= 0; i--) {
            if (thisTableOffset >= this->hashSizeCumsumGT.GetValue(i)) {
                tableIndex = i;
                break;
            }
        }

        int64_t total = thisTableLen;
        int64_t remain = total;
        int numOfOut = 2;
        int indicesNumOneBlock = this->blockLen / numOfOut / this->maxD;
        int outIndex = 0 * this->maxD;
        int outIndex1 = 1 * this->maxD;
        UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        while (remain > 0) {
            int64_t thisLen = 0;
            while (thisLen < indicesNumOneBlock && remain > 0) {
                int64_t thisIndForTotalTable = thisTableOffset + total - remain;
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

                updateArgs[thisLen].embedDim = embedDim;
                updateArgs[thisLen].thisOutOffset = thisOutOffset;

                thisLen += 1;
            }

            LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                DataCopy(inputLt[i * this->maxD * numOfOut + outIndex], this->outGT[theArgs.thisOutOffset],
                         theArgs.embedDim);
                DataCopy(inputLt[i * this->maxD * numOfOut + outIndex1], this->momentum1DevGT[theArgs.thisOutOffset],
                         theArgs.embedDim);
            }
            this->queIn.template EnQue(inputLt);

            LocalTensor<float> newInputLt = this->queIn.template DeQue<float>();
            LocalTensor<float> outLt = this->queOut.template AllocTensor<float>();

            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * this->maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * this->maxD * numOfOut + outIndex1;
                Mul<float>(outLt[thisGradIndex], newInputLt[thisGradIndex], newInputLt[thisGradIndex],
                           theArgs.embedDim);
                Add<float>(outLt[thisGradIndex], newInputLt[thisMomentIndex], outLt[thisGradIndex],
                           theArgs.embedDim);

                Sqrt<float>(outLt[thisGradIndex], outLt[thisGradIndex], theArgs.embedDim);
                Adds<float>(outLt[thisGradIndex], outLt[thisGradIndex], this->eps, theArgs.embedDim);
                Duplicate<float>(outLt[thisMomentIndex], this->learning_rate, theArgs.embedDim);
                Div<float>(outLt[thisGradIndex], outLt[thisMomentIndex], outLt[thisGradIndex], theArgs.embedDim);

                Mul<float>(outLt[thisGradIndex], outLt[thisGradIndex], newInputLt[thisGradIndex], theArgs.embedDim);
                Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], -1, theArgs.embedDim);

                Mul<float>(outLt[thisMomentIndex], newInputLt[thisGradIndex], newInputLt[thisGradIndex],
                           theArgs.embedDim);
            }

            this->queOut.template EnQue(outLt);
            this->queIn.template FreeTensor(newInputLt);
            LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * this->maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * this->maxD * numOfOut + outIndex1;
                DataCopy(this->weightsDevOutGT[theArgs.thisOutOffset], newOutLt[thisGradIndex], theArgs.embedDim);
                DataCopy(this->momentum1DevOutGT[theArgs.thisOutOffset], newOutLt[thisMomentIndex], theArgs.embedDim);
            }
            SetAtomicNone();
            this->queOut.template FreeTensor(newOutLt);
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

        UpdateEmbedAda();
    }
};
}  // namespace BackwardCodegenAdagradUnweightedExact
#endif