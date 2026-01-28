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

#ifndef BACKWARD_CODEGEN_SGD_UNWEIGHTED_EXACT_KERNEL_FUN_H
#define BACKWARD_CODEGEN_SGD_UNWEIGHTED_EXACT_KERNEL_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenSgdUnweightedExact {

class BackwardCodegenSgdUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenSgdUnweightedExactKernel() {}

    __aicore__ inline void InitSgd(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);

        numOfOut = 1;  // 输出个数为1：grad
        indicesNumOneBlock = blockLen / numOfOut / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
    }

    __aicore__ inline void Tilling()
    {
        int64_t allLen = totalHashSize;
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

        for (int64_t i = weightsOffsetsDim0; i >= 0; i--) {
            if (thisTableOffset >= hashSizeCumsumGT.GetValue(i)) {
                tableIndex = i;
                break;
            }
        }
    }

    __aicore__ inline int64_t FillUpdateArgs(UpdateArgs* updateArgs, int64_t& remain)
    {
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;

        int64_t tableHashStart = hashSizeCumsumGT.GetValue(tableIndex);
        int64_t tableHashEnd = hashSizeCumsumGT.GetValue(tableIndex + 1);
        int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
        int64_t weightOffsetBase = *(weightsOffsetsPtr + tableIndex);

        int64_t cnt = 0;
        while (cnt < indicesNumOneBlock && remain > 0) {
            int64_t thisIndForTotalTable = thisTableOffset + thisTableLen - remain;
            remain = remain - 1;
            while (thisIndForTotalTable >= tableHashEnd) {
                tableIndex = tableIndex + 1;
                tableHashStart = tableHashEnd;
                tableHashEnd = hashSizeCumsumGT.GetValue(tableIndex + 1);
                embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                weightOffsetBase = *(weightsOffsetsPtr + tableIndex);
            }

            if (workspaceGT.GetValue(thisIndForTotalTable) != static_cast<uint32_t>(UpdateState::NEED_UPDATE)) {
                continue;
            }

            int64_t thisIndForThisTable = thisIndForTotalTable - tableHashStart;
            int64_t thisOutOffset = weightOffsetBase + thisIndForThisTable * embedDim;

            updateArgs[cnt].embedDim = embedDim;
            updateArgs[cnt].thisOutOffset = thisOutOffset;

            cnt += 1;
        }
        return cnt;
    }

    __aicore__ inline void DataCopyIn(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> inputLt = queIn.AllocTensor<float>();
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs gradArgs = updateArgs[i];
            DataCopy(inputLt[i * maxD * numOfOut], outGT[gradArgs.thisOutOffset], gradArgs.embedDim);
        }
        queIn.EnQue(inputLt);
    }
    
    __aicore__ inline void ComputeSgd(UpdateArgs* updateArgs, int64_t cnt)
    {
        float minusLearningRate = -learning_rate;

        LocalTensor<float> inputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();

        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs gradArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * numOfOut;

            // p[:] -= hyperparams['lr'] * p.grad
            Muls<float>(outLt[thisGradIndex], inputLt[thisGradIndex], minusLearningRate, gradArgs.embedDim);
        }

        queOut.EnQue(outLt);
        queIn.FreeTensor(inputLt);
    }

    __aicore__ inline void DataCopyOut(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> outLt = queOut.DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs gradArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * numOfOut;
            DataCopy(weightsDevOutGT[gradArgs.thisOutOffset], outLt[thisGradIndex], gradArgs.embedDim);
        }
        SetAtomicNone();
        queOut.FreeTensor(outLt);
    }

    __aicore__ inline void UpdateEmbedSgd(Args args)
    {
        this->UniqIndices();
        SyncAll();
        
        InitSgd(args);
        Tilling();

        UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
        int64_t remain = thisTableLen;
        while (remain > 0) {
            auto cnt = FillUpdateArgs(updateArgs, remain);
            DataCopyIn(updateArgs, cnt);
            ComputeSgd(updateArgs, cnt);
            DataCopyOut(updateArgs, cnt);
        }
    }

    __aicore__ inline void Compute(Args args)
    {
        Init(args);

        ClearGT(workspaceGT, totalHashSize);
        ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();

        ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
    }

private:

    int numOfOut;
    int indicesNumOneBlock;

    int64_t thisTableLen;
    int64_t thisTableOffset;
    int64_t tableIndex;
};
} // namespace BackwardCodegenSgdUnweightedExact
#endif
