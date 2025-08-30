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

#ifndef BACKWARD_CODEGEN_ADAM_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_ADAM_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel_unique.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;
using namespace BackwardCodegenUnweightedExactUnique;
namespace BackwardCodegenUnweightedAdamExactUnique {
constexpr int NUM_OUTPUTS = 3; // grad, momentum1, momentum2
class BackwardCodegenAdamUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique {
public:
    __aicore__ inline BackwardCodegenAdamUnweightedExactKernelUnique() {}
    __aicore__ inline void InitAdam(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        momentum2Dev = args.momentum2Dev;
        momentum2DevOut = args.momentum2DevOut;
        momentum2DevGT.SetGlobalBuffer((__gm__ float*)momentum2Dev, outDim0);
        momentum2DevOutGT.SetGlobalBuffer((__gm__ float*)momentum2DevOut, outDim0);
        
        beta1 = tilingData.beta1;
        beta2 = tilingData.beta2;
        iter = tilingData.iter;
        beta1pow = tilingData.beta1pow;
        beta2pow = tilingData.beta2pow;
        beta2sqrt = tilingData.beta2sqrt;
        numOfOut = NUM_OUTPUTS;

        indicesNumOneBlock = blockLen / numOfOut / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
    }
    
    __aicore__ inline void AdamScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < uniqueHashDim0; i++) {
            if (uniqueHashSizeGT.GetValue(i) != lastIndices) { // 每张表上的indices尽量均分到每张卡上
                Scheduler(uniqueHashSizeGT.GetValue(i) - lastIndices, offsetOfThisCore, thisTableLen);
                if (thisTableLen > 0) {
                    tableIndex = i - 1;
                    thisTableOffset = offsetOfThisCore + lastIndices;
                    UpdateEmbedAdam();
                }
                lastIndices = uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeAdam(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t totalLen)
    {
        int64_t thisGradIndex = 0;
        float oneMinusBeta1 = (1 - beta1);
        float oneMinusBeta2 = (1 - beta2);
        float minusLearningRate = -learning_rate;
        thisMoment1Index = totalLen * M1_INDEX;
        thisMoment2Index = totalLen * M2_INDEX;
        stepSize = minusLearningRate * beta2sqrt;

        // v[:] = beta1 * v + (1 - beta1) * p.grad
        Muls<float>(outLt[thisMoment1Index], inputLt[thisMoment1Index], beta1, totalLen);
        Muls<float>(outLt[thisGradIndex], inputLt[thisGradIndex], oneMinusBeta1, totalLen);
        Add<float>(outLt[thisMoment1Index], outLt[thisMoment1Index], outLt[thisGradIndex], totalLen);

        // s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
        Muls<float>(outLt[thisMoment2Index], inputLt[thisMoment2Index], beta2, totalLen);
        Mul<float>(outLt[thisGradIndex], inputLt[thisGradIndex], inputLt[thisGradIndex], totalLen);
        Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], oneMinusBeta2, totalLen);
        Add<float>(outLt[thisMoment2Index], outLt[thisMoment2Index], outLt[thisGradIndex], totalLen);

        // p[:] -= stepSize * v / (torch.sqrt(s) + eps)
        Sqrt<float>(inputLt[thisMoment2Index], outLt[thisMoment2Index], totalLen);
        Adds<float>(inputLt[thisMoment2Index], inputLt[thisMoment2Index], eps, totalLen);
        Div<float>(outLt[thisGradIndex], outLt[thisMoment1Index], inputLt[thisMoment2Index], totalLen);
        Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], stepSize, totalLen);
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
            DataCopy(inputLt[i * maxD + thisMoment2Index], momentum2DevGT[updateArgs[i]], embedDim);
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
        }
        SetAtomicNone();
        for (int64_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * maxD;
            DataCopy(momentum1DevOutGT[outOffset[i]], newOutLt[thisMoment1Index + thisGradIndex], embedDim);
            DataCopy(momentum2DevOutGT[outOffset[i]], newOutLt[thisMoment2Index + thisGradIndex], embedDim);
        }
        queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void UpdateEmbedAdam()
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
            thisMoment2Index = calcLen * M2_INDEX;
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
            
            ComputeAdam(inputLt, outputLt, calcLen);
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
        InitAdam(args);
        ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        ComputeGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        AdamScheduler();
    }
private:
    
    GM_ADDR momentum2Dev;
    GM_ADDR momentum2DevOut;

    GlobalTensor<float> momentum2DevGT;
    GlobalTensor<float> momentum2DevOutGT;
    GlobalTensor<float> dynamicWeightsGT;
    GlobalTensor<float> dynamicM1GT;
    GlobalTensor<float> dynamicM2GT;

    float beta1;
    float beta2;
    float beta1pow;
    float beta2pow;
    float beta2sqrt;
    float stepSize;
    int64_t iter;
    int numOfOut;
    int indicesNumOneBlock;

    int64_t thisMoment1Index;
    int64_t thisMoment2Index;
    int64_t thisTableLen;
    int64_t thisTableOffset;
    int64_t tableIndex;
};
}  // namespace BackwardCodegenAdagradUnweightedExactUnique
#endif