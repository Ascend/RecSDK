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

template <typename wType>
class BackwardCodegenAdamUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique<wType> {
public:
    __aicore__ inline BackwardCodegenAdamUnweightedExactKernelUnique() {}
    __aicore__ inline void InitAdam(Args args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        momentum2Dev = args.momentum2Dev;
        momentum2DevOut = args.momentum2DevOut;
        momentum2DevGT.SetGlobalBuffer((__gm__ wType*)momentum2Dev, this->outDim0);
        momentum2DevOutGT.SetGlobalBuffer((__gm__ wType*)momentum2DevOut, this->outDim0);
        
        beta1 = tilingData.beta1;
        beta2 = tilingData.beta2;
        iter = tilingData.iter;
        beta1pow = tilingData.beta1pow;
        beta2pow = tilingData.beta2pow;
        beta2sqrt = tilingData.beta2sqrt;
        indicesNumOneBlock = this->blockLen / numOfOut / this->maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
    }
    
    __aicore__ inline void AdamScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < this->uniqueHashDim0; i++) {
            if (this->uniqueHashSizeGT.GetValue(i) != lastIndices) { // 每张表上的indices尽量均分到每张卡上
                Scheduler(this->uniqueHashSizeGT.GetValue(i) - lastIndices, this->offsetOfThisCore, thisTableLen);
                if (thisTableLen > 0) {
                    tableIndex = i - 1;
                    thisTableOffset = this->offsetOfThisCore + lastIndices;
                    UpdateEmbedAdam();
                }
                lastIndices = this->uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeAdam(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t totalLen)
    {
        int64_t thisGradIndex = 0;
        float oneMinusBeta1 = (1 - beta1);
        float oneMinusBeta2 = (1 - beta2);
        float minusLearningRate = -this->learning_rate;
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
        Adds<float>(inputLt[thisMoment2Index], inputLt[thisMoment2Index], this->eps, totalLen);
        Div<float>(outLt[thisGradIndex], outLt[thisMoment1Index], inputLt[thisMoment2Index], totalLen);
        Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], stepSize, totalLen);
    }

    __aicore__ inline void CopyInNormal(int64_t *updateArgs, int thisLen, int embedDim)
    {
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)this->weightsOffsets;
        LocalTensor<float> inputLt = this->queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int64_t thisIndForThisTable = this->uniqueIdGT.GetValue(thisTableOffset + i);
            int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
            updateArgs[i] = thisWeightOffset + thisIndForThisTable * embedDim;
            DataCopy(inputLt[i * this->maxD + thisMoment1Index], this->momentum1DevGT[updateArgs[i]], embedDim);
            DataCopy(inputLt[i * this->maxD + thisMoment2Index], momentum2DevGT[updateArgs[i]], embedDim);
        }
        this->queIn.template EnQue(inputLt);
    }

    __aicore__ inline void CopyInDynamic(DynamicArgs *updateArgs, int64_t thisLen, int64_t embedDim)
    {
        LocalTensor<float> inputLt = this->queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            updateArgs[i].weightsAddr = this->weightsDevOutGT.GetValue(thisTableOffset + i);
            updateArgs[i].m1Addr = this->momentum1DevGT.GetValue(thisTableOffset + i);
            updateArgs[i].m2Addr = momentum2DevGT.GetValue(thisTableOffset + i);
            dynamicM1GT.SetGlobalBuffer((__gm__ float*)updateArgs[i].m1Addr, embedDim);
            dynamicM2GT.SetGlobalBuffer((__gm__ float*)updateArgs[i].m2Addr, embedDim);
            DataCopy(inputLt[i * this->maxD + thisMoment1Index], dynamicM1GT, embedDim);
            DataCopy(inputLt[i * this->maxD + thisMoment2Index], dynamicM2GT, embedDim);
        }
        this->queIn.template EnQue(inputLt);
    }

    __aicore__ inline void CopyOutDynamic(DynamicArgs*updateArgs, int thisLen, int embedDim)
    {
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int32_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * this->maxD;
            dynamicWeightsGT.SetGlobalBuffer((__gm__ float*)updateArgs[i].weightsAddr, embedDim);
            DataCopy(dynamicWeightsGT, newOutLt[thisGradIndex], embedDim);
        }
        SetAtomicNone();
        for (int32_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * this->maxD;
            dynamicM1GT.SetGlobalBuffer((__gm__ float*)updateArgs[i].m1Addr, embedDim);
            dynamicM2GT.SetGlobalBuffer((__gm__ float*)updateArgs[i].m2Addr, embedDim);
            DataCopy(dynamicM1GT, newOutLt[thisMoment1Index + thisGradIndex], embedDim);
            DataCopy(dynamicM2GT, newOutLt[thisMoment2Index + thisGradIndex], embedDim);
        }
        
        this->queOut.template FreeTensor(newOutLt);
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
        for (int64_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * this->maxD;
            DataCopy(this->momentum1DevOutGT[outOffset[i]], newOutLt[thisMoment1Index + thisGradIndex], embedDim);
            DataCopy(momentum2DevOutGT[outOffset[i]], newOutLt[thisMoment2Index + thisGradIndex], embedDim);
        }
        this->queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void UpdateEmbedAdam()
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
            thisMoment1Index = calcLen * M1_INDEX;
            thisMoment2Index = calcLen * M2_INDEX;
            remain -= thisLen;
            LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
            LocalTensor<float> outputLt = this->queOut.template AllocTensor<float>();
            
            // copyIn
            CpGm2Local(inputLt, this->outGT[thisTableOffset * this->maxD], calcLen);
            this->queIn.template EnQue(inputLt);
            
            if constexpr(std::is_same<wType, float>::value) {
                // CopyIn
                int64_t updateArgs[MAX_ARGS_PIPE_LEN];
                CopyInNormal(updateArgs, thisLen, embedDim);
                // compute
                inputLt = this->queIn.template DeQue<float>();
               
                ComputeAdam(inputLt, outputLt, calcLen);
                this->queOut.template EnQue(outputLt);

                // copyOut
                CopyOutNormal(updateArgs, thisLen, embedDim);
            } else {
                // CopyIn
                DynamicArgs   updateArgs[MAX_ARGS_PIPE_LEN];
                CopyInDynamic(updateArgs, thisLen, embedDim);
                // compute
                inputLt = this->queIn.template DeQue<float>();
                ComputeAdam(inputLt, outputLt, calcLen);
                this->queOut.template EnQue(outputLt);
                // copyOut
                CopyOutDynamic(updateArgs, thisLen, embedDim);
            }
            this->queIn.template FreeTensor(inputLt);
            thisTableOffset += thisLen;
            thisLen = remain;
        }
    }
    __aicore__ inline void Compute(Args args)
    {
        this->Init(args);
        this->InitUnique(args);
        InitAdam(args);
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

    GlobalTensor<wType> momentum2DevGT;
    GlobalTensor<wType> momentum2DevOutGT;
    GlobalTensor<float> dynamicWeightsGT;
    GlobalTensor<float> dynamicM1GT;
    GlobalTensor<float> dynamicM2GT;

    float beta1;
    float beta2;
    float beta1pow;
    float beta2pow;
    float stepSize;
    float beta2sqrt;
    int64_t iter;
    int numOfOut = 3;
    int indicesNumOneBlock;

    int64_t thisMoment1Index;
    int64_t thisMoment2Index;
    int64_t thisTableLen;
    int64_t thisTableOffset;
    int64_t tableIndex;
};
}  // namespace BackwardCodegenAdagradUnweightedExactUnique
#endif