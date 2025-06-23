/**
 * @file backward_codegen_adam_unweighted_exact_kernel_unique.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

#ifndef BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H
#define BACKWARD_CODEGEN_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_UNIQUE_FUN_H

#include <cstdint>

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel_unique.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;
using namespace BackwardCodegenUnweightedExactUnique;

namespace BackwardCodegenUnweightedExactAdagradUnique {

template <typename wType>
class BackwardCodegenAdagradUnweightedExactKernelUnique : public BackwardCodegenUnweightedExactKernelUnique<wType> {
public:
    __aicore__ inline BackwardCodegenAdagradUnweightedExactKernelUnique() {}

    __aicore__ inline void AdagradScheduler()
    {
        int64_t lastIndices = 0;
        for (int64_t i = 1; i < this->uniqueHashDim0; i++) {
            if (this->uniqueHashSizeGT.GetValue(i) != lastIndices) { // 每张表上的indices尽量均分到每张卡上
                Scheduler(this->uniqueHashSizeGT.GetValue(i) - lastIndices, this->offsetOfThisCore, thisTableLen);
                if (thisTableLen > 0) {
                    tableIndex = i - 1;
                    thisTableOffset = this->offsetOfThisCore + lastIndices;
                    UpdateEmbedAdagrad();
                }
                lastIndices = this->uniqueHashSizeGT.GetValue(i);
            }
        }
    }

    __aicore__ inline void ComputeAdagrad(LocalTensor<float>inputLt, LocalTensor<float>outLt, int64_t totalLen)
    {
        int64_t momentum1Offset = totalLen;
        Mul<float>(outLt, inputLt, inputLt, momentum1Offset);
        Add<float>(outLt, inputLt[momentum1Offset], outLt, momentum1Offset);
        Sqrt<float>(outLt, outLt, momentum1Offset);
        Adds<float>(outLt, outLt, this->eps, momentum1Offset);
        Duplicate<float>(outLt[momentum1Offset], this->learning_rate, momentum1Offset);
        Div<float>(outLt, outLt[momentum1Offset], outLt, momentum1Offset);
        Mul<float>(outLt, outLt, inputLt, momentum1Offset);
        Muls<float>(outLt, outLt, -1, momentum1Offset);
        Mul<float>(outLt[momentum1Offset], inputLt, inputLt, momentum1Offset);
    }

    __aicore__ inline void CopyInNormal(int *updateArgs, int thisLen, int embedDim)
    {
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)this->weightsOffsets;
        LocalTensor<float> inputLt = this->queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int64_t thisIndForThisTable = this->uniqueIdGT.GetValue(thisTableOffset + i);
            int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
            updateArgs[i] = thisWeightOffset + thisIndForThisTable * embedDim;
            DataCopy(inputLt[i * this->maxD + thisMoment1Index], this->momentum1DevGT[updateArgs[i]], embedDim);
        }
        this->queIn.template EnQue(inputLt);
    }

    __aicore__ inline void CopyInDynamic(DynamicArgs *updateArgs, int64_t thisLen, int64_t embedDim)
    {
        LocalTensor<float> inputLt = this->queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            updateArgs[i].weightsAddr = this->weightsDevOutGT.GetValue(thisTableOffset + i);
            updateArgs[i].m1Addr = this->momentum1DevGT.GetValue(thisTableOffset + i);
            dynamicM1GT.SetGlobalBuffer((__gm__ float*)updateArgs[i].m1Addr, embedDim);
            DataCopy(inputLt[i * this->maxD + thisMoment1Index], dynamicM1GT, embedDim);
        }
        this->queIn.template EnQue(inputLt);
    }

    __aicore__ inline void CopyOutDynamic(DynamicArgs *updateArgs, int thisLen, int embedDim)
    {
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int32_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * this->maxD;
            dynamicWeightsGT.SetGlobalBuffer((__gm__ float*)updateArgs[i].weightsAddr, embedDim);
            dynamicM1GT.SetGlobalBuffer((__gm__ float*)updateArgs[i].m1Addr, embedDim);
            DataCopy(dynamicWeightsGT, newOutLt[thisGradIndex], embedDim);
            DataCopy(dynamicM1GT, newOutLt[thisMoment1Index + thisGradIndex], embedDim);
        }
        SetAtomicNone();
        this->queOut.template FreeTensor(newOutLt);
    }
    
    __aicore__ inline void CopyOutNormal(int *outOffset, int thisLen, int embedDim)
    {
        LocalTensor<float> newOutLt = this->queOut.template DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int thisGradIndex = i * this->maxD;
            DataCopy(this->weightsDevOutGT[outOffset[i]], newOutLt[thisGradIndex], embedDim);
            DataCopy(this->momentum1DevOutGT[outOffset[i]], newOutLt[thisMoment1Index + thisGradIndex], embedDim);
        }
        SetAtomicNone();
        this->queOut.template FreeTensor(newOutLt);
    }

    __aicore__ inline void UpdateEmbedAdagrad()
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
            remain -= thisLen;
            LocalTensor<float> inputLt = this->queIn.template AllocTensor<float>();
            LocalTensor<float> outputLt = this->queOut.template AllocTensor<float>();
            
            // copyIn
            CpGm2Local(inputLt, this->outGT[thisTableOffset * this->maxD], calcLen);
            this->queIn.template EnQue(inputLt);
            
            if constexpr(std::is_same<wType, float>::value) {
                // CopyIn
                int updateArgs[MAX_ARGS_PIPE_LEN];
                CopyInNormal(updateArgs, thisLen, embedDim);
                // compute
                inputLt = this->queIn.template DeQue<float>();
               
                ComputeAdagrad(inputLt, outputLt, calcLen);
                this->queOut.template EnQue(outputLt);

                // copyOut
                CopyOutNormal(updateArgs, thisLen, embedDim);
            } else {
                // CopyIn
                DynamicArgs updateArgs[MAX_ARGS_PIPE_LEN];
                CopyInDynamic(updateArgs, thisLen, embedDim);
                // compute
                inputLt = this->queIn.template DeQue<float>();
                ComputeAdagrad(inputLt, outputLt, calcLen);
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
        this->ClearGrad();
        pipe_barrier(PIPE_ALL);
        SyncAll();
        this->ComputeGrad();
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