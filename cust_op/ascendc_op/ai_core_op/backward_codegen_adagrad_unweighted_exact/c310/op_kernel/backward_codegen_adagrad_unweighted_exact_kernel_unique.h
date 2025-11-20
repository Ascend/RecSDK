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

template <typename T>
__aicore__ inline void AdagradCompute(__local_mem__ T* dstGrad, __local_mem__ T* dstMoment,
                                      __local_mem__ T* srcGrad, __local_mem__ T* srcMoment,
                                      uint32_t calCount, uint16_t repeatCount, uint32_t oneRepeat,
                                      float eps, float learning_rate)
{
    AscendC::MicroAPI::RegTensor<T> dstVregG;
    AscendC::MicroAPI::RegTensor<T> dstVregM;
    AscendC::MicroAPI::RegTensor<T> srcVregG;
    AscendC::MicroAPI::RegTensor<T> srcVregM;
    AscendC::MicroAPI::MaskReg mask;

    for (uint16_t i = 0; i < repeatCount; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<uint32_t>(calCount);
        AscendC::MicroAPI::DataCopy(srcVregG, srcGrad + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregM, srcMoment + i * oneRepeat);

        AscendC::MicroAPI::Mul(dstVregG, srcVregG, srcVregG, mask);
        AscendC::MicroAPI::Add(dstVregG, srcVregM, dstVregG, mask);

        AscendC::MicroAPI::Sqrt(dstVregG, dstVregG, mask);
        AscendC::MicroAPI::Adds(dstVregG, dstVregG, eps, mask);
        AscendC::MicroAPI::Duplicate(dstVregM, learning_rate, mask);
        AscendC::MicroAPI::Div(dstVregG, dstVregM, dstVregG, mask);

        AscendC::MicroAPI::Mul(dstVregG, dstVregG, srcVregG, mask);
        AscendC::MicroAPI::Muls(dstVregG, dstVregG, -1, mask);
        AscendC::MicroAPI::Mul(dstVregM, srcVregG, srcVregG, mask);

        AscendC::MicroAPI::DataCopy(dstGrad + i * oneRepeat, dstVregG, mask);
        AscendC::MicroAPI::DataCopy(dstMoment + i * oneRepeat, dstVregM, mask);
    }
}

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
        if (useRegBase) {
            __local_mem__ float* dstGrad = (__local_mem__ float*)outLt.GetPhyAddr();
            __local_mem__ float* dstMoment = (__local_mem__ float*)outLt[totalLen].GetPhyAddr();
            __local_mem__ float* srcGrad = (__local_mem__ float*)inputLt.GetPhyAddr();
            __local_mem__ float* srcMoment = (__local_mem__ float*)inputLt[totalLen].GetPhyAddr();

            constexpr uint32_t vecLen = AscendC::GetVecLen();
            constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
            uint16_t repeatCount = (totalLen + oneRepeat - 1) / oneRepeat;

            VF_CALL<AdagradCompute<float>>(dstGrad, dstMoment, srcGrad, srcMoment, totalLen,
                                           repeatCount, oneRepeat, eps, learning_rate);
        } else {
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
    }

    __aicore__ inline void CopyInNormal(int64_t *updateArgs, int thisLen, int embedDim, int64_t weightOffsetBase)
    {
        LocalTensor<float> inputLt = queIn.template DeQue<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            int64_t thisIndForThisTable = uniqueIdGT.GetValue(thisTableOffset + i);
            updateArgs[i] = weightOffsetBase + thisIndForThisTable * embedDim;
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
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;

        indicesNumOneBlock = blockLen / numOfOut / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        if (weightsOffsetsDim0 <= 2) {
            int64_t remain = thisTableLen;
            while (remain > 0) {
                int64_t thisLen = remain;
                if (thisLen > indicesNumOneBlock) {
                    thisLen = indicesNumOneBlock;
                }

                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t weightOffsetBase = *(weightsOffsetsPtr + tableIndex);
                int calcLen = thisLen * maxD;
                thisMoment1Index = calcLen * M1_INDEX;
                remain -= thisLen;

                LocalTensor<float> inputLt = queIn.template AllocTensor<float>();
                LocalTensor<float> outputLt = queOut.template AllocTensor<float>();

                CpGm2Local(inputLt, outGT[thisTableOffset * maxD], calcLen);
                queIn.template EnQue(inputLt);

                int64_t updateArgs[MAX_ARGS_PIPE_LEN];
                CopyInNormal(updateArgs, thisLen, embedDim, weightOffsetBase);

                inputLt = queIn.template DeQue<float>();
                ComputeAdagrad(inputLt, outputLt, calcLen);
                queOut.template EnQue(outputLt);

                CopyOutNormal(updateArgs, thisLen, embedDim);
                queIn.template FreeTensor(inputLt);
                thisTableOffset += thisLen;

                if ((tableIndex + 1) <= weightsOffsetsDim0 &&
                    thisTableOffset >= hashSizeCumsumGT.GetValue(tableIndex + 1)) {
                    tableIndex = tableIndex + 1;
                }
            }
            return;
        }

        int64_t tableHashStart = hashSizeCumsumGT.GetValue(tableIndex);
        int64_t tableHashEnd = hashSizeCumsumGT.GetValue(tableIndex + 1);
        int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
        int64_t weightOffsetBase = *(weightsOffsetsPtr + tableIndex);
        int64_t remain = thisTableLen;
        int64_t thisLen = thisTableLen;

        while (remain > 0) {
            if (remain > indicesNumOneBlock) {
                thisLen = indicesNumOneBlock;
            }

            int calcLen = thisLen * maxD;
            thisMoment1Index = calcLen * M1_INDEX;
            remain -= thisLen;
            LocalTensor<float> inputLt = queIn.template AllocTensor<float>();
            LocalTensor<float> outputLt = queOut.template AllocTensor<float>();

            CpGm2Local(inputLt, outGT[thisTableOffset * maxD], calcLen);
            queIn.template EnQue(inputLt);

            int64_t updateArgs[MAX_ARGS_PIPE_LEN];
            while (thisTableOffset >= tableHashEnd) {
                tableIndex = tableIndex + 1;
                tableHashStart = tableHashEnd;
                tableHashEnd = hashSizeCumsumGT.GetValue(tableIndex + 1);
                embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                weightOffsetBase = *(weightsOffsetsPtr + tableIndex);
            }
            CopyInNormal(updateArgs, thisLen, embedDim, weightOffsetBase);

            inputLt = queIn.template DeQue<float>();
            ComputeAdagrad(inputLt, outputLt, calcLen);
            queOut.template EnQue(outputLt);

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
