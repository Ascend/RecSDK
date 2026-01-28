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

class BackwardCodegenAdagradUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenAdagradUnweightedExactKernel() {}

    __aicore__ inline void UpdateEmbedAda()
    {
        this->UniqIndices();
        SyncAll();
        
        __gm__ int32_t* dOffsetsPtr = (__gm__ int32_t*)dOffsets;
        __gm__ int64_t* weightsOffsetsPtr = (__gm__ int64_t*)weightsOffsets;
        __gm__ int64_t* offsetsPtr = (__gm__ int64_t*)offsets;

        int64_t allLen = totalHashSize;
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
        for (int64_t i = weightsOffsetsDim0; i >= 0; i--) {
            if (thisTableOffset >= hashSizeCumsumGT.GetValue(i)) {
                tableIndex = i;
                break;
            }
        }

        int64_t total = thisTableLen;
        int64_t remain = total;
        int numOfOut = 2;
        int indicesNumOneBlock = blockLen / numOfOut / maxD;
        int outIndex = 0 * maxD;
        int outIndex1 = 1 * maxD;
        UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        int64_t tableHashStart = hashSizeCumsumGT.GetValue(tableIndex);
        int64_t tableHashEnd = hashSizeCumsumGT.GetValue(tableIndex + 1);
        int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
        int64_t weightOffsetBase = *(weightsOffsetsPtr + tableIndex);

        while (remain > 0) {
            int64_t thisLen = 0;
            while (thisLen < indicesNumOneBlock && remain > 0) {
                int64_t thisIndForTotalTable = thisTableOffset + total - remain;
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

                updateArgs[thisLen].embedDim = embedDim;
                updateArgs[thisLen].thisOutOffset = thisOutOffset;

                thisLen += 1;
            }

            LocalTensor<float> inputLt = queIn.AllocTensor<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                DataCopy(inputLt[i * maxD * numOfOut + outIndex], outGT[theArgs.thisOutOffset], theArgs.embedDim);
                DataCopy(inputLt[i * maxD * numOfOut + outIndex1], momentum1DevGT[theArgs.thisOutOffset],
                         theArgs.embedDim);
            }
            queIn.EnQue(inputLt);

            LocalTensor<float> newInputLt = queIn.DeQue<float>();
            LocalTensor<float> outLt = queOut.AllocTensor<float>();

            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * maxD * numOfOut + outIndex1;

                if (useRegBase) {
                    __local_mem__ float* dstGrad = (__local_mem__ float*)outLt[thisGradIndex].GetPhyAddr();
                    __local_mem__ float* dstMoment = (__local_mem__ float*)outLt[thisMomentIndex].GetPhyAddr();
                    __local_mem__ float* srcGrad = (__local_mem__ float*)newInputLt[thisGradIndex].GetPhyAddr();
                    __local_mem__ float* srcMoment = (__local_mem__ float*)newInputLt[thisMomentIndex].GetPhyAddr();

                    constexpr uint32_t vecLen = AscendC::GetVecLen();
                    constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
                    uint16_t repeatCount = (theArgs.embedDim + oneRepeat - 1) / oneRepeat;

                    VF_CALL<AdagradCompute<float>>(dstGrad, dstMoment, srcGrad, srcMoment, theArgs.embedDim,
                                                   repeatCount, oneRepeat, eps, learning_rate);
                } else {
                    Mul<float>(outLt[thisGradIndex], newInputLt[thisGradIndex], newInputLt[thisGradIndex],
                               theArgs.embedDim);
                    Add<float>(outLt[thisGradIndex], newInputLt[thisMomentIndex], outLt[thisGradIndex],
                               theArgs.embedDim);

                    Sqrt<float>(outLt[thisGradIndex], outLt[thisGradIndex], theArgs.embedDim);
                    Adds<float>(outLt[thisGradIndex], outLt[thisGradIndex], eps, theArgs.embedDim);
                    Duplicate<float>(outLt[thisMomentIndex], learning_rate, theArgs.embedDim);
                    Div<float>(outLt[thisGradIndex], outLt[thisMomentIndex], outLt[thisGradIndex], theArgs.embedDim);

                    Mul<float>(outLt[thisGradIndex], outLt[thisGradIndex], newInputLt[thisGradIndex], theArgs.embedDim);
                    Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], -1, theArgs.embedDim);

                    Mul<float>(outLt[thisMomentIndex], newInputLt[thisGradIndex], newInputLt[thisGradIndex],
                               theArgs.embedDim);
                }
            }

            queOut.EnQue(outLt);
            queIn.FreeTensor(newInputLt);
            LocalTensor<float> newOutLt = queOut.DeQue<float>();
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * maxD * numOfOut + outIndex1;
                DataCopy(weightsDevOutGT[theArgs.thisOutOffset], newOutLt[thisGradIndex], theArgs.embedDim);
                DataCopy(momentum1DevOutGT[theArgs.thisOutOffset], newOutLt[thisMomentIndex], theArgs.embedDim);
            }
            SetAtomicNone();
            queOut.FreeTensor(newOutLt);
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
};
}  // namespace BackwardCodegenAdagradUnweightedExact
#endif
