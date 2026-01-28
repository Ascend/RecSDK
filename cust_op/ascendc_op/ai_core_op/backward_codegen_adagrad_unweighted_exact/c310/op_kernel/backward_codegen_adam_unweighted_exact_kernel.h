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

constexpr int NUM_OUTPUTS = 3; // grad, momentum1, momentum2
constexpr int GRAD_OFFSET_IDX = 0;
constexpr int MOMENTUM1_OFFSET_IDX = 1;
constexpr int MOMENTUM2_OFFSET_IDX = 2;

template <typename T>
__aicore__ inline void AdamCompute(__local_mem__ T* dstG, __local_mem__ T* dstM1, __local_mem__ T* dstM2,
                                   __local_mem__ T* srcG, __local_mem__ T* srcM1, __local_mem__ T* srcM2,
                                   uint32_t calCount, uint16_t repeatCount, uint32_t oneRepeat,
                                   float eps, float beta1, float oneMinusBeta1, float beta2, float oneMinusBeta2,
                                   float stepSize)
{
    AscendC::MicroAPI::RegTensor<T> dstVregG;
    AscendC::MicroAPI::RegTensor<T> dstVregM1;
    AscendC::MicroAPI::RegTensor<T> dstVregM2;
    AscendC::MicroAPI::RegTensor<T> srcVregG;
    AscendC::MicroAPI::RegTensor<T> srcVregM1;
    AscendC::MicroAPI::RegTensor<T> srcVregM2;
    AscendC::MicroAPI::MaskReg mask;

    for (uint16_t i = 0; i < repeatCount; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<uint32_t>(calCount);
        AscendC::MicroAPI::DataCopy(srcVregG, srcG + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregM1, srcM1 + i * oneRepeat);
        AscendC::MicroAPI::DataCopy(srcVregM2, srcM2 + i * oneRepeat);

        AscendC::MicroAPI::Muls(dstVregM1, srcVregM1, beta1, mask);
        AscendC::MicroAPI::Muls(dstVregG, srcVregG, oneMinusBeta1, mask);
        AscendC::MicroAPI::Add(dstVregM1, dstVregM1, dstVregG, mask);

        AscendC::MicroAPI::Muls(dstVregM2, srcVregM2, beta2, mask);
        AscendC::MicroAPI::Mul(dstVregG, srcVregG, srcVregG, mask);
        AscendC::MicroAPI::Muls(dstVregG, dstVregG, oneMinusBeta2, mask);
        AscendC::MicroAPI::Add(dstVregM2, dstVregM2, dstVregG, mask);

        AscendC::MicroAPI::Sqrt(srcVregM2, dstVregM2, mask);
        AscendC::MicroAPI::Adds(srcVregM2, srcVregM2, eps, mask);
        AscendC::MicroAPI::Div(dstVregG, dstVregM1, srcVregM2, mask);
        AscendC::MicroAPI::Muls(dstVregG, dstVregG, stepSize, mask);

        AscendC::MicroAPI::DataCopy(dstG + i * oneRepeat, dstVregG, mask);
        AscendC::MicroAPI::DataCopy(dstM1 + i * oneRepeat, dstVregM1, mask);
        AscendC::MicroAPI::DataCopy(dstM2 + i * oneRepeat, dstVregM2, mask);
    }
}

class BackwardCodegenAdamUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenAdamUnweightedExactKernel() {}

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

        numOfOut = NUM_OUTPUTS;  // 输出个数为3：grad, momentum1, momentum2
        indicesNumOneBlock = blockLen / numOfOut / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        outIndex = GRAD_OFFSET_IDX * maxD;   // grad偏移
        outIndex1 = MOMENTUM1_OFFSET_IDX * maxD;  // momentum1偏移
        outIndex2 = MOMENTUM2_OFFSET_IDX * maxD;  // momentum2偏移
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
            UpdateArgs theArgs = updateArgs[i];
            DataCopy(inputLt[i * maxD * numOfOut + outIndex], outGT[theArgs.thisOutOffset], theArgs.embedDim);
            DataCopy(inputLt[i * maxD * numOfOut + outIndex1], momentum1DevGT[theArgs.thisOutOffset], theArgs.embedDim);
            DataCopy(inputLt[i * maxD * numOfOut + outIndex2], momentum2DevGT[theArgs.thisOutOffset], theArgs.embedDim);
        }
        queIn.EnQue(inputLt);
    }

    __aicore__ inline void ComputeAdam(UpdateArgs* updateArgs, int64_t cnt)
    {
        float oneMinusBeta1 = (1 - beta1);
        float oneMinusBeta2 = (1 - beta2);
        float minusLearningRate = -learning_rate;
        float stepSize = minusLearningRate * beta2sqrt;

        LocalTensor<float> inputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();

        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
            int64_t thisMoment1Index = i * maxD * numOfOut + outIndex1;
            int64_t thisMoment2Index = i * maxD * numOfOut + outIndex2;

            if (useRegBase) {
                __local_mem__ float* dstG = (__local_mem__ float*)outLt[thisGradIndex].GetPhyAddr();
                __local_mem__ float* dstM1 = (__local_mem__ float*)outLt[thisMoment1Index].GetPhyAddr();
                __local_mem__ float* dstM2 = (__local_mem__ float*)outLt[thisMoment2Index].GetPhyAddr();
                __local_mem__ float* srcG = (__local_mem__ float*)inputLt[thisGradIndex].GetPhyAddr();
                __local_mem__ float* srcM1 = (__local_mem__ float*)inputLt[thisMoment1Index].GetPhyAddr();
                __local_mem__ float* srcM2 = (__local_mem__ float*)inputLt[thisMoment2Index].GetPhyAddr();

                constexpr uint32_t vecLen = AscendC::GetVecLen();
                constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
                uint16_t repeatCount = (theArgs.embedDim + oneRepeat - 1) / oneRepeat;

                VF_CALL<AdamCompute<float>>(dstG, dstM1, dstM2, srcG, srcM1, srcM2,
                                            theArgs.embedDim, repeatCount, oneRepeat,
                                            eps, beta1, oneMinusBeta1,  beta2, oneMinusBeta2, stepSize);
            } else {
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
                Adds<float>(inputLt[thisMoment2Index], inputLt[thisMoment2Index], eps, theArgs.embedDim);
                Div<float>(outLt[thisGradIndex], outLt[thisMoment1Index], inputLt[thisMoment2Index], theArgs.embedDim);
                Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], stepSize, theArgs.embedDim);
            }
        }

        queOut.EnQue(outLt);
        queIn.FreeTensor(inputLt);
    }

    __aicore__ inline void DataCopyOut(UpdateArgs* updateArgs, int64_t cnt)
    {
        LocalTensor<float> outLt = queOut.DeQue<float>();
        SetAtomicAdd<float>();
        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
            DataCopy(weightsDevOutGT[theArgs.thisOutOffset], outLt[thisGradIndex], theArgs.embedDim);
        }
        SetAtomicNone();

        for (int64_t i = 0; i < cnt; i++) {
            UpdateArgs theArgs = updateArgs[i];
            int64_t thisMoment1Index = i * maxD * numOfOut + outIndex1;
            int64_t thisMoment2Index = i * maxD * numOfOut + outIndex2;
            DataCopy(momentum1DevOutGT[theArgs.thisOutOffset], outLt[thisMoment1Index], theArgs.embedDim);
            DataCopy(momentum2DevOutGT[theArgs.thisOutOffset], outLt[thisMoment2Index], theArgs.embedDim);
        }

        queOut.FreeTensor(outLt);
    }

    __aicore__ inline void UpdateEmbedAdam(Args args)
    {
        this->UniqIndices();
        SyncAll();
        
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
    GM_ADDR momentum2Dev;
    GM_ADDR momentum2DevOut;

    GlobalTensor<float> momentum2DevGT;
    GlobalTensor<float> momentum2DevOutGT;

    float beta1;
    float beta2;
    float beta1pow;
    float beta2pow;
    float beta2sqrt;
    int64_t iter;

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
