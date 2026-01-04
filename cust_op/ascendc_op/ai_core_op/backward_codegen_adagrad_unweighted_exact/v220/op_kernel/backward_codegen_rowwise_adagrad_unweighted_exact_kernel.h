/**
 * @file backward_codegen_rowwise_adagrad_unweighted_exact_kernel.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef BACKWARD_CODEGEN_ROWWISE_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_H
#define BACKWARD_CODEGEN_ROWWISE_ADAGRAD_UNWEIGHTED_EXACT_KERNEL_H

#include "kernel_operator.h"
#include "backward_codegen_unweighted_exact_kernel.h"

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenRowwiseAdagradUnweightedExact {

class BackwardCodegenRowwiseAdagradUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenRowwiseAdagradUnweightedExactKernel() {}
    __aicore__ inline ~BackwardCodegenRowwiseAdagradUnweightedExactKernel() {}

    __aicore__ inline void UpdateEmbedRowwiseAda()
    {
        int MOMENTUM_PAD_NUM = 16;

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
        
        TQue<TPosition::VECIN, 1> queReduction;
        TQue<TPosition::VECIN, 1> queGradSq;
        pipe.InitBuffer(queReduction, 1, sizeof(float)); // 只需 1 个 float
        pipe.InitBuffer(queGradSq, 1, maxD * sizeof(float)); // 只需 maxD 个 float
       
        while (remain > 0) {
            int64_t thisLen = 0;
            while (thisLen < indicesNumOneBlock && remain > 0) {
                int64_t thisIndForTotalTable = thisTableOffset + total - remain;
                remain = remain - 1;
                if (thisIndForTotalTable >= hashSizeCumsumGT.GetValue(tableIndex + 1)) {
                    tableIndex = tableIndex + 1;
                }

                if (workspaceGT.GetValue(thisIndForTotalTable) != NEED_UPDATE) {
                    continue;
                }

                int64_t thisIndForThisTable = thisIndForTotalTable - hashSizeCumsumGT.GetValue(tableIndex);
                int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
                int64_t thisWeightOffset = *(weightsOffsetsPtr + tableIndex);
                int64_t thisOutOffset = thisWeightOffset + thisIndForThisTable * embedDim;

                updateArgs[thisLen].embedDim = embedDim;
                updateArgs[thisLen].thisOutOffset = thisOutOffset;
                updateArgs[thisLen].thisMomentumOffset = thisIndForTotalTable * MOMENTUM_PAD_NUM;

                thisLen += 1;
            }
            
            LocalTensor<float> inputLt = queIn.AllocTensor<float>();

            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                DataCopy(inputLt[i * maxD * numOfOut + outIndex], outGT[theArgs.thisOutOffset], theArgs.embedDim);
                DataCopy(inputLt[i * maxD * numOfOut + outIndex1], momentum1DevGT[theArgs.thisMomentumOffset],
                         MOMENTUM_PAD_NUM);
            }
            queIn.EnQue(inputLt);

            LocalTensor<float> newInputLt = queIn.DeQue<float>();
            LocalTensor<float> outLt = queOut.AllocTensor<float>();
            LocalTensor<float> sumSqTensor = queReduction.AllocTensor<float>();
            LocalTensor<float> gradSq = queGradSq.AllocTensor<float>();

            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * maxD * numOfOut + outIndex1;
                int64_t embedDim = theArgs.embedDim;
                Mul(gradSq, newInputLt[thisGradIndex], newInputLt[thisGradIndex], embedDim);

                // Step 2: ReduceSum(grad^2) -> scalar
                uint32_t srcShape[2] = {1, static_cast<uint32_t>(embedDim)};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumSqTensor, gradSq, srcShape, false);
                float sum_sq = sumSqTensor.GetValue(0);
                float mean_sq = sum_sq / static_cast<float>(embedDim);
                float current_momentum = newInputLt.GetValue(thisMomentIndex);
                float new_momentum = current_momentum + mean_sq;
                
                float adaptive_lr = learning_rate / (sqrt(new_momentum) + eps);
                // Step 3: 计算 delta = -adaptive_lr * grad
 
                Muls(outLt[thisGradIndex], newInputLt[thisGradIndex], -adaptive_lr, embedDim);
                // Step 4: 更新 momentum = old_momentum + mean_sq
                outLt.SetValue(thisMomentIndex, mean_sq);
            }
            queGradSq.FreeTensor(gradSq);
            queReduction.FreeTensor(sumSqTensor);
            queOut.EnQue(outLt);
            queIn.FreeTensor(newInputLt);
            
            LocalTensor<float> newOutLt = queOut.DeQue<float>();
            
            SetAtomicAdd<float>();
            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * maxD * numOfOut + outIndex1;
                DataCopy(weightsDevOutGT[theArgs.thisOutOffset], newOutLt[thisGradIndex], theArgs.embedDim);
                DataCopy(momentum1DevOutGT[theArgs.thisMomentumOffset], newOutLt[thisMomentIndex], MOMENTUM_PAD_NUM);
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
}  // namespace BackwardCodegenRowwiseAdagradUnweightedExact
#endif