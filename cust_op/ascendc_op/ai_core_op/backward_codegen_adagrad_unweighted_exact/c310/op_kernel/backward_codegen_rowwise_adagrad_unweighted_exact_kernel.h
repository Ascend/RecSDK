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

        int64_t tableHashStart = hashSizeCumsumGT.GetValue(tableIndex);
        int64_t tableHashEnd = hashSizeCumsumGT.GetValue(tableIndex + 1);
        int64_t embedDim = *(dOffsetsPtr + tableIndex + 1) - *(dOffsetsPtr + tableIndex);
        int64_t weightOffsetBase = *(weightsOffsetsPtr + tableIndex);

        TQue<TPosition::VECIN, 1> queGradSq;
        pipe.InitBuffer(queGradSq, 1, maxD * sizeof(float));
        TQue<TPosition::VECIN, 1> queScalar;
        pipe.InitBuffer(queScalar, 2, sizeof(float));
        auto adaptiveLr = queScalar.AllocTensor<float>();

        TQue<TPosition::VECIN, 1> queArg;
        pipe.InitBuffer(queArg, 1, 3 * sizeof(int64_t) * MAX_ARGS_PIPE_LEN);

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

                if (workspaceGT.GetValue(thisIndForTotalTable) != NEED_UPDATE) {
                    continue;
                }

                int64_t thisIndForThisTable = thisIndForTotalTable - tableHashStart;
                int64_t thisOutOffset = weightOffsetBase + thisIndForThisTable * embedDim;

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
            LocalTensor<float> gradSq = queGradSq.AllocTensor<float>();
            LocalTensor<float> accumScalar = queScalar.AllocTensor<float>();

            for (int64_t i = 0; i < thisLen; i++) {
                UpdateArgs theArgs = updateArgs[i];
                int64_t thisGradIndex = i * maxD * numOfOut + outIndex;
                int64_t thisMomentIndex = i * maxD * numOfOut + outIndex1;
                int64_t embedDim = theArgs.embedDim;
                useRegBase = false;
                const float inv_embed_dim = 1.0f / static_cast<float>(embedDim);
                
                // 1. grad^2
                Mul<float>(gradSq, newInputLt[thisGradIndex], newInputLt[thisGradIndex], embedDim);

                // 2. Compute mean_sq
                uint32_t srcShape[2] = {1, static_cast<uint32_t>(embedDim)};
                ReduceSum<float, Pattern::Reduce::AR, false>(adaptiveLr, gradSq, srcShape, false);
                Muls<float>(outLt[thisMomentIndex], adaptiveLr, inv_embed_dim, 1); // adaptiveLr = mean_sq

                // 3. Compute accum_t = old_accum + mean_sq (for lr only)
                Add<float>(adaptiveLr, newInputLt[thisMomentIndex], outLt[thisMomentIndex], 1); // now = accum_t

                // 4. Compute adaptive_lr = lr / (sqrt(accum_t) + eps)
                Sqrt<float>(adaptiveLr, adaptiveLr, 1);
                Adds<float>(adaptiveLr, adaptiveLr, eps, 1);
                Reciprocal<float>(adaptiveLr, adaptiveLr, 1);
                Muls<float>(adaptiveLr, adaptiveLr, learning_rate, 1);

                // 5. Apply to gradient
                Duplicate<float>(gradSq, adaptiveLr, embedDim);
                Mul<float>(outLt[thisGradIndex], gradSq, newInputLt[thisGradIndex], embedDim);
                Muls<float>(outLt[thisGradIndex], outLt[thisGradIndex], -1.0f, embedDim);
            }

            queGradSq.FreeTensor(gradSq);
            queScalar.FreeTensor(accumScalar);
            queScalar.FreeTensor(adaptiveLr);

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