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

#include <type_traits>
#include "backward_codegen_unweighted_exact_kernel.h"


/// Padding number for momentum storage
constexpr int64_t MOMENTUM_STORAGE_PAD_NUM = 16;

/// Number of output tensors
constexpr int OUTPUT_TENSOR_COUNT = 2;
/// Index for output gradient tensor in the output tensors array
constexpr int OUTPUT_GRADIENT_TENSOR_INDEX = 0;
/// Index for output momentum tensor in the output tensors array
constexpr int OUTPUT_MOMENTUM_TENSOR_INDEX = 1;

using namespace AscendC;
using namespace BackwardCodegenUnweightedExact;

namespace BackwardCodegenRowwiseAdagradUnweightedExact {
/**
 * Performs Adagrad computation for rowwise updates
 *
 * @tparam T Data type
 * @param dstGrad Destination gradient tensor
 * @param dstMoment Destination moment tensor
 * @param srcGrad Source gradient tensor
 * @param srcMoment Source moment tensor
 * @param calCount Number of elements to calculate
 * @param repeatCount Number of repeat operations
 * @param oneRepeat Size of one repeat operation
 * @param eps Epsilon value for numerical stability
 * @param learning_rate Learning rate
 * @param invEmbedDim Inverse of embedding dimension
 */
template <typename T>
__aicore__ __simd_vf__ inline void AdagradCompute(__local_mem__ T* dstGrad, __local_mem__ T* dstMoment,
                                                  __local_mem__ T* srcGrad, __local_mem__ T* srcMoment,
                                                  uint32_t calCount, uint16_t repeatCount, uint32_t oneRepeat,
                                                  float eps, float learning_rate, float invEmbedDim)
{
    // === Scalar mask: must be a variable (lvalue) ===
    uint32_t scalarLen = 1;
    auto maskScalar = AscendC::MicroAPI::UpdateMask<uint32_t>(scalarLen);

    // --- Step 1: Compute total sum of squared gradients ---
    AscendC::MicroAPI::RegTensor<float> vSumTotal;
    AscendC::MicroAPI::Duplicate(vSumTotal, 0.0f, maskScalar);

    // 在函数开始处声明所有需要的变量
    uint32_t offset, remaining, blockLen;
    AscendC::MicroAPI::RegTensor<float> vGrad, vGradSq, vSumSq, vOutGrad;
    
    for (uint16_t i = 0; i < repeatCount; ++i) {
        offset = i * oneRepeat;
        remaining = calCount - offset;
        blockLen = (remaining > oneRepeat) ? oneRepeat : remaining;

        // === Vector mask: must be a variable ===
        auto maskVec = AscendC::MicroAPI::UpdateMask<uint32_t>(blockLen);

        AscendC::MicroAPI::DataCopy(vGrad, srcGrad + offset);
        AscendC::MicroAPI::Mul(vGradSq, vGrad, vGrad, maskVec);
        AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(vSumSq, vGradSq, maskVec);
        AscendC::MicroAPI::Add(vSumTotal, vSumTotal, vSumSq, maskScalar);
    }

    AscendC::MicroAPI::Muls(vSumTotal, vSumTotal, invEmbedDim, maskScalar);

    // --- Step 2: Update momentum (scalar) ---
    AscendC::MicroAPI::RegTensor<float> vMomentOld, vNewMoment;
    AscendC::MicroAPI::DataCopy(vMomentOld, srcMoment);
    AscendC::MicroAPI::Add(vNewMoment, vMomentOld, vSumTotal, maskScalar);

    AscendC::MicroAPI::RegTensor<float> vDenom, vAdaptiveLr;
    AscendC::MicroAPI::Sqrt(vDenom, vNewMoment, maskScalar);
    AscendC::MicroAPI::Adds(vDenom, vDenom, eps, maskScalar);
    AscendC::MicroAPI::Duplicate(vAdaptiveLr, learning_rate, maskScalar);
    AscendC::MicroAPI::Div(vAdaptiveLr, vAdaptiveLr, vDenom, maskScalar);

    // === Broadcast scalar lr to full vector (use full mask as lvalue) ===
    uint32_t fullVecLen = oneRepeat;
    auto fullMask = AscendC::MicroAPI::UpdateMask<uint32_t>(fullVecLen);
    AscendC::MicroAPI::Duplicate(vAdaptiveLr, vAdaptiveLr, fullMask);

    // --- Step 3: Apply update per chunk ---

    for (uint16_t i = 0; i < repeatCount; ++i) {
        offset = i * oneRepeat;
        remaining = calCount - offset;
        blockLen = (remaining > oneRepeat) ? oneRepeat : remaining;

        auto maskVec = AscendC::MicroAPI::UpdateMask<uint32_t>(blockLen);

        AscendC::MicroAPI::DataCopy(vGrad, srcGrad + offset);
        AscendC::MicroAPI::Mul(vOutGrad, vAdaptiveLr, vGrad, maskVec);
        AscendC::MicroAPI::Muls(vOutGrad, vOutGrad, -1.0f, maskVec);
        AscendC::MicroAPI::DataCopy(dstGrad + offset, vOutGrad, maskVec);
    }

    // --- Step 4: Store momentum ---
    AscendC::MicroAPI::DataCopy(dstMoment, vNewMoment, maskScalar);
}

/**
 * SIMT gather function to collect momentum values
 *
 * @param inputMoment Input momentum tensor
 * @param index Index tensor
 * @param gatherOutput Output tensor for gathered values
 * @param outputTotalLength Total length of output
 * @param maxD Maximum dimension
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtGather(__gm__ float volatile* inputMoment,
                                                                                  __gm__ int64_t volatile* index,
                                                                                  __ubuf__ float* gatherOutput,
                                                                                  int64_t outputTotalLength,
                                                                                  int64_t maxD)
{
    int idx = AscendC::Simt::GetThreadIdx<0>();
    if (idx >= outputTotalLength) {
        return;
    }

    int64_t totalIndex = index[idx * static_cast<int64_t>(TriadIndex::ACCESS_STRIDE)];

    gatherOutput[idx * maxD * OUTPUT_TENSOR_COUNT + OUTPUT_MOMENTUM_TENSOR_INDEX * maxD] =
        inputMoment[totalIndex * MOMENTUM_STORAGE_PAD_NUM];

    AscendC::Simt::ThreadBarrier();
}

/**
 * SIMT scatter function to distribute updated values
 *
 * @param inputMoment Input momentum tensor
 * @param index Index tensor
 * @param gatherOutput Values to scatter
 * @param outputTotalLength Total length of output
 * @param maxD Maximum dimension
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtScatter(__gm__ float volatile* inputMoment,
                                                                                   __gm__ volatile int64_t* index,
                                                                                   __ubuf__ float* gatherOutput,
                                                                                   int64_t outputTotalLength,
                                                                                   int64_t maxD)
{
    int idx = AscendC::Simt::GetThreadIdx<0>();
    if (idx >= outputTotalLength) {
        return;
    }
    int64_t totalIndex = index[idx * static_cast<int64_t>(TriadIndex::ACCESS_STRIDE)];

    inputMoment[totalIndex * MOMENTUM_STORAGE_PAD_NUM] =
        gatherOutput[idx * maxD * OUTPUT_TENSOR_COUNT + OUTPUT_MOMENTUM_TENSOR_INDEX * maxD];

    AscendC::Simt::ThreadBarrier();
}

/**
 * Kernel class for backward codegen rowwise adagrad unweighted exact
 */
class BackwardCodegenRowwiseAdagradUnweightedExactKernel : public BackwardCodegenUnweightedExactKernel {
public:
    __aicore__ inline BackwardCodegenRowwiseAdagradUnweightedExactKernel() {}
    __aicore__ inline ~BackwardCodegenRowwiseAdagradUnweightedExactKernel() {}

    /**
     * Initializes pipeline queues
     */
    __aicore__ inline void InitPipe()
    {
        BackwardCodegenUnweightedExactKernel::InitPipe();
    }

    __aicore__ inline void InitRowwiseAdagrad()
    {
        momentum1DevOutGT.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        momentum1DevGT.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        indicesUniqGT.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        indicesNumOneBlock = blockLen / OUTPUT_TENSOR_COUNT / maxD;
        if (indicesNumOneBlock >= MAX_ARGS_PIPE_LEN) {
            indicesNumOneBlock = MAX_ARGS_PIPE_LEN;
        }
        outIndex = OUTPUT_GRADIENT_TENSOR_INDEX * maxD;
        outIndex1 = OUTPUT_MOMENTUM_TENSOR_INDEX * maxD;

        // 初始化新加入的成员队列
        pipe.InitBuffer(queGradSq, 1, maxD * sizeof(float));
        pipe.InitBuffer(queScalar, 2, sizeof(float));
    }
    /**
     * Updates embedding using rowwise AdaGrad algorithm
     */
    __aicore__ inline void UpdateEmbedRowwiseAda()
    {
        this->UniqIndices();
        this->InitRowwiseAdagrad();

        int64_t total = validListLen;
        int64_t remain = total;

        UpdateArgs updateArgs[MAX_ARGS_PIPE_LEN];

        while (remain > 0) {
            int64_t totalProcessed = total - remain;  // 计算处理前的值
            int64_t thisLen = this->FillUpdateArgs(updateArgs, remain, total);

            this->RowwiseAdagradDataCopyIn(updateArgs, thisLen, offsetOfThisCore + totalProcessed, maxD);

            if (useRegBase) {
                this->ProcessRowwiseAdagradRegBase(updateArgs, thisLen, maxD);
            } else {
                this->ProcessRowwiseAdagradNormal(updateArgs, thisLen, maxD);
            }

            this->RowwiseAdagradDataCopyOut(updateArgs, thisLen, offsetOfThisCore + totalProcessed, maxD);
        }

        pipe_barrier(PIPE_ALL);
        SyncAll();
    }

    /**
     * Copy data from GM to local tensors for rowwise adagrad processing
     */
    __aicore__ inline void RowwiseAdagradDataCopyIn(UpdateArgs* updateArgs, int64_t thisLen, int64_t totalProcessed,
                                                    int64_t maxD)
    {
        LocalTensor<float> inputLt = queIn.AllocTensor<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            UpdateArgs& theArgs = updateArgs[i];
            DataCopy(inputLt[i * maxD * OUTPUT_TENSOR_COUNT + outIndex], outGT[theArgs.thisOutOffset],
                     theArgs.embedDim);
        }
        AscendC::Simt::VF_CALL<SimtGather>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(thisLen), 1, 1},
            (__gm__ float*)momentum1DevGT.GetPhyAddr(),
            (__gm__ int64_t*)indicesUniqGT.GetPhyAddr() +
                totalProcessed * static_cast<int64_t>(TriadIndex::ACCESS_STRIDE),
            (__ubuf__ float*)inputLt.GetPhyAddr(), thisLen, maxD);
        queIn.EnQue(inputLt);
    }

    /**
     * Copy data from local tensors back to GM for rowwise adagrad processing
     */
    __aicore__ inline void RowwiseAdagradDataCopyOut(UpdateArgs* updateArgs, int64_t thisLen, int64_t totalProcessed,
                                                     int64_t maxD)
    {
        LocalTensor<float> newOutLt = queOut.DeQue<float>();

        AscendC::Simt::VF_CALL<SimtScatter>(
            AscendC::Simt::Dim3{static_cast<uint32_t>(thisLen), 1, 1},
            (__gm__ float*)momentum1DevOutGT.GetPhyAddr(),
            (__gm__ int64_t*)indicesUniqGT.GetPhyAddr() +
                totalProcessed * static_cast<int64_t>(TriadIndex::ACCESS_STRIDE),
            (__ubuf__ float*)newOutLt.GetPhyAddr(), thisLen, maxD);

        SetAtomicAdd<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            UpdateArgs& theArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * OUTPUT_TENSOR_COUNT + outIndex;
            DataCopy(weightsDevOutGT[theArgs.thisOutOffset], newOutLt[thisGradIndex], theArgs.embedDim);
        }
        SetAtomicNone();

        queOut.FreeTensor(newOutLt);
    }
    /**
     * Main compute function
     *
     * @param args Arguments for computation
     */
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

    /**
     * Fills update arguments based on unique indices
     *
     * @param updateArgs Array of update arguments to fill
     * @param remain Reference to remaining count to update
     * @return Count of filled arguments
     */
    __aicore__ inline int64_t FillUpdateArgs(UpdateArgs* updateArgs, int64_t& remain, int64_t total)
    {
        int64_t cnt = 0;
        int64_t totalProcessed = total - remain;
        __gm__ int64_t* indicesUniqStartOffsetPtr =
            (__gm__ int64_t*)indicesUniqGT.GetPhyAddr() +
            (offsetOfThisCore + total - remain) * static_cast<int64_t>(TriadIndex::ACCESS_STRIDE);
        __gm__ int64_t* triadPtr = indicesUniqStartOffsetPtr;

        while (cnt < indicesNumOneBlock && remain > 0) {
            int64_t thisIndForTotalTable = triadPtr[static_cast<int64_t>(TriadIndex::UNIQUE_INDEX)];
            int64_t actualWeightOffset = triadPtr[static_cast<int64_t>(TriadIndex::WEIGHT_OFFSET)];
            int64_t embedDim = triadPtr[static_cast<int64_t>(TriadIndex::EMBED_DIM)];

            int64_t thisMomentumOffset = thisIndForTotalTable * MOMENTUM_STORAGE_PAD_NUM;

            updateArgs[cnt].embedDim = embedDim;
            updateArgs[cnt].thisOutOffset = actualWeightOffset;
            updateArgs[cnt].thisMomentumOffset = thisMomentumOffset;

            cnt += 1;
            triadPtr += static_cast<int64_t>(TriadIndex::ACCESS_STRIDE);
            remain -= 1;
        }
        return cnt;
    }

    /**
     * Process using register base method for rowwise adagrad
     */
    __aicore__ inline void ProcessRowwiseAdagradRegBase(UpdateArgs* updateArgs, int64_t thisLen, int64_t maxD)
    {
        LocalTensor<float> newInputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();
        for (int64_t i = 0; i < thisLen; i++) {
            UpdateArgs& theArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * OUTPUT_TENSOR_COUNT + OUTPUT_GRADIENT_TENSOR_INDEX * maxD;
            int64_t thisMomentIndex = i * maxD * OUTPUT_TENSOR_COUNT + OUTPUT_MOMENTUM_TENSOR_INDEX * maxD;
            int64_t embedDim = theArgs.embedDim;

            const float invEmbedDim = 1.0f / static_cast<float>(embedDim);
            __local_mem__ float* dstGrad = (__local_mem__ float*)outLt[thisGradIndex].GetPhyAddr();
            __local_mem__ float* dstMoment = (__local_mem__ float*)outLt[thisMomentIndex].GetPhyAddr();
            __local_mem__ float* srcGrad = (__local_mem__ float*)newInputLt[thisGradIndex].GetPhyAddr();
            __local_mem__ float* srcMoment = (__local_mem__ float*)newInputLt[thisMomentIndex].GetPhyAddr();

            constexpr uint32_t vecLen = AscendC::GetVecLen();
            constexpr uint32_t oneRepeat = vecLen / static_cast<uint32_t>(sizeof(float));
            uint16_t repeatCount = (theArgs.embedDim + oneRepeat - 1) / oneRepeat;

            VF_CALL<AdagradCompute<float>>(dstGrad, dstMoment, srcGrad, srcMoment, theArgs.embedDim, repeatCount,
                                           oneRepeat, eps, learning_rate, invEmbedDim);
        }
        queOut.EnQue(outLt);
        queIn.FreeTensor(newInputLt);
    }

    /**
     * Process using normal method for rowwise adagrad
     */
    __aicore__ inline void ProcessRowwiseAdagradNormal(UpdateArgs* updateArgs, int64_t thisLen, int64_t maxD)
    {
        LocalTensor<float> newInputLt = queIn.DeQue<float>();
        LocalTensor<float> outLt = queOut.AllocTensor<float>();
        LocalTensor<float> gradSq = queGradSq.AllocTensor<float>();
        LocalTensor<float> adaptiveLr = queScalar.AllocTensor<float>();

        for (int64_t i = 0; i < thisLen; i++) {
            UpdateArgs& theArgs = updateArgs[i];
            int64_t thisGradIndex = i * maxD * OUTPUT_TENSOR_COUNT + OUTPUT_GRADIENT_TENSOR_INDEX * maxD;
            int64_t thisMomentIndex = i * maxD * OUTPUT_TENSOR_COUNT + OUTPUT_MOMENTUM_TENSOR_INDEX * maxD;
            int64_t embedDim = theArgs.embedDim;

            const float invEmbedDim = 1.0f / static_cast<float>(embedDim);

            // 1. grad^2
            Mul<float>(gradSq, newInputLt[thisGradIndex], newInputLt[thisGradIndex], embedDim);

            // 2. Compute mean_sq
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(embedDim)};
            ReduceSum<float, Pattern::Reduce::AR, false>(adaptiveLr, gradSq, srcShape, false);
            Muls<float>(outLt[thisMomentIndex], adaptiveLr, invEmbedDim, 1);

            // 3. Compute accum_t = old_accum + mean_sq (for lr only)
            Add<float>(adaptiveLr, newInputLt[thisMomentIndex], outLt[thisMomentIndex], 1);

            // Accumulate momentum: new_momentum = old_momentum + sum_of_gradients_squared
            Add<float>(outLt[thisMomentIndex], newInputLt[thisMomentIndex], outLt[thisMomentIndex], 1);

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

        queOut.EnQue(outLt);
        queGradSq.FreeTensor(gradSq);
        queScalar.FreeTensor(adaptiveLr);
        queIn.FreeTensor(newInputLt);
    }

private:
    int indicesNumOneBlock;
    int outIndex;
    int outIndex1;
    TQue<TPosition::VECIN, 1> queGradSq;
    TQue<TPosition::VECIN, 1> queScalar;
};
}  // namespace BackwardCodegenRowwiseAdagradUnweightedExact
#endif