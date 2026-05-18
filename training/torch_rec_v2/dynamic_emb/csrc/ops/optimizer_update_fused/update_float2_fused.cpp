/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <type_traits>
#include <iostream>
#include "../AdamW_update/AdamW_update_kernel.h"
#include "../Adagrad_update/Adagrad_update_kernel.h"
#include "../Rowwise_adagrad_update/Rowwise_adagrad_update_kernel.h"
#include "../sgd_update/sgd_update_kernel.h"
#include "../../optimizer_kind.h"
#include "update_float2_fused_kernel.h"
#include "../ops_utils.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_THREADS = UpdateFloat2FusedSimt::MAX_THREADS_PER_BLOCK;

template <int32_t kMaxElementsPerThread, typename OptimizerT>
__aicore__ inline void VfCallSimtSmallInBlockFloat2Fused(
    __gm__ float2* grads, __gm__ float2* values, __gm__ bool* founds, uint32_t gradDimVec, uint32_t valDim,
    int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float decayFactor, float eps, int32_t gradDimVecShift, bool isPowerOfTwo, OptimizerT optimizer)
{
    if (isPowerOfTwo) {
        AscendC::Simt::VF_CALL<UpdateFloat2FusedSimt::SimtSmallInBlockDataCompute<kMaxElementsPerThread, true, OptimizerT>>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, valDim, inVecLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            gradDimVecShift, optimizer);
    } else {
        AscendC::Simt::VF_CALL<UpdateFloat2FusedSimt::SimtSmallInBlockDataCompute<kMaxElementsPerThread, false, OptimizerT>>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, valDim, inVecLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            gradDimVecShift, optimizer);
    }
}

template <int32_t kMaxElementsPerThread, typename OptimizerT>
__aicore__ inline void VfCallSimtLargeDataFloat2Fused(
    __gm__ float2* grads, __gm__ float2* values, __gm__ bool* founds, uint32_t gradDimVec, uint32_t valDim,
    int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks, int32_t blockStartIdx, int32_t curBlocksCount,
    int32_t gradDimVecShift, bool isPowerOfTwo, OptimizerT optimizer)
{
    if (isPowerOfTwo) {
        AscendC::Simt::VF_CALL<UpdateFloat2FusedSimt::SimtLargeDataCompute<kMaxElementsPerThread, true, OptimizerT>>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, valDim, inVecLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            totalBlocks, blockStartIdx, curBlocksCount, gradDimVecShift, optimizer);
    } else {
        AscendC::Simt::VF_CALL<UpdateFloat2FusedSimt::SimtLargeDataCompute<kMaxElementsPerThread, false, OptimizerT>>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, valDim, inVecLength,
            beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            totalBlocks, blockStartIdx, curBlocksCount, gradDimVecShift, optimizer);
    }
}

template <int32_t kMaxElementsPerThread, typename OptimizerT>
__aicore__ inline void DispatchOptimizerUpdateFusedFloat2(
    __gm__ float2* grads, __gm__ float2* values, __gm__ bool* founds, bool isPowerOfTwo,
    uint32_t gradDimVec, uint32_t valDim, int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1,
    float oneMinusBeta2, float stepSize, float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks,
    int32_t blocksPerCore, int32_t remainderBlocks, bool isSmall, int32_t gradDimVecShift, int32_t coreId)
{
    OptimizerT optimizer;
    if (isSmall) {
        VfCallSimtSmallInBlockFloat2Fused<kMaxElementsPerThread, OptimizerT>(grads, values, founds, gradDimVec, valDim,
            inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            gradDimVecShift, isPowerOfTwo, optimizer);
    } else {
        int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);
        VfCallSimtLargeDataFloat2Fused<kMaxElementsPerThread, OptimizerT>(grads, values, founds, gradDimVec, valDim,
            inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, totalBlocks,
            blockStartIdx, curBlocksCount, gradDimVecShift, isPowerOfTwo, optimizer);
    }
}

template <int32_t kMaxElementsPerThread>
__aicore__ inline void DispatchOptimizerUpdateFusedFloat2ByKind(
    OptimizerKind kind, __gm__ float2* gradsPtr, __gm__ float2* valuesPtr, __gm__ bool* foundsPtr, bool isPowerOfTwo,
    uint32_t gradDimVec, uint32_t valDim, int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1,
    float oneMinusBeta2, float stepSize, float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks,
    int32_t blocksPerCore, int32_t remainderBlocks, bool isSmall, int32_t gradDimVecShift, int32_t coreId)
{
    switch (kind) {
        case OptimizerKind::AdamW:
            DispatchOptimizerUpdateFusedFloat2<kMaxElementsPerThread, AdamWOptimizer>(gradsPtr, valuesPtr, foundsPtr,
                isPowerOfTwo, gradDimVec, valDim, inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize,
                invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift,
                coreId);
            break;
        case OptimizerKind::AdaGrad:
            DispatchOptimizerUpdateFusedFloat2<kMaxElementsPerThread, AdaGradOptimizer>(gradsPtr, valuesPtr, foundsPtr,
                isPowerOfTwo, gradDimVec, valDim, inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize,
                invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift,
                coreId);
            break;
        case OptimizerKind::RowWiseAdaGrad:
            DispatchOptimizerUpdateFusedFloat2<kMaxElementsPerThread, RowWiseAdaGradOptimizer>(gradsPtr, valuesPtr,
                foundsPtr, isPowerOfTwo, gradDimVec, valDim, inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2,
                stepSize, invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall,
                gradDimVecShift, coreId);
            break;
        case OptimizerKind::SGD:
            DispatchOptimizerUpdateFusedFloat2<kMaxElementsPerThread, SGDOptimizer>(gradsPtr, valuesPtr, foundsPtr,
                isPowerOfTwo, gradDimVec, valDim, inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize,
                invVHatDenom, decayFactor, eps, totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift,
                coreId);
            break;
        default:
            return;
    }
}

__global__ __aicore__ void update_float2_fused(GM_ADDR grads, GM_ADDR values, GM_ADDR founds, uint32_t gradDim,
    uint32_t valDim, int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks, int32_t blocksPerCore,
    int32_t remainderBlocks, bool isSmall, uint32_t gradTypeRaw, uint32_t weightTypeRaw, uint32_t optimizerKindRaw,
    int32_t maxElementsPerThread)
{
    int32_t coreId = AscendC::GetBlockIdx();

    __gm__ float2* gradsPtr = reinterpret_cast<__gm__ float2*>(grads);
    __gm__ float2* valuesPtr = reinterpret_cast<__gm__ float2* >(values);
    __gm__ bool* foundsPtr = reinterpret_cast<__gm__ bool*>(founds);
    uint32_t gradDimVec = gradDim >> 1;
    uint32_t valDimVec = valDim >> 1;

    bool isPowerOfTwo = (gradDimVec & (gradDimVec - 1)) == 0;
    int32_t gradDimVecShift = 0;
    if (isPowerOfTwo) {
        uint32_t gradDimCopy = gradDimVec;
        while ((gradDimCopy >>= 1) != 0) {
            gradDimVecShift++;
        }
    }
    OptimizerKind kind = static_cast<OptimizerKind>(optimizerKindRaw);
    if (maxElementsPerThread == 2) {
        DispatchOptimizerUpdateFusedFloat2ByKind<2>(kind, gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDimVec,
            valDimVec, inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift, coreId);
    } else {
        DispatchOptimizerUpdateFusedFloat2ByKind<4>(kind, gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDimVec,
            valDimVec, inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
            totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift, coreId);
    }
}
