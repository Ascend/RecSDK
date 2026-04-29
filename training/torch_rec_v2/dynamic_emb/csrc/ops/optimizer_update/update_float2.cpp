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
#include "update_float2_kernel.h"
#include "../ops_utils.h"
#include "kernel_operator.h"


constexpr int32_t BLOCK_THREADS = UpdateFloat2Simt::MAX_THREADS_PER_BLOCK;

template <typename OptimizerT>
__aicore__ inline void DispatchOptimizerUpdateFloat2(
    __gm__ float2* gradsPtr, __gm__ float2* __gm__* valuesPtr, __gm__ bool* foundsPtr, bool isPowerOfTwo,
    uint32_t gradDimVec, int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
    float stepSize, float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks, int32_t blocksPerCore,
    int32_t remainderBlocks, bool isSmall, int32_t gradDimVecShift, int32_t coreId)
{
    OptimizerT optimizer;
    __gm__ float2* grads = reinterpret_cast<__gm__ float2*>(gradsPtr);
    __gm__ float2* __gm__* values = reinterpret_cast<__gm__ float2* __gm__*>(valuesPtr);
    __gm__ bool* founds = reinterpret_cast<__gm__ bool*>(foundsPtr);
    if (isSmall) {
        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<UpdateFloat2Simt::SimtSmallInBlockDataCompute<true, OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                gradDimVecShift, optimizer);
        } else {
            AscendC::Simt::VF_CALL<UpdateFloat2Simt::SimtSmallInBlockDataCompute<false, OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                gradDimVecShift, optimizer);
        }

    } else {
        int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);
       
        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<UpdateFloat2Simt::SimtLargeDataCompute<true,  OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimVecShift, optimizer);
        } else {
            AscendC::Simt::VF_CALL<UpdateFloat2Simt::SimtLargeDataCompute<false,  OptimizerT>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, grads, values, founds, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimVecShift, optimizer);
        }

    }
}

__global__ __aicore__ void update_float2(GM_ADDR grads, GM_ADDR values, GM_ADDR founds, uint32_t gradDim,
    int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks,
    bool isSmall, uint32_t gradTypeRaw, uint32_t weightTypeRaw, uint32_t optimizerKindRaw)
{
    int32_t coreId = AscendC::GetBlockIdx();

    __gm__ float2* gradsPtr = reinterpret_cast<__gm__ float2*>(grads);
    __gm__ float2* __gm__* valuesPtr = reinterpret_cast<__gm__ float2* __gm__*>(values);
    __gm__ bool* foundsPtr = reinterpret_cast<__gm__ bool*>(founds);
    uint32_t gradDimVec = gradDim >> 1;

    bool isPowerOfTwo = (gradDimVec & (gradDimVec - 1)) == 0;
    int32_t gradDimVecShift = 0;
    if (isPowerOfTwo) {
        uint32_t gradDimCopy = gradDimVec;
        while ((gradDimCopy >>= 1) != 0) {
            gradDimVecShift++;
        }
    }
    OptimizerKind kind = static_cast<OptimizerKind>(optimizerKindRaw);
    switch (kind) {
        // 分支 1：AdamW 优化器
        case OptimizerKind::AdamW:
            DispatchOptimizerUpdateFloat2<AdamWOptimizer>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDimVec,
                inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift, coreId);
            break;
        // 分支 2：AdaGrad 优化器
        case OptimizerKind::AdaGrad:
            DispatchOptimizerUpdateFloat2<AdaGradOptimizer>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDimVec,
                inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift, coreId);
            break;
        // 分支 3：RowWiseAdaGrad 优化器
        case OptimizerKind::RowWiseAdaGrad:
            DispatchOptimizerUpdateFloat2<RowWiseAdaGradOptimizer>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDimVec,
                inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift, coreId);
            break;
        // 分支 4：SGD优化器
        case OptimizerKind::SGD:
            DispatchOptimizerUpdateFloat2<SGDOptimizer>(gradsPtr, valuesPtr, foundsPtr, isPowerOfTwo, gradDimVec,
                inVecLength, beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blocksPerCore, remainderBlocks, isSmall, gradDimVecShift, coreId);
            break;
        default:
            AscendC::printf("Unsupported optimizer kind: %d\n", optimizerKindRaw);
            return;
    }
}

