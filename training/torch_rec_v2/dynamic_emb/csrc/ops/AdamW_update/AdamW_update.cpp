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
#include "AdamW_update_kernel.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_THREADS = AdamWUpdateSimt::MAX_THREADS_PER_BLOCK;

extern "C" __global__ __aicore__ void AdamW_update(GM_ADDR grads, GM_ADDR values, int32_t gradDim, int32_t inLength,
    float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize, float invVHatDenom,
    float decayFactor, float eps, int32_t totalBlocks, int32_t blocksPerCore, int32_t remainderBlocks, bool isSmall)
{
    int32_t coreId = AscendC::GetBlockIdx();

    __gm__ float* gradsPtr = reinterpret_cast<__gm__ float*>(grads);
    __gm__ float* __gm__* valuesPtr = reinterpret_cast<__gm__ float* __gm__*>(values);

    bool isPowerOfTwo = (gradDim & (gradDim - 1)) == 0;

    int32_t gradDimShift = 0;
    if (isPowerOfTwo) {
        int32_t gradDimCopy = gradDim;
        while (gradDimCopy >>= 1) {
            gradDimShift++;
        }
    }

    if (isSmall) {
        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<AdamWUpdateSimt::SimtSmallInBlockDataCompute<true>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradDimShift);
        } else {
            AscendC::Simt::VF_CALL<AdamWUpdateSimt::SimtSmallInBlockDataCompute<false>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradDimShift);
        }
    } else {
        int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<AdamWUpdateSimt::SimtLargeDataCompute<true>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimShift);
        } else {
            AscendC::Simt::VF_CALL<AdamWUpdateSimt::SimtLargeDataCompute<false>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDim, inLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimShift);
        }
    }
}