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

#include "AdamW_update_float2_kernel.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_THREADS = AdamWUpdateFloat2Simt::MAX_THREADS_PER_BLOCK;

extern "C" __global__ __aicore__ void AdamW_update_float2(GM_ADDR grads, GM_ADDR values, uint32_t gradDim,
    int32_t inVecLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks, int32_t blocksPerCore,
    int32_t remainderBlocks, bool isSmall)
{
    // 获取当前核心ID
    int32_t coreId = AscendC::GetBlockIdx();

    // 强转为 float2 指针
    __gm__ float2* gradsPtr = reinterpret_cast<__gm__ float2*>(grads);
    __gm__ float2* __gm__* valuesPtr = reinterpret_cast<__gm__ float2* __gm__*>(values);

    // float2 下的 gradDim 步长，右移一位，等价于除以 2
    uint32_t gradDimVec = gradDim >> 1;

    // 判断 gradDimVec 是否为 2 的幂，便于后续逻辑分发
    bool isPowerOfTwo = (gradDimVec & (gradDimVec - 1)) == 0;
    int32_t gradDimVecShift = 0;
    if (isPowerOfTwo) {
        uint32_t gradDimCopy = gradDimVec;
        while ((gradDimCopy >>= 1) != 0) {
            gradDimVecShift++;
        }
    }

    if (isSmall) {
        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<AdamWUpdateFloat2Simt::SimtSmallInBlockDataCompute<true>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradDimVecShift);
        } else {
            AscendC::Simt::VF_CALL<AdamWUpdateFloat2Simt::SimtSmallInBlockDataCompute<false>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps, gradDimVecShift);
        }
    } else {
        int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

        if (isPowerOfTwo) {
            AscendC::Simt::VF_CALL<AdamWUpdateFloat2Simt::SimtLargeDataCompute<true>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimVecShift);
        } else {
            AscendC::Simt::VF_CALL<AdamWUpdateFloat2Simt::SimtLargeDataCompute<false>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDimVec, inVecLength,
                beta1, beta2, oneMinusBeta1, oneMinusBeta2, stepSize, invVHatDenom, decayFactor, eps,
                totalBlocks, blockStartIdx, curBlocksCount, gradDimVecShift);
        }
    }
}