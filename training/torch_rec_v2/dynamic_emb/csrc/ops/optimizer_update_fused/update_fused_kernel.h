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
#pragma once
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

namespace UpdateFusedSimt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;

template <bool isPowerOfTwo, typename grad_t, typename weight_t, typename OptimizerFunc>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallInBlockDataCompute(
    __gm__ grad_t* grads, __gm__ weight_t*  valuesPtr,  __gm__ bool* founds, uint32_t gradDim,
    uint32_t valDim, int32_t inLength, float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, 
    float stepSize,float invVHatDenom, float decayFactor, float eps, int32_t gradDimShift, 
    OptimizerFunc optimizer)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t blockBase = blockIdx * blockElementCapacity;
    int32_t lastRowIdx = -1;
    __gm__ weight_t* valuesRowBasePtr = nullptr;

#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
        int32_t globalIdx = blockBase + i * blockThreadNum + threadIdx;
        if (globalIdx >= inLength) {
            break;
        }
        int rowIdx = isPowerOfTwo ? (globalIdx >> gradDimShift) : (globalIdx / gradDim);
        bool mask = founds ? founds[rowIdx] : true;
        if (!mask) {
            continue;
        }
        int colIdx = isPowerOfTwo ? (globalIdx & (gradDim - 1)) : (globalIdx % gradDim);

        if (rowIdx != lastRowIdx) {
            valuesRowBasePtr = reinterpret_cast<__gm__ weight_t*>(valuesPtr + rowIdx * valDim);
            lastRowIdx = rowIdx;
        }

        grad_t tmpGrad = grads[globalIdx];
        optimizer.update(reinterpret_cast<__gm__ weight_t*>(valuesRowBasePtr), colIdx, gradDim, tmpGrad, beta1, beta2,
                          oneMinusBeta1, oneMinusBeta2,
                          stepSize, invVHatDenom, decayFactor, eps);
    }
}

template <bool isPowerOfTwo, typename grad_t, typename weight_t, typename OptimizerFunc>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ grad_t* grads, __gm__ weight_t*  valuesPtr,  __gm__ bool* founds, uint32_t gradDim,
    uint32_t valDim,int32_t inLength,float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
    float stepSize,float invVHatDenom, float decayFactor, float eps, int32_t totalBlocks,
    int32_t blockStartIdx, int32_t curBlocksCount, int32_t gradDimShift, OptimizerFunc optimizer)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        if (blockBase >= inLength) {
            break;
        }

        int32_t lastRowIdx = -1;
        __gm__ weight_t* valuesRowBasePtr = nullptr;

#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
            int32_t globalIdx = blockBase + i * blockThreadNum + threadIdx;
            if (globalIdx >= inLength) {
                break;
            }
            int rowIdx = isPowerOfTwo ? (globalIdx >> gradDimShift) : (globalIdx / gradDim);
            bool mask = founds ? founds[rowIdx] : true;
            if (!mask) {
                continue;
            }
            int colIdx = isPowerOfTwo ? (globalIdx & (gradDim - 1)) : (globalIdx % gradDim);

            if (rowIdx != lastRowIdx) {
                valuesRowBasePtr = reinterpret_cast<__gm__ weight_t*>(valuesPtr + rowIdx * valDim);
                lastRowIdx = rowIdx;
            }

            grad_t tmpGrad = grads[globalIdx];
            optimizer.update(reinterpret_cast<__gm__ weight_t*>(valuesRowBasePtr), colIdx, gradDim, tmpGrad, beta1, beta2,
                             oneMinusBeta1, oneMinusBeta2,
                             stepSize, invVHatDenom, decayFactor, eps);
        }
    }
}

} // namespace UpdateSimt