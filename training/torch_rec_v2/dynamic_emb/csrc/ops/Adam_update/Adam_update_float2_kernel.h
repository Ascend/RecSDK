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

namespace AdamUpdateFloat2Simt {

constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 2;

template <bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtSmallInBlockDataCompute(
    __gm__ float2* grads, __gm__ float2* __gm__* valuesPtr, uint32_t gradDimVec, int32_t inVecLength,
    float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float weightDecay, float eps, int32_t gradDimVecShift)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;
    int32_t blockBase = blockIdx * blockElementCapacity;

    int32_t lastRowIdx = -1;
    __gm__ float2* valuesRowBasePtr = nullptr;

#pragma unroll
    for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
        int32_t globalVecIdx = blockBase + i * blockThreadNum + threadIdx;
        
        if (globalVecIdx >= inVecLength) {
            break;
        }

        int rowIdx = isPowerOfTwo ? (globalVecIdx >> gradDimVecShift) : (globalVecIdx / gradDimVec);
        int colVecIdx = isPowerOfTwo ? (globalVecIdx & (gradDimVec - 1)) : (globalVecIdx % gradDimVec);

        if (rowIdx != lastRowIdx) {
            valuesRowBasePtr = reinterpret_cast<__gm__ float2*>(valuesPtr[rowIdx]);
            lastRowIdx = rowIdx;
        }

        float2 tmpGrad = grads[globalVecIdx];
        int weightIdx = colVecIdx;
        int mIdx = weightIdx + gradDimVec;
        int vIdx = mIdx + gradDimVec;

        float2 tmpWeight = valuesRowBasePtr[weightIdx];
        float2 tmpM = valuesRowBasePtr[mIdx];
        float2 tmpV = valuesRowBasePtr[vIdx];
    
        float2 effGrad, newM, newV, vHat, res_w;

        effGrad.x = tmpGrad.x + weightDecay * tmpWeight.x;
        effGrad.y = tmpGrad.y + weightDecay * tmpWeight.y;

        newM.x = beta1 * tmpM.x + oneMinusBeta1 * effGrad.x;
        newM.y = beta1 * tmpM.y + oneMinusBeta1 * effGrad.y;

        newV.x = beta2 * tmpV.x + oneMinusBeta2 * effGrad.x * effGrad.x;
        newV.y = beta2 * tmpV.y + oneMinusBeta2 * effGrad.y * effGrad.y;
        
        vHat.x = newV.x * invVHatDenom;
        vHat.y = newV.y * invVHatDenom;
        
        res_w.x = tmpWeight.x - stepSize * newM.x / (AscendC::Simt::Sqrt(vHat.x) + eps);
        res_w.y = tmpWeight.y - stepSize * newM.y / (AscendC::Simt::Sqrt(vHat.y) + eps);

        valuesRowBasePtr[weightIdx] = res_w;
        valuesRowBasePtr[mIdx] = newM;
        valuesRowBasePtr[vIdx] = newV;
    }
}

template <bool isPowerOfTwo>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void SimtLargeDataCompute(
    __gm__ float2* grads, __gm__ float2* __gm__* valuesPtr, uint32_t gradDimVec, int32_t inVecLength,
    float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2, float stepSize,
    float invVHatDenom, float weightDecay, float eps, int32_t totalBlocks,
    int32_t blockStartIdx, int32_t curBlocksCount, int32_t gradDimVecShift)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockThreadNum = AscendC::Simt::GetThreadNum<0>();
    int32_t blockElementCapacity = blockThreadNum * MAX_ELEMENTS_PER_THREAD;

    for (int32_t iter = 0; iter < curBlocksCount; ++iter) {
        int32_t globalBlockIdx = blockStartIdx + iter;
        int32_t blockBase = globalBlockIdx * blockElementCapacity;
        
        if (blockBase >= inVecLength) {
            break;
        }

        int32_t lastRowIdx = -1;
        __gm__ float2* valuesRowBasePtr = nullptr;

#pragma unroll
        for (int32_t i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
            int32_t globalVecIdx = blockBase + i * blockThreadNum + threadIdx;
            
            if (globalVecIdx >= inVecLength) {
                break;
            }

            int rowIdx = isPowerOfTwo ? (globalVecIdx >> gradDimVecShift) : (globalVecIdx / gradDimVec);
            int colVecIdx = isPowerOfTwo ? (globalVecIdx & (gradDimVec - 1)) : (globalVecIdx % gradDimVec);

            if (rowIdx != lastRowIdx) {
                valuesRowBasePtr = reinterpret_cast<__gm__ float2*>(valuesPtr[rowIdx]);
                lastRowIdx = rowIdx;
            }

            float2 tmpGrad = grads[globalVecIdx];
            int weightIdx = colVecIdx;
            int mIdx = weightIdx + gradDimVec;
            int vIdx = mIdx + gradDimVec;

            float2 tmpWeight = valuesRowBasePtr[weightIdx];
            float2 tmpM = valuesRowBasePtr[mIdx];
            float2 tmpV = valuesRowBasePtr[vIdx];

            float2 effGrad, newM, newV, vHat, res_w;

            effGrad.x = tmpGrad.x + weightDecay * tmpWeight.x;
            effGrad.y = tmpGrad.y + weightDecay * tmpWeight.y;

            newM.x = beta1 * tmpM.x + oneMinusBeta1 * effGrad.x;
            newM.y = beta1 * tmpM.y + oneMinusBeta1 * effGrad.y;

            newV.x = beta2 * tmpV.x + oneMinusBeta2 * effGrad.x * effGrad.x;
            newV.y = beta2 * tmpV.y + oneMinusBeta2 * effGrad.y * effGrad.y;
            
            vHat.x = newV.x * invVHatDenom;
            vHat.y = newV.y * invVHatDenom;
            
            res_w.x = tmpWeight.x - stepSize * newM.x / (AscendC::Simt::Sqrt(vHat.x) + eps);
            res_w.y = tmpWeight.y - stepSize * newM.y / (AscendC::Simt::Sqrt(vHat.y) + eps);

            valuesRowBasePtr[weightIdx] = res_w;
            valuesRowBasePtr[mIdx] = newM;
            valuesRowBasePtr[vIdx] = newV;
        }
    }
}
} // namespace AdamUpdateFloat2Simt