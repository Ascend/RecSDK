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

struct AdamWOptimizer {
    static constexpr int32_t MAX_THREADS_PER_BLOCK  = 1024;
    static constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
    __aicore__ inline void updatefloat2(__gm__ float2* valuesRowBasePtr,
        int32_t colVecIdx,
        uint32_t gradDim,
        float2 grad,
        float beta1,
        float beta2,
        float oneMinusBeta1,
        float oneMinusBeta2,
        float stepSize,
        float invVHatDenom,
        float decayFactor,
        float eps) const
    {
        int32_t weightIdx = colVecIdx;
        int32_t mIdx = weightIdx + static_cast<int32_t>(gradDim);
        int32_t vIdx = mIdx + static_cast<int32_t>(gradDim);

        float2 tmpWeight = valuesRowBasePtr[weightIdx];
        float2 tmpM      = valuesRowBasePtr[mIdx];
        float2 tmpV      = valuesRowBasePtr[vIdx];

        float2 newM, newV, vHat, res_w;

        newM.x = beta1 * tmpM.x + oneMinusBeta1 * grad.x;
        newM.y = beta1 * tmpM.y + oneMinusBeta1 * grad.y;

        newV.x = beta2 * tmpV.x + oneMinusBeta2 * grad.x * grad.x;
        newV.y = beta2 * tmpV.y + oneMinusBeta2 * grad.y * grad.y;

        // 使用 Host 侧传来的乘法因子代替除法
        vHat.x = newV.x * invVHatDenom;
        vHat.y = newV.y * invVHatDenom;

        res_w.x = tmpWeight.x * decayFactor
        - stepSize * newM.x / (AscendC::Simt::Sqrt(vHat.x + eps) );
        res_w.y = tmpWeight.y * decayFactor
        - stepSize * newM.y / (AscendC::Simt::Sqrt(vHat.y + eps) );

        valuesRowBasePtr[weightIdx] = res_w;
        valuesRowBasePtr[mIdx]      = newM;
        valuesRowBasePtr[vIdx]      = newV;     
    }
    template <typename grad_t, typename weight_t>
    __aicore__ inline void update(__gm__ weight_t* valuesRowBasePtr,
                                  int32_t colIdx,
                                  uint32_t gradDim,
                                  grad_t grad,
                                  float beta1,
                                  float beta2,
                                  float oneMinusBeta1,
                                  float oneMinusBeta2,
                                  float stepSize,
                                  float invVHatDenom,
                                  float decayFactor,
                                  float eps) const
    {
        int32_t weightIdx = colIdx;
        int32_t mIdx = weightIdx + static_cast<int32_t>(gradDim);
        int32_t vIdx = mIdx + static_cast<int32_t>(gradDim);

        weight_t tmpWeight = valuesRowBasePtr[weightIdx];
        weight_t tmpM = valuesRowBasePtr[mIdx];
        weight_t tmpV = valuesRowBasePtr[vIdx];

        grad_t newM = beta1 * tmpM + oneMinusBeta1 * grad;
        grad_t newV = beta2 * tmpV + oneMinusBeta2 * grad * grad;

        float vHat = newV * invVHatDenom;
        float resW = tmpWeight * decayFactor
                    - stepSize * newM / (AscendC::Simt::Sqrt(vHat + eps) );

        valuesRowBasePtr[weightIdx] = resW;
        valuesRowBasePtr[mIdx]      = newM;
        valuesRowBasePtr[vIdx]      = newV;
    }
};