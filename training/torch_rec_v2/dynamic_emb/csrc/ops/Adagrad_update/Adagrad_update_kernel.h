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

struct AdaGradOptimizer {
    __simt_callee__ inline void updatefloat2(__gm__ float2* valuesRowBasePtr, int32_t colVecIdx, uint32_t gradDim,
        float2 grad, float, float, float, float, float stepSize, float, float, float eps) const
    {
        const int32_t weightIdx = colVecIdx;
        const int32_t gIdx = weightIdx + static_cast<int32_t>(gradDim);

        float2 tmpWeight = valuesRowBasePtr[weightIdx];
        float2 tmpG = valuesRowBasePtr[gIdx];

        float2 newG;
        newG.x = tmpG.x + grad.x * grad.x;
        newG.y = tmpG.y + grad.y * grad.y;

        const float denomX = AscendC::Simt::Sqrt(newG.x) + eps;
        const float denomY = AscendC::Simt::Sqrt(newG.y) + eps;

        float2 resW;
        resW.x = tmpWeight.x - stepSize * grad.x / denomX;
        resW.y = tmpWeight.y - stepSize * grad.y / denomY;

        valuesRowBasePtr[weightIdx] = resW;
        valuesRowBasePtr[gIdx] = newG;
    }

    template <typename grad_t, typename weight_t>
    __simt_callee__ inline void update(__gm__ weight_t* valuesRowBasePtr, int32_t colIdx, uint32_t gradDim, grad_t grad,
        float, float, float, float, float stepSize, float, float, float eps) const
    {
        const int32_t weightIdx = colIdx;
        const int32_t gIdx = weightIdx + static_cast<int32_t>(gradDim);
        weight_t tmpWeight = valuesRowBasePtr[weightIdx];
        weight_t tmpG = valuesRowBasePtr[gIdx];

        const float gradF = static_cast<float>(grad);
        const float gAcc = static_cast<float>(tmpG) + gradF * gradF;
        const float wF = static_cast<float>(tmpWeight);
        const float denom = AscendC::Simt::Sqrt(gAcc) + eps;
        const float resW = wF - stepSize * gradF / denom;
        valuesRowBasePtr[weightIdx] = static_cast<weight_t>(resW);
        valuesRowBasePtr[gIdx] = static_cast<weight_t>(gAcc);
    }
};
