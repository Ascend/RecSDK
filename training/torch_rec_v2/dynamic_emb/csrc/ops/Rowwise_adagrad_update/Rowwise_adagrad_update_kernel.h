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

struct RowWiseAdaGradOptimizer {
    static constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
    static constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
    __simt_callee__ inline void updatefloat2(__gm__ float2* valuesRowBasePtr, int32_t colVecIdx, uint32_t gradDim,
        float2 grad, float, float, float, float, float stepSize, float, float, float eps) const
    {
        const int32_t weightIdx = colVecIdx;
        const int32_t rowAccumIdx = static_cast<int32_t>(gradDim);

        float2 tmpWeight = valuesRowBasePtr[weightIdx];
        const float delta = grad.x * grad.x + grad.y * grad.y;
        __gm__ float* rowAccumPtr = reinterpret_cast<__gm__ float*>(valuesRowBasePtr + rowAccumIdx);
        float rowAccum = AscendC::Simt::AtomicAdd<float>(rowAccumPtr, delta) + delta;

        const float denom = AscendC::Simt::Sqrt(rowAccum) + eps;
        float2 resW;
        resW.x = tmpWeight.x - stepSize * grad.x / denom;
        resW.y = tmpWeight.y - stepSize * grad.y / denom;

        valuesRowBasePtr[weightIdx] = resW;
    }

    template <typename grad_t, typename weight_t>
    __simt_callee__ inline void update(__gm__ weight_t* valuesRowBasePtr, int32_t colIdx, uint32_t gradDim, grad_t grad,
        float, float, float, float, float stepSize, float, float, float eps) const
    {
        const int32_t weightIdx = colIdx;
        const int32_t rowAccumIdx = static_cast<int32_t>(gradDim);

        weight_t tmpWeight = valuesRowBasePtr[weightIdx];
        const float gradF = static_cast<float>(grad);
        const weight_t delta = static_cast<weight_t>(gradF * gradF);
        __gm__ weight_t* rowAccumPtr = valuesRowBasePtr + rowAccumIdx;
        float rowAccum = static_cast<float>(AscendC::Simt::AtomicAdd<weight_t>(rowAccumPtr, delta));
        rowAccum += static_cast<float>(delta);
        const float denom = AscendC::Simt::Sqrt(rowAccum) + eps;
        const float resW = static_cast<float>(tmpWeight) - stepSize * gradF / denom;

        valuesRowBasePtr[weightIdx] = static_cast<weight_t>(resW);
    }
};
