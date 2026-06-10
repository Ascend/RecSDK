/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

struct SGDOptimizer {
    static constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
    static constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
    __simt_callee__ inline void updatefloat2(__gm__ float2* valuesRowBasePtr, int32_t colVecIdx, uint32_t gradDim,
                                             float2 grad, float beta1, float beta2, float oneMinusBeta1,
                                             float oneMinusBeta2, float stepSize, float invVHatDenom, float decayFactor,
                                             float eps) const
    {
        float2 tmpWeight = valuesRowBasePtr[colVecIdx];

        float2 res_w;

        res_w.x = tmpWeight.x - grad.x * decayFactor;
        res_w.y = tmpWeight.y - grad.y * decayFactor;

        valuesRowBasePtr[colVecIdx] = res_w;
    }

    template <typename grad_t, typename weight_t>
    __simt_callee__ inline void update(__gm__ weight_t* valuesRowBasePtr, int32_t colIdx, uint32_t gradDim, grad_t grad,
                                       float beta1, float beta2, float oneMinusBeta1, float oneMinusBeta2,
                                       float stepSize, float invVHatDenom, float decayFactor, float eps) const
    {
        float tmpWeight = valuesRowBasePtr[colIdx];

        float res_w = tmpWeight - grad * decayFactor;

        valuesRowBasePtr[colIdx] = res_w;
    }
};
