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

namespace LoadFromPointerSimt {

constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int UNROLL_SHIFT = 2;

template <typename DATA, typename INDEX>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void LoadFromPointerCompute(
    const __gm__ DATA* __gm__* input, __gm__ DATA* output, uint32_t dim,
    INDEX startIdx, INDEX endIdx, INDEX endUnrollIdx)
{
    uint32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    uint32_t threadNum = AscendC::Simt::GetThreadNum<0>();
    INDEX i = startIdx + threadIdx;

    for (; i < endUnrollIdx; i += (threadNum << UNROLL_SHIFT)) {
        INDEX i0 = i;
        uint32_t index0 = i0 / dim;
        uint32_t dimIdx0 = i0 % dim;

        INDEX i1 = i + threadNum;
        uint32_t index1 = i1 / dim;
        uint32_t dimIdx1 = i1 % dim;

        INDEX i2 = i + threadNum * 2;
        uint32_t index2 = i2 / dim;
        uint32_t dimIdx2 = i2 % dim;

        INDEX i3 = i + threadNum * 3;
        uint32_t index3 = i3 / dim;
        uint32_t dimIdx3 = i3 % dim;

        const __gm__ DATA* pointer0 = input[index0];
        const __gm__ DATA* pointer1 = input[index1];
        const __gm__ DATA* pointer2 = input[index2];
        const __gm__ DATA* pointer3 = input[index3];

        DATA out0 = pointer0[dimIdx0];
        DATA out1 = pointer1[dimIdx1];
        DATA out2 = pointer2[dimIdx2];
        DATA out3 = pointer3[dimIdx3];

        output[i0] = out0;
        output[i1] = out1;
        output[i2] = out2;
        output[i3] = out3;
    }

    for (; i < endIdx; i += threadNum) {
        uint32_t index = i / dim;
        uint32_t dimIdx = i % dim;

        const __gm__ DATA* pointer = input[index];
        output[i] = pointer[dimIdx];
    }
}
}