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

template <typename DATA>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) inline void LoadFromPointerCompute(
    const __gm__ DATA* __gm__* input, __gm__ DATA* output, int dim,
    int startIdx, int endIdx, int endUnrollIdx)
{
    int threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int threadNum = AscendC::Simt::GetThreadNum<0>();

    int i = startIdx + threadIdx;
    for (; i < endUnrollIdx; i += (threadNum << UNROLL_SHIFT)) {
        int i0 = i;
        int index0 = i0 / dim;
        int dimIdx0 = i0 % dim;

        int i1 = i + threadNum;
        int index1 = i1 / dim;
        int dimIdx1 = i1 % dim;

        int i2 = i + threadNum * 2;
        int index2 = i2 / dim;
        int dimIdx2 = i2 % dim;

        int i3 = i + threadNum * 3;
        int index3 = i3 / dim;
        int dimIdx3 = i3 % dim;

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
        int index = i / dim;
        int dimIdx = i % dim;

        const __gm__ DATA* pointer = input[index];
        output[i] = pointer[dimIdx];
    }
}
}