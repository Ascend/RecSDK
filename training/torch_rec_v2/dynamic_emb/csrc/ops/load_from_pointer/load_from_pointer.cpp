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
#include "kernel_operator.h"
#include "load_from_pointer_kernel.h"

template <typename T>
__aicore__ void load_imp(GM_ADDR inData, GM_ADDR outData,
    uint32_t dim, uint64_t total, uint32_t blkPerCore, uint32_t remainderBlk, uint32_t threads)
{
    uint32_t coreId = AscendC::GetBlockIdx();
    T curBlk = (coreId < remainderBlk) ? (blkPerCore + 1) : blkPerCore;
    T startBlk = coreId * blkPerCore + ((coreId < remainderBlk) ? coreId : remainderBlk);
    T startIdx = startBlk * threads;
    T endIdx = (startBlk + curBlk) * threads;
    T endUnrollIdx = 0;
    T curNum = 0;
    T unrollNum = (threads << LoadFromPointerSimt::UNROLL_SHIFT);

    if (endIdx < total) {
        endIdx = startIdx + curBlk * threads;
    } else {
        endIdx = total;
    }
    curNum = endIdx - startIdx;
    endUnrollIdx = startIdx + curNum / unrollNum * unrollNum;

    if (dim % 2 == 0) {
        const __gm__ float2* __gm__* input = reinterpret_cast<const __gm__ float2* __gm__*>(inData);
        __gm__ float2* output = reinterpret_cast<__gm__ float2*>(outData);

        AscendC::Simt::VF_CALL<LoadFromPointerSimt::LoadFromPointerCompute<float2, T>>(
            AscendC::Simt::Dim3{threads, 1, 1}, input, output, dim >> 1,
            startIdx, endIdx, endUnrollIdx);
    } else {
        const __gm__ float* __gm__* input = reinterpret_cast<const __gm__ float* __gm__*>(inData);
        __gm__ float* output = reinterpret_cast<__gm__ float*>(outData);

        AscendC::Simt::VF_CALL<LoadFromPointerSimt::LoadFromPointerCompute<float, T>>(
            AscendC::Simt::Dim3{threads, 1, 1}, input, output, dim,
            startIdx, endIdx, endUnrollIdx);
    }
}

extern "C" __global__ __aicore__ void load_from_pointer(GM_ADDR inData, GM_ADDR outData,
    uint32_t dim, uint64_t total, uint32_t blkPerCore, uint32_t remainderBlk, uint32_t threads, uint32_t small)
{
    if (small == 1) {
        load_imp<uint32_t>(inData, outData, dim, total, blkPerCore, remainderBlk, threads);
    } else {
        load_imp<uint64_t>(inData, outData, dim, total, blkPerCore, remainderBlk, threads);
    }
}