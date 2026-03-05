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
#include "gather_dim0_kernel.h"
#include "kernel_operator.h"
#include "../ops_utils.h"

extern "C" __global__ __aicore__ void gather_dim0(GM_ADDR inData, GM_ADDR indicesData, GM_ADDR outData,
    uint32_t dim, uint64_t total, uint32_t blkPerCore, uint32_t remainderBlk, uint32_t threads,
    uint32_t iType, uint32_t eleSz)
{
    uint32_t coreId = AscendC::GetBlockIdx();
    uint32_t curBlk = (coreId < remainderBlk) ? (blkPerCore + 1) : blkPerCore;
    uint32_t startBlk = coreId * blkPerCore + ((coreId < remainderBlk) ? coreId : remainderBlk);

    uint32_t startIdx = startBlk * threads;
    uint32_t endIdx = (startBlk + curBlk) * threads;
    uint32_t endUnrollIdx = 0;
    uint32_t curNum = 0;
    uint32_t unrollNum = (threads << GatherDimSimt::UNROLL_SHIFT);

    if (endIdx < total) {
        endIdx = startIdx + curBlk * threads;
    } else {
        endIdx = total;
    }
    curNum = endIdx - startIdx;
    endUnrollIdx = startIdx + curNum / unrollNum * unrollNum;

    if (dim % 4 == 0) {
        const __gm__ float2* input = reinterpret_cast<const __gm__ float2*>(inData);
        const __gm__ uint64_t* indices = reinterpret_cast<const __gm__ uint64_t*>(indicesData);
        __gm__ float2* output = reinterpret_cast<__gm__ float2*>(outData);

        if (eleSz == 4) {
            AscendC::Simt::VF_CALL<GatherDimSimt::GatherDim<float2>>(
                AscendC::Simt::Dim3{threads, 1, 1}, input, indices, output, dim >> 1,
                startIdx, endIdx, endUnrollIdx);
        } else {
            AscendC::Simt::VF_CALL<GatherDimSimt::GatherDim<float2>>(
                AscendC::Simt::Dim3{threads, 1, 1}, input, indices, output, dim >> 2,
                startIdx, endIdx, endUnrollIdx);
        }
    } else {
        dyn_emb::DataType inType = static_cast<dyn_emb::DataType>(iType);

        FLOAT_TYPE_DISPATCH(inType, DataType, {
            const __gm__ DataType* input = reinterpret_cast<const __gm__ DataType*>(inData);
            const __gm__ uint64_t* indices = reinterpret_cast<const __gm__ uint64_t*>(indicesData);
            __gm__ DataType* output = reinterpret_cast<__gm__ DataType*>(outData);

            AscendC::Simt::VF_CALL<GatherDimSimt::GatherDim<DataType>>(
                AscendC::Simt::Dim3{threads, 1, 1}, input, indices, output, dim,
                startIdx, endIdx, endUnrollIdx);
        });
    }
}

