/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

constexpr int32_t BLOCK_THREADS = GatherDim0Simt::MAX_THREADS_PER_BLOCK;
constexpr int32_t ELEMS_PER_THREAD = GatherDim0Simt::MAX_ELEMENTS_PER_THREAD;
constexpr int32_t CACHE_ALIGN = GatherDim0Simt::CACHE_ALIGN;
constexpr int32_t DATA_ALIGN_BYTES = 32;

#define DTYPE_DISPATCH(isInt32, DTYPE, ...) \
    do {                                     \
        if (isInt32) {                       \
            using DTYPE = int32_t;           \
            __VA_ARGS__;                     \
        } else {                             \
            using DTYPE = int64_t;           \
            __VA_ARGS__;                     \
        }                                    \
    } while (0)

extern "C" __global__ __aicore__ void gather_dim0(GM_ADDR input_data, GM_ADDR row_indices, GM_ADDR output_data,
                                                  int32_t inLength, int32_t dataDim, int32_t indicesLength,
                                                  int32_t outLength, int32_t totalBlocks, int32_t blocksPerCore,
                                                  int32_t remainderBlocks, bool isInt32, bool isSmall)
{
    int32_t coreId = AscendC::GetBlockIdx();

    DTYPE_DISPATCH(isInt32, DTYPE_X, {
        __gm__ float* input = reinterpret_cast<__gm__ float*>(input_data);
        __gm__ DTYPE_X* indices = reinterpret_cast<__gm__ DTYPE_X*>(row_indices);
        __gm__ float* output = reinterpret_cast<__gm__ float*>(output_data);
        if (isSmall) {
            AscendC::Simt::VF_CALL<GatherDim0Simt::SimtSmallDataCompute<DTYPE_X>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, input, indices, output, dataDim, inLength, outLength);

        } else {
            int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
            int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

            AscendC::Simt::VF_CALL<GatherDim0Simt::SimtLargeDataCompute<DTYPE_X>>(
                AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, input, indices, output, dataDim, inLength, outLength,
                totalBlocks, blockStartIdx, curBlocksCount);
        }
    });
}
