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
#include "kernel_operator.h"

#include "load_from_pointer_kernel.h"

constexpr int32_t BLOCK_THREADS = LoadFromPointerSimt::MAX_THREADS_PER_BLOCK;

extern "C" __global__ __aicore__ void load_from_pointer(GM_ADDR input_data, GM_ADDR output_data, int64_t inLength,
                                                        int32_t dataDim, int64_t outLength, int64_t totalBlocks,
                                                        int64_t blocksPerCore, int32_t remainderBlocks, bool isSmall)
{
    int32_t coreId = AscendC::GetBlockIdx();

    __gm__ float* __gm__* input = reinterpret_cast<__gm__ float* __gm__*>(input_data);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(output_data);
    if (isSmall) {
        AscendC::Simt::VF_CALL<LoadFromPointerSimt::SmallDataLoadFromPointerCompute>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, input, output, dataDim, inLength, outLength);

    } else {
        int64_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        int64_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

        AscendC::Simt::VF_CALL<LoadFromPointerSimt::LargeDataLoadFromPointerCompute>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, input, output, dataDim, inLength, outLength, totalBlocks,
            blockStartIdx, curBlocksCount);
    }
}