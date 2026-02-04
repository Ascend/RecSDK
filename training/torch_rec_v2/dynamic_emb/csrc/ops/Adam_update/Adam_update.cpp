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

#include "Adam_update_kernel.h"
#include "kernel_operator.h"

constexpr int32_t BLOCK_THREADS = AdamUpdateSimt::MAX_THREADS_PER_BLOCK;

extern "C" __global__ __aicore__ void Adam_update(GM_ADDR grads, GM_ADDR values, int32_t gradDim, int32_t inLength,
                                                    float lr, float beta1, float beta2, float eps, float weightDecay,
                                                    int32_t iterNum, int32_t totalBlocks, int32_t blocksPerCore,
                                                    int32_t remainderBlocks, bool isSmall)
{
    // 获取当前核心ID
    int32_t coreId = AscendC::GetBlockIdx();

    __gm__ float* gradsPtr = reinterpret_cast<__gm__ float*>(grads);
    __gm__ float* __gm__* valuesPtr = reinterpret_cast<__gm__ float* __gm__*>(values);

    if (isSmall) {
        AscendC::Simt::VF_CALL<AdamUpdateSimt::SimtSmallInBlockDataCompute>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDim, inLength, lr, beta1, beta2, eps,
            weightDecay, iterNum);
    } else {
        // 计算当前核心需要处理的块数
        int32_t curBlocksCount = (coreId < remainderBlocks) ? (blocksPerCore + 1) : blocksPerCore;
        // 计算当前核心需要处理的起始块索引
        int32_t blockStartIdx = coreId * blocksPerCore + ((coreId < remainderBlocks) ? coreId : remainderBlocks);

        AscendC::Simt::VF_CALL<AdamUpdateSimt::SimtLargeDataCompute>(
            AscendC::Simt::Dim3{BLOCK_THREADS, 1, 1}, gradsPtr, valuesPtr, gradDim, inLength, lr, beta1, beta2, eps,
            weightDecay, iterNum, totalBlocks, blockStartIdx, curBlocksCount);
    }
}
