/* Copyright 2026. Huawei Technologies Co., Ltd. All rights reserved.

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

#include <simt_api/common_functions.h>
#include "kernel_operator.h"
#include "lookup_backward_v2_simt.h"

using namespace AscendC;

template <typename IndexT, typename ValueT, bool IsMean, bool IsFloat2>
__global__ __vector__ void lookup_backward_v2_kernel(__gm__ ValueT* grad, __gm__ ValueT* uniqueBuffer,
                                                     __gm__ IndexT* inverseIndices, __gm__ IndexT* biasedOffsets,
                                                     int32_t launchDim, int32_t numSlots, int32_t totalBlocks,
                                                     int32_t blocksPerCore, int32_t remainderBlocks, bool isSmall)
{
    const int32_t coreId = GetBlockIdx();
    LookupBackwardV2Simt::LaunchBackwardCompute<IndexT, ValueT, IsMean, IsFloat2>(
        grad, uniqueBuffer, inverseIndices, biasedOffsets, launchDim, numSlots, totalBlocks, blocksPerCore,
        remainderBlocks, isSmall, coreId);
}
