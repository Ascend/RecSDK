/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#include "invert_permute_kernel.h"
#include "kernel_operator.h"


extern "C" __global__ __aicore__ void invert_permute(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    int32_t totalLength = tilingData.xDim0;
    uint32_t actualThreadsPerBlock = tilingData.actualThreadsPerBlock;

    if (TILING_KEY_IS(0)) {
        __gm__ int64_t* dst = (__gm__ int64_t*)y;
        __gm__ int64_t* src = (__gm__ int64_t*)x;
        AscendC::Simt::VF_CALL<InvertPermute::SimtCompute<int64_t>>(
                AscendC::Simt::Dim3{actualThreadsPerBlock, 1, 1}, dst, src, totalLength, actualThreadsPerBlock);

    } else if (TILING_KEY_IS(1)) {
        __gm__ int32_t* dst = (__gm__ int32_t*)y;
        __gm__ int32_t* src = (__gm__ int32_t*)x;
        AscendC::Simt::VF_CALL<InvertPermute::SimtCompute<int32_t>>(
                AscendC::Simt::Dim3{actualThreadsPerBlock, 1, 1}, dst, src, totalLength, actualThreadsPerBlock);
    }
}