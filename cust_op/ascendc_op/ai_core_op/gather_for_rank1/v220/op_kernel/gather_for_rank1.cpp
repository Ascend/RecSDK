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

#include "gather_for_rank1_kernel.h"
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void gather_for_rank1(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    GatherForRank1::Args args{x, index, y, workspace, tiling};

    if (TILING_KEY_IS(0)) {
        GatherForRank1::GatherForRank1Kernel<float> kernel(args);
        kernel.Compute();
    } else if (TILING_KEY_IS(1)) {
        GatherForRank1::GatherForRank1Kernel<half> kernel(args);
        kernel.Compute();
    }
}