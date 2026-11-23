/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#include "kernel_operator.h"
#include "gen_position_ids_reverse_v2_kernel.h"

extern "C" __global__ __aicore__ void gen_position_ids_reverse_v2(GM_ADDR seqlen, GM_ADDR seqlenOffsets,
                                                                  GM_ADDR rspos, GM_ADDR positionIds,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GenPositionIdsReverseV2::Args args{seqlen, seqlenOffsets, rspos, positionIds, workspace, tiling};

    GenPositionIdsReverseV2::GenPositionIdsReverseV2Kernel kernel(args);
    kernel.Compute();
}