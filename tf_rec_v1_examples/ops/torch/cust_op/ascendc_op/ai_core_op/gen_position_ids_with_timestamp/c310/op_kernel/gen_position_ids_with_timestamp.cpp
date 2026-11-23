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
#include "gen_position_ids_with_timestamp_kernel.h"

// Kernel entry point
extern "C" __global__ __aicore__ void gen_position_ids_with_timestamp(GM_ADDR seqlen, GM_ADDR seqlenOffsets,
                                                                      GM_ADDR timestamps, GM_ADDR positionIds,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GenPositionIdsWithTimestamp::Args args{seqlen, seqlenOffsets, timestamps, positionIds, workspace, tiling};

    GenPositionIdsWithTimestamp::GenPositionIdsWithTimestampKernel kernel(args);
    kernel.Compute();
}