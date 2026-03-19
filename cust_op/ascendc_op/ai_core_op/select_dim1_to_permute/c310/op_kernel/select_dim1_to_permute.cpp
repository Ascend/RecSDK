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

#include "kernel_operator.h"
#include "select_dim1_to_permute.h"

extern "C" __global__ __aicore__ void select_dim1_to_permute(GM_ADDR indices, GM_ADDR lengths, GM_ADDR permute,
                                                             GM_ADDR outputLengths, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    SelectDim1ToPermute::Args args{indices, lengths, permute, outputLengths, workspace, tiling};

    SelectDim1ToPermute::SelectDim1ToPermuteKernel<DTYPE_INDICES, DTYPE_LENGTHS> kernel(args, &pipe);
    kernel.Process(args);
}