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
#include "expand_into_jagged_permute.h"

extern "C" __global__ __aicore__ void expand_into_jagged_permute(
    GM_ADDR permute,
    GM_ADDR input_offsets,
    GM_ADDR output_offsets,
    GM_ADDR output_permute,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    Args args{
        permute, input_offsets, output_offsets,
        output_permute, workspace, tiling
    };

    ExpandIntoJaggedPermute::ExpandIntoJaggedPermuteKernel<DTYPE_PERMUTE> kernel(args);
    kernel.Compute(args);
}
