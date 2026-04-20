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
#include "lengths_index_kernel.h"

// Kernel entry point
extern "C" __global__ __aicore__ void lengths_index(GM_ADDR offsets, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    LengthsIndex::Args args{offsets, output, workspace, tiling};

    LengthsIndex::LengthsIndexKernel<DTYPE_OFFSETS> kernel(args);
    kernel.Compute();
}