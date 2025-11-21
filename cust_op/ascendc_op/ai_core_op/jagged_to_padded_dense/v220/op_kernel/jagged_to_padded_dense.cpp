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

#include "jagged_to_padded_dense_kernel.h"
#include "kernel_operator.h"
#include "utils.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void jagged_to_padded_dense(GM_ADDR values, GM_ADDR offsets, GM_ADDR out,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif

    JaggedToPaddedDense::Args args{values, offsets, out, workspace, tiling};
    JaggedToPaddedDense::JaggedToPaddedDenseKernel<DTYPE_VALUES, DTYPE_OFFSETS> kernel(args);
    kernel.Compute();
}