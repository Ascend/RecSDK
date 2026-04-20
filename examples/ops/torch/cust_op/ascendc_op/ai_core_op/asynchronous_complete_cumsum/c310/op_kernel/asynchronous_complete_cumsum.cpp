/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

#include "asynchronous_complete_cumsum_kernel.h"

// Kernel入口函数
extern "C" __global__ __aicore__ void asynchronous_complete_cumsum(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    Args args{
        x, y, workspace, tiling
    };

    AsynchronousCompleteCumsum::AsynchronousCompleteCumsumKernel<DTYPE_X> kernel(args);
    kernel.Compute();
}