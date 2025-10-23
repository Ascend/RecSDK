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
#include "sgd_kernel.h"

extern "C" __global__ __aicore__ void sgd(GM_ADDR gradient, GM_ADDR indices, GM_ADDR inputVar, GM_ADDR learningRate,
                                          GM_ADDR inputVar_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

#ifdef KERNEL_TASK_TYPE_DEFAULT
    // Set kernel type with new versions of CANN to avoid matmul error during compiling.
    // In previous versions of CANN, avoid matmul error by using '#ifndef __GET_CODE_CHANNEL__'.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif

    if (TILING_KEY_IS(0)) {
        SgdKernel<DTYPE_GRADIENT, DTYPE_INDICES, false> kernel;
        kernel.Init(gradient, indices, inputVar, learningRate, inputVar_ref, tiling_data);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        SgdKernel<DTYPE_GRADIENT, DTYPE_INDICES, true> kernel;
        kernel.Init(gradient, indices, inputVar, learningRate, inputVar_ref, tiling_data);
        kernel.Process();
    }
}