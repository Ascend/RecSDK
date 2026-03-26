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

#include "kernel_operator.h"
#include "concat_jagged_tensor_grad.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void concat_jagged_tensor_grad(GM_ADDR values, GM_ADDR result,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    ConcatJaggedTensorGradKernel<DTYPE_VALUES> op;
    op.Init(values, result, workspace, &tiling_data);
    op.Process(values, result, workspace, &tiling_data);
}