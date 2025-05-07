/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <cstdint>
#include "kernel_operator.h"
#include "index_select_for_rank1_backward_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void index_select_for_rank1_backward(
    GM_ADDR gradY,
    GM_ADDR x,
    GM_ADDR index,
    GM_ADDR gradX,
    GM_ADDR gradIndex,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    IndexSelectForRank1BackwardKernel op(tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    op.Init(gradY, index, gradX);
    op.Process();
}
