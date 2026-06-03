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
#include "rowwise_adagrad_simd_kernel.h"

extern "C" __global__ __aicore__ void rowwise_adagrad_simd(GM_ADDR grads, GM_ADDR rowPtrs, GM_ADDR founds,
                                                           GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const __gm__ RowwiseAdagradSimdTilingData* tilingData =
        reinterpret_cast<const __gm__ RowwiseAdagradSimdTilingData*>(tiling);
    dyn_emb_rowwise_adagrad_simd::RowwiseAdagradSimd op(&pipe);
    op.Init(grads, rowPtrs, founds, tilingData);
    op.Process();
}
