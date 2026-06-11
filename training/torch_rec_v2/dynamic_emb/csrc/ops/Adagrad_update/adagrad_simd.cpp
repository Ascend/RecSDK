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
#include "ops_utils.h"
#include "adagrad_simd_kernel.h"

extern "C" __global__ __aicore__ void adagrad_simd(GM_ADDR grads, GM_ADDR rowPtrs, GM_ADDR founds, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const __gm__ AdagradSimdTilingData* tilingData = reinterpret_cast<const __gm__ AdagradSimdTilingData*>(tiling);
    const dyn_emb::DataType gradType = static_cast<dyn_emb::DataType>(tilingData->gradType);
    const dyn_emb::DataType weightType = static_cast<dyn_emb::DataType>(tilingData->weightType);

    FLOAT_TYPE_DISPATCH(gradType, grad_t, {
        FLOAT_TYPE_DISPATCH(weightType, weight_t, {
            dyn_emb_adagrad_simd::AdagradSimd<grad_t, weight_t> op(&pipe);
            op.Init(grads, rowPtrs, founds, tilingData);
            op.Process();
        });
    });
}
