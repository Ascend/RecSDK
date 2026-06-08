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

#include "../ops_utils.h"
#include "kernel_operator.h"
#include "pooling_embeddings_simd_kernel.h"

extern "C" __global__ __aicore__ void pooling_embeddings_simd(GM_ADDR src, GM_ADDR dst, GM_ADDR offset, GM_ADDR inverse,
                                                              GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const __gm__ PoolingEmbeddingsSimdTilingData* tilingData =
        reinterpret_cast<const __gm__ PoolingEmbeddingsSimdTilingData*>(tiling);

    dyn_emb::DataType srcDtype = static_cast<dyn_emb::DataType>(tilingData->srcType);
    dyn_emb::DataType dstDtype = static_cast<dyn_emb::DataType>(tilingData->dstType);
    dyn_emb::DataType offsetDtype = static_cast<dyn_emb::DataType>(tilingData->offsetType);

    FLOAT_TYPE_DISPATCH(srcDtype, src_t, {
        FLOAT_TYPE_DISPATCH(dstDtype, dst_t, {
            INT_TYPE_DISPATCH(offsetDtype, offset_t, {
                dyn_emb_pooling_embeddings_simd::PoolingEmbeddingsSimd<offset_t, src_t, dst_t> op(&pipe);
                op.Init(src, dst, offset, inverse, tilingData);
                op.Process();
            });
        });
    });
}
