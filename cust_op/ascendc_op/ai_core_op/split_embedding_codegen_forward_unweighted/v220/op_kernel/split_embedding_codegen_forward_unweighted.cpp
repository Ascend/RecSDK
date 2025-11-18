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
#include "common.h"
#include "split_embedding_kernel.h"
#include "split_embedding_nobag_kernel.h"
#include "split_embedding_codegen_forward_unweighted_kernel.h"

extern "C" __global__ __aicore__ void split_embedding_codegen_forward_unweighted(
    GM_ADDR devWeights, GM_ADDR uvmWeights, GM_ADDR lxuCacheWeights, GM_ADDR weightsPlacements, GM_ADDR weightsOffsets,
    GM_ADDR dOffsets, GM_ADDR indices, GM_ADDR offsets, GM_ADDR lxuCacheLocations, GM_ADDR hashIndices,
    GM_ADDR indiceSizeCumsum, GM_ADDR rowsPerTable, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    Args args{
        devWeights, weightsPlacements, weightsOffsets, dOffsets, indices, offsets, hashIndices, indiceSizeCumsum,
        rowsPerTable, out, tiling, workspace};
    if (TILING_KEY_IS(0)) { // SUM
        SplitEmbeddingCodegenForwardUnweighted::SplitEmbeddingKernel<PoolingMode::SUM> kernel(args, &pipe);
        kernel.Compute();
    } else if (TILING_KEY_IS(1)) { // MEAN
        SplitEmbeddingCodegenForwardUnweighted::SplitEmbeddingKernel<PoolingMode::MEAN> kernel(args, &pipe);
        kernel.Compute();
    } else if (TILING_KEY_IS(2)) { // NONE
        SplitEmbeddingCodegenForwardUnweighted::SplitEmbeddingNobagKernel kernel(args, &pipe);
        kernel.Compute();
    } else if (TILING_KEY_IS(3)) {
        SplitEmbeddingCodegenForwardUnweighted::SplitEmbeddingCodegenForwardUnweightedKernel kernel(args, &pipe);
        kernel.Compute();
    }
}