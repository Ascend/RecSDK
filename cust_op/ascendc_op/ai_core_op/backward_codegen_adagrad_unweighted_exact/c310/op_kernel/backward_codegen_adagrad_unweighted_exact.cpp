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

#include "backward_codegen_adagrad_unweighted_exact_kernel.h"
#include "backward_codegen_adam_unweighted_exact_kernel.h"
#include "backward_codegen_sgd_unweighted_exact_kernel.h"
#include "backward_codegen_adagrad_unweighted_exact_kernel_unique.h"
#include "backward_codegen_adam_unweighted_exact_kernel_unique.h"
#include "backward_codegen_sgd_unweighted_exact_kernel_unique.h"
#include "backward_codegen_rowwise_adagrad_unweighted_exact_kernel.h"
#include "kernel_operator.h"


extern "C" __global__ __aicore__ void backward_codegen_adagrad_unweighted_exact(GM_ADDR gradOutput,
                                                                                GM_ADDR devWeights,
                                                                                GM_ADDR uvmWeights,
                                                                                GM_ADDR lxuCacheWeights,
                                                                                GM_ADDR weightsPlacements,
                                                                                GM_ADDR weightsOffsets,
                                                                                GM_ADDR dOffsets,
                                                                                GM_ADDR hashSizeCumsum,
                                                                                GM_ADDR indices, GM_ADDR offsets,
                                                                                GM_ADDR lxuCacheLocations,
                                                                                GM_ADDR momentum1Dev,
                                                                                GM_ADDR momentum1Uvm,
                                                                                GM_ADDR momentum1Placements,
                                                                                GM_ADDR momentum1Offsets,
                                                                                GM_ADDR momentum2Dev,
                                                                                GM_ADDR momentum2Uvm,
                                                                                GM_ADDR momentum2Placements,
                                                                                GM_ADDR momentum2Offsets,
                                                                                GM_ADDR hashIndices, GM_ADDR uniqueId,
                                                                                GM_ADDR uniqueHashSize,
                                                                                GM_ADDR uniqueInverse,
                                                                                GM_ADDR indiceSizeCumsum,
                                                                                GM_ADDR out,
                                                                                GM_ADDR momentum1DevOut,
                                                                                GM_ADDR momentum2DevOut,
                                                                                GM_ADDR weightsDevOut,
                                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    BackwardCodegenUnweightedExact::Args args{
        gradOutput, devWeights,      weightsPlacements, weightsOffsets, dOffsets,  hashSizeCumsum, indices,
        offsets,    momentum1Dev,    momentum2Dev,      hashIndices,    uniqueId,  uniqueHashSize, uniqueInverse,
        indiceSizeCumsum, out,       momentum1DevOut, momentum2DevOut,   weightsDevOut,  workspace, tiling};
    if (TILING_KEY_IS(1)) {  // NORMAL_ADAGRAD
        BackwardCodegenAdagradUnweightedExact::BackwardCodegenAdagradUnweightedExactKernel kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.UpdateEmbedAda();
        } else {
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(2)) {  // NORMAL_ADAM
        BackwardCodegenAdamUnweightedExact::BackwardCodegenAdamUnweightedExactKernel kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.UpdateEmbedAdam(args);
        } else {
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(3)) {  // NORMAL_SGD
        BackwardCodegenSgdUnweightedExact::BackwardCodegenSgdUnweightedExactKernel kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.UpdateEmbedSgd(args);
        } else {
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(4)) {
        BackwardCodegenUnweightedExactAdagradUnique::BackwardCodegenAdagradUnweightedExactKernelUnique kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.AdagradScheduler();
        } else {
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(5)) {
        BackwardCodegenUnweightedAdamExactUnique::BackwardCodegenAdamUnweightedExactKernelUnique kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.AdamScheduler();
        } else {
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(6)) {
        BackwardCodegenUnweightedSgdExactUnique::BackwardCodegenSgdUnweightedExactKernelUnique kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.SgdScheduler();
        } else {
            kernel.Compute(args);
        }
    } else if (TILING_KEY_IS(7)) {
        BackwardCodegenRowwiseAdagradUnweightedExact::BackwardCodegenRowwiseAdagradUnweightedExactKernel kernel;
        if (tiling_data.useOptimize) {
            kernel.Compute(args);
            kernel.UpdateEmbedRowwiseAda();
        } else {
            kernel.Compute(args);
        }
    }
}