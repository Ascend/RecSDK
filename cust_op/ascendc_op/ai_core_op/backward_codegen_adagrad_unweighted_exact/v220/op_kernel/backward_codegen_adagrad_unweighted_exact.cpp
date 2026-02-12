/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

#include "backward_codegen_unweighted_exact_kernel.h"
#include "backward_codegen_unweighted_exact_kernel_unique.h"
#include "structure/args_struct.h"
#include "optimizers/adagrad_optimizer.h"
#include "optimizers/adam_optimizer.h"
#include "optimizers/sgd_optimizer.h"
#include "optimizers/rowwise_adagrad_optimizer.h"
#include "structure/optimizer_layout.h"

using namespace BackwardCodegenUnweightedExact;

// normal版本的类型别名
using AdagradKernel = BackwardCodegenUnweightedExactKernel<
    MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_PER_DIM, Optimizers::AdagradOptimizer>;
using AdamKernel = BackwardCodegenUnweightedExactKernel<
    MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_MOMENTUM2, Optimizers::AdamOptimizer>;
using SgdKernel = BackwardCodegenUnweightedExactKernel<
    MomentumLayoutType::LAYOUT_GRAD_ONLY, Optimizers::SgdOptimizer>;
using RowwiseAdagradKernel = BackwardCodegenUnweightedExactKernel<
    MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_ROWWISE, Optimizers::RowWiseAdagradOptimizer>;

// uniq版本的类型别名
using AdagradUniqueKernel = BackwardCodegenUnweightedExactKernelUnique<
    MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_PER_DIM, Optimizers::AdagradOptimizer>;
using AdamUniqueKernel = BackwardCodegenUnweightedExactKernelUnique<
    MomentumLayoutType::LAYOUT_GRAD_MOMENTUM1_MOMENTUM2, Optimizers::AdamOptimizer>;
using SgdUniqueKernel = BackwardCodegenUnweightedExactKernelUnique<
    MomentumLayoutType::LAYOUT_GRAD_ONLY, Optimizers::SgdOptimizer>;

#define BACKWARD_CODEGEN_DISPATCH(id, kernel_class, name) \
    if (TILING_KEY_IS(id)) { \
        kernel_class kernel; \
        if (tiling_data.useOptimize) { \
            kernel.Compute(args); \
            kernel.UpdateWeightsScheduler(args); \
        } else { \
            kernel.Compute(args); \
        } \
    }

extern "C" __global__ __aicore__ void backward_codegen_adagrad_unweighted_exact(
    GM_ADDR gradOutput, GM_ADDR devWeights, GM_ADDR uvmWeights, GM_ADDR lxuCacheWeights, GM_ADDR weightsPlacements,
    GM_ADDR weightsOffsets, GM_ADDR dOffsets, GM_ADDR hashSizeCumsum, GM_ADDR indices, GM_ADDR offsets,
    GM_ADDR lxuCacheLocations, GM_ADDR momentum1Dev, GM_ADDR momentum1Uvm, GM_ADDR momentum1Placements,
    GM_ADDR momentum1Offsets, GM_ADDR momentum2Dev, GM_ADDR momentum2Uvm, GM_ADDR momentum2Placements,
    GM_ADDR momentum2Offsets, GM_ADDR hashIndices, GM_ADDR uniqueId, GM_ADDR uniqueHashSize, GM_ADDR uniqueInverse,
    GM_ADDR indiceSizeCumsum, GM_ADDR out, GM_ADDR momentum1DevOut, GM_ADDR momentum2DevOut, GM_ADDR weightsDevOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    Args args{gradOutput,      devWeights,      weightsPlacements, weightsOffsets,
              dOffsets,        hashSizeCumsum,  indices,           offsets,
              momentum1Dev,    momentum2Dev,    hashIndices,       uniqueId,
              uniqueHashSize,  uniqueInverse,   indiceSizeCumsum,  out,
              momentum1DevOut, momentum2DevOut, weightsDevOut,     workspace,
              tiling};

    BACKWARD_CODEGEN_DISPATCH(1, AdagradKernel, NORMAL_ADAGRAD)
    else BACKWARD_CODEGEN_DISPATCH(2, AdamKernel, NORMAL_ADAM)
    else BACKWARD_CODEGEN_DISPATCH(3, SgdKernel, NORMAL_SGD)
    else BACKWARD_CODEGEN_DISPATCH(4, AdagradUniqueKernel, UNIQUE_ADAGRAD)
    else BACKWARD_CODEGEN_DISPATCH(5, AdamUniqueKernel, UNIQUE_ADAM)
    else BACKWARD_CODEGEN_DISPATCH(6, SgdUniqueKernel, UNIQUE_SGD)
    else BACKWARD_CODEGEN_DISPATCH(7, RowwiseAdagradKernel, ROWWISE_ADAGRAD)

#undef BACKWARD_CODEGEN_DISPATCH
}