/**
 * @file backward_codegen_adagrad_unweighted_exact.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 */

// #include "backward_codegen_adagrad_unweighted_exact_kernel.h"
// #include "backward_codegen_adagrad_unweighted_exact_kernel_unique.h"
// #include "backward_codegen_adam_unweighted_exact_kernel.h"
#include "backward_codegen_sgd_unweighted_exact_kernel.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void backward_codegen_adagrad_unweighted_exact(GM_ADDR gradOutput,
                                                                                GM_ADDR devWeights,
                                                                                GM_ADDR uvmWeights,
                                                                                GM_ADDR lxuCacheWeights,
                                                                                GM_ADDR weightsPlacements,
                                                                                GM_ADDR weightsOffsets,
                                                                                GM_ADDR dOffsets,
                                                                                GM_ADDR hashSizeCumsum,
                                                                                GM_ADDR indices,
                                                                                GM_ADDR offsets,
                                                                                GM_ADDR lxuCacheLocations,
                                                                                GM_ADDR momentum1Dev,
                                                                                GM_ADDR momentum1Uvm,
                                                                                GM_ADDR momentum1Placements,
                                                                                GM_ADDR momentum1Offsets,
                                                                                GM_ADDR momentum2Dev,
                                                                                GM_ADDR momentum2Uvm,
                                                                                GM_ADDR momentum2Placements,
                                                                                GM_ADDR momentum2Offsets,
                                                                                GM_ADDR hashIndices,
                                                                                GM_ADDR uniqueId,
                                                                                GM_ADDR uniqueHashSize,
                                                                                GM_ADDR uniqueInverse,
                                                                                GM_ADDR out,
                                                                                GM_ADDR momentum1DevOut,
                                                                                GM_ADDR momentum2DevOut,
                                                                                GM_ADDR weightsDevOut,
                                                                                GM_ADDR workspace,
                                                                                GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    BackwardCodegenUnweightedExact::Args args{
        gradOutput, devWeights,      weightsPlacements, weightsOffsets, dOffsets,  hashSizeCumsum, indices,
        offsets,    momentum1Dev,    momentum2Dev,      hashIndices,    uniqueId,  uniqueHashSize, uniqueInverse,
        out,        momentum1DevOut, momentum2DevOut,   weightsDevOut,  workspace, tiling};
    if (TILING_KEY_IS(5)) {
        BackwardCodegenSgdUnweightedExact::BackwardCodegenSgdUnweightedExactKernel kernel;
        kernel.Compute(args);
    }
}