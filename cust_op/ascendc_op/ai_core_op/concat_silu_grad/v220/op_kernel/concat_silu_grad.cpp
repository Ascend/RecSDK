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
#include "lib/matmul_intf.h"

#include "kernel_operator.h"

#if defined(IS_A5) && IS_A5
#define IS_A5
#define USE_TLA
#else
#define IS_A2
#endif

#if defined(IS_A2)
#define ARCH_CODE AtlasA2
#define CATLASS_ARCH_A2_ENABLED
#endif

#if defined(IS_A5)
#define ARCH_CODE AtlasA5
#define CATLASS_ARCH_A5_ENABLED
#ifndef USE_TLA
#error use tla in a5 arch pls.
#endif
#endif

#include <acl/acl.h>
constexpr int32_t SPLIT_NUM = 4;

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "concat_silu_grad_tile.h"
#include "concat_silu_grad_kernel.h"
#include "concat_silu_grad_block_epilogue.h"

namespace PROJECT_NAMESPACE {

using namespace Catlass;

template <class InDType>
CATLASS_DEVICE void ConcatSiluGradImpl(GemmCoord problemShape, GM_ADDR grad1, GM_ADDR grad2, GM_ADDR grad3,
                                       GM_ADDR grad4, GM_ADDR silu_input, GM_ADDR grad_silu_input, int64_t* splitList)
{
    using LayoutT = layout::RowMajor;
    using ArchTag = Arch::ARCH_CODE;

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k1 = 0;
    constexpr int k0 = 0;
    using DispatchPolicy = void;
    using L1TileShape = GemmShape<m, n, k1>;
    using L0TileShape = GemmShape<m, n, k0>;
    using TType = Gemm::GemmType<InDType, LayoutT>;

    // Block level, define BlockEpilogue
    using EpilogueDispatchPolicy = EpilogueConcatSiluGrad;
    using ComputeType = Gemm::GemmType<float, LayoutT>;
    constexpr uint32_t computeLength = (m * n) / 2;
    using TileElemWiseEpilogue = TileSiluGrad<ArchTag, ComputeType, computeLength>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, TType, TType>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, TType, TType, TileElemWiseEpilogue, EpilogueTileCopy>;
    using EpilogueParams = typename BlockEpilogue::Params;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = ConcatSiluGradKernel<void, BlockEpilogue, BlockScheduler, L1TileShape>;

    GM_ADDR inputGradAddr[] = {grad1, grad2, grad3, grad4};
    LayoutT layoutSilu{problemShape.m(), problemShape.n()};
    EpilogueParams epilogueParams((GM_ADDR*)inputGradAddr, layoutSilu, splitList);
    typename MatmulKernel::Params params(problemShape, silu_input, grad_silu_input, layoutSilu, epilogueParams);
    MatmulKernel matmulKernel;
    matmulKernel(params);
}

}  // namespace PROJECT_NAMESPACE

extern "C" __global__ __aicore__ void concat_silu_grad(GM_ADDR grad1, GM_ADDR grad2, GM_ADDR grad3, GM_ADDR grad4,
                                                       GM_ADDR silu_input, GM_ADDR grad_silu_input, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    Catlass::GemmCoord problemShape{tiling_data.m, tiling_data.n, 0};

    if (TILING_KEY_IS(27)) {
        // bf16
        PROJECT_NAMESPACE::ConcatSiluGradImpl<bfloat16_t>(problemShape, grad1, grad2, grad3, grad4, silu_input,
                                                          grad_silu_input, tiling_data.splitList);
    } else if (TILING_KEY_IS(0)) {
        // float32
        PROJECT_NAMESPACE::ConcatSiluGradImpl<float>(problemShape, grad1, grad2, grad3, grad4, silu_input,
                                                     grad_silu_input, tiling_data.splitList);
    } else if (TILING_KEY_IS(1)) {
        // float16
        PROJECT_NAMESPACE::ConcatSiluGradImpl<half>(problemShape, grad1, grad2, grad3, grad4, silu_input,
                                                    grad_silu_input, tiling_data.splitList);
    }
}
