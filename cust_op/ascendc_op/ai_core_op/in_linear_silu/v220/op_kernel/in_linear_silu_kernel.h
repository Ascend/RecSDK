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

#ifndef _IN_LINEAR_SILU_KERNEL_H
#define _IN_LINEAR_SILU_KERNEL_H

#if defined(IS_A5) && IS_A5
#define IS_A2 0
#define USE_TLA 1
#else
#define IS_A2 1
#define IS_A5 0
#define USE_TLA 0
#endif

#if IS_A2
#define ARCH_CODE AtlasA2
#define CATLASS_ARCH_A2_ENABLED
#endif

#if IS_A5
#define ARCH_CODE AtlasA5
#define CATLASS_ARCH_A5_ENABLED
#if USE_TLA == 0
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

#include "tile_elemwise_silu.h"
#include "block_epilogue_elemwise_split.h"

// A5架构下MatMul Block及Kernel部分需要改成TLA
#if USE_TLA
#include "catlass/gemm/block/block_mmad_pingpong_tla.hpp"
#include "matmul_bias_kernel_tla.h"
#include "tla/layout.hpp"
#else
#include "block_mmad_pingpong_bias.h"
#include "matmul_bias_kernel.h"
#endif

namespace InLinearSilu_Kernel {
template <class LayoutB, class InDType>
CATLASS_DEVICE void InLinearSiluImpl(GemmCoord problemShape, GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR user,
                                     GM_ADDR value, GM_ADDR query, GM_ADDR key, GM_ADDR linearOutput,
                                     int64_t* splitList)
{
    using LayoutA = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using LayoutD = layout::RowMajor;
    using LayoutBias = layout::VectorLayout;

    using ArchTag = Arch::ARCH_CODE;

    constexpr int m = 256;
    constexpr int n = 128;
    constexpr int k1 = 512 / sizeof(InDType);
    constexpr int k0 = 128 / sizeof(InDType);

#if USE_TLA
    using MmadDispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = tla::Shape<tla::Int<m>, tla::Int<n>, tla::Int<k1>>;
    using L0TileShape = tla::Shape<tla::Int<m>, tla::Int<n>, tla::Int<k0>>;
#else
    // 1. catlass仓模板有bug，重写
    // 2. 2025.12.25 catlass仓模板bug修复， 性能不行， 继续使用自定义模板
    using DispatchPolicy = InLinearSilu_Kernel::MmadPingpongBias<true>;
    using L1TileShape = GemmShape<m, n, k1>;
    using L0TileShape = GemmShape<m, n, k0>;
#endif
    using AType = Gemm::GemmType<InDType, LayoutA>;
    using BType = Gemm::GemmType<InDType, LayoutB>;
    using BiasType = Gemm::GemmType<float, LayoutBias>;
    using CType = Gemm::GemmType<InDType, LayoutC>;
    using DType = Gemm::GemmType<InDType, LayoutD>;

#if USE_TLA
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementC = typename CType::Element;
    using ElementBias = typename BiasType::Element;

    using LayoutTagA = LayoutA;
    using LayoutTagB = LayoutB;
    using LayoutTagC = LayoutC;

    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC,
                                                   LayoutTagC, ElementBias>;
    using BlockMmad = Gemm::Block::BlockMmadTla<MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB,
                                                ElementC, ElementBias, TileCopy>;
#else
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType>;
#endif

    // Block level, define BlockEpilogue
    using EpilogueDispatchPolicy = MyEpilogueElemWiseSplit;
    using ComputeType = Gemm::GemmType<float, LayoutD>;
    constexpr uint32_t computeLength = (m * n) / 2;
    using TileElemWiseEpilogue = TileElemWiseSilu<ArchTag, ComputeType, computeLength>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<Arch::AtlasA2, CType, DType>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;
    using EpilogueParams = typename BlockEpilogue::Params;

    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
// kernel level
#if USE_TLA
        using MatmulKernel = MatmulBiasTlaKernel<BlockMmad, BlockEpilogue, BlockScheduler>;
#else
        using MatmulKernel = MatmulBiasKernel<BlockMmad, BlockEpilogue, BlockScheduler>;
#endif

        GM_ADDR outputAddr[] = {user, value, query, key};

#if USE_TLA
        // Create TLA layouts for kernel usage
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(problemShape.m(), problemShape.k());
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(problemShape.k(), problemShape.n());
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(problemShape.m(), problemShape.n());
        LayoutC layoutC0{problemShape.m(), problemShape.n()};
        EpilogueParams epilogueParams(linearOutput, (GM_ADDR*)outputAddr, layoutC0, splitList);
#else
        LayoutA layoutA{problemShape.m(), problemShape.k()};
        LayoutB layoutB{problemShape.k(), problemShape.n()};
        LayoutC layoutC{problemShape.m(), problemShape.n()};
        EpilogueParams epilogueParams(linearOutput, (GM_ADDR*)outputAddr, layoutC, splitList);
#endif

        typename MatmulKernel::Params params(problemShape, x, layoutA, weight, layoutB, linearOutput, layoutC, bias,
                                             epilogueParams);
        MatmulKernel matmulKernel;
        matmulKernel(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // kernel level
#if USE_TLA
        using MatmulKernel = MatmulBiasTlaKernel<BlockMmad, BlockEpilogue, BlockScheduler>;
#else
        using MatmulKernel = MatmulBiasKernel<BlockMmad, BlockEpilogue, BlockScheduler>;
#endif

        GM_ADDR outputAddr[] = {user, value, query, key};

#if USE_TLA
        // Create TLA layouts for kernel usage
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(problemShape.m(), problemShape.k());
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(problemShape.k(), problemShape.n());
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(problemShape.m(), problemShape.n());
        LayoutC layoutC0{problemShape.m(), problemShape.n()};
        EpilogueParams epilogueParams(linearOutput, (GM_ADDR*)outputAddr, layoutC0, splitList);
#else
        LayoutA layoutA{problemShape.m(), problemShape.k()};
        LayoutB layoutB{problemShape.k(), problemShape.n()};
        LayoutC layoutC{problemShape.m(), problemShape.n()};
        EpilogueParams epilogueParams(linearOutput, (GM_ADDR*)outputAddr, layoutC, splitList);
#endif

        typename MatmulKernel::Params params(problemShape, x, layoutA, weight, layoutB, linearOutput, layoutC, bias,
                                             epilogueParams);
        MatmulKernel matmulKernel;
        matmulKernel(params);
    }
}
}  // namespace InLinearSilu_Kernel

#endif