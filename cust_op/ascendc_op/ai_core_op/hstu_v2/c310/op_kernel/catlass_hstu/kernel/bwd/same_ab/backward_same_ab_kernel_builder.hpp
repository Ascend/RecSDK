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
============================================================================== */

#pragma once

#include "../backward_kernel_builder.hpp"
#include "../../../gemm/block/bwd_same_ab/backward_same_ab_block_mmad_qk.hpp"
#include "../../../gemm/block/bwd_same_ab/backward_same_ab_block_mmad_kv_grad.hpp"
#include "../../../gemm/block/bwd_same_ab/backward_same_ab_block_mmad_dq.hpp"
#include "../../../gemm/tile/bwd_same_ab/backward_same_ab_copy_l0c_to_ub.hpp"
#include "backward_same_ab_kernel_resource.hpp"
#include "../../../epilogue/block/bwd_same_ab/backward_same_ab_score_grad.hpp"
#include "../../../epilogue/block/bwd_same_ab/backward_same_ab_rab_grad.hpp"
#include "../../../epilogue/block/bwd_same_ab/backward_same_ab_trans_out.hpp"
#include "backward_same_ab_kernel_mmad_mainloop.hpp"
#include "backward_same_ab_kernel_epilogue_mainloop.hpp"

namespace Catlass::Kernel::SameAB {

// Step 1 switch. Keep this false until all Same AB modules are implemented
// and the host tiling selects a dedicated, validated template branch.
static constexpr bool ENABLE_SAME_AB_BRANCH = true;

template <typename Element, uint32_t TILE_K>
struct TileSelector {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported Same AB backward tile shape");
};

template <typename Element>
struct TileSelector<Element, 128> {
    // One logical QK/GV tile is 128 x 128. Two complete logical tiles can be
    // resident in L1 without aliasing Grad and Grab.
    using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
    using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
};

template <typename Element>
struct TileSelector<Element, 256> {
    // TILE_K=256 means headDim is up to 256, so the K extent has to cover the
    // whole head dimension. M is reduced to 64 instead: one L0A/L0B tile
    // (64 x 256) then occupies exactly one L0 stage slot and the matching L0C
    // tile fits its region, so every block still issues a single Mmad and the
    // Fixpipe SPLIT_M contract (M is split across the two AIVs) is unchanged.
    using L1TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<256>>;
    using L0TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<256>>;
};

template <typename ArchTag_, typename ElementType_, typename ElementOffset_, uint32_t TILE_K_, bool HAS_RAB_,
          bool IS_LOCAL_, bool IS_CAUSAL_, bool IS_ARBITRARY_>
struct KernelConfig {
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using ElementOffset = ElementOffset_;

    static constexpr uint32_t TILE_K = TILE_K_;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = IS_LOCAL_ || IS_CAUSAL_ || IS_ARBITRARY_;
    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t AIV_PER_AIC = 2;

    using L1TileShape = typename SameAB::TileSelector<ElementType, TILE_K>::L1TileShape;
    using L0TileShape = typename SameAB::TileSelector<ElementType, TILE_K>::L0TileShape;

    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Same AB backward currently supports Ascend950 only");
    static_assert(TILE_K == 128 || TILE_K == 256, "Same AB backward TILE_K must be 128 or 256");
};

template <typename ElementOffset_, typename L1TileShape_>
struct BlockSchedulerBuilder {
    using ElementOffset = ElementOffset_;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});

    // Same AB does not change the outer logical tile traversal. Reuse the
    // existing schedulers; M-half ownership belongs to Fixpipe/Epilogue.
    using KBlockScheduler = Gemm::Block::RowBlockScheduler<ElementOffset, BLOCK_N, BLOCK_M>;
    using QBlockScheduler = Gemm::Block::ColumnBlockScheduler<KBlockScheduler, BLOCK_N, BLOCK_M, true>;
};

template <typename KernelConfig_>
struct KernelBuilder {
    using Config = KernelConfig_;
    using LegacyBuilder = Catlass::Kernel::BackwardKernelBuilder<Config>;

    using ElementType = typename Config::ElementType;
    using ElementOffset = typename Config::ElementOffset;
    using ArchTag = typename Config::ArchTag;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementType, ElementType>::ElementAccumulator;
    using L1TileShape = typename Config::L1TileShape;
    using L0TileShape = typename Config::L0TileShape;

    static constexpr uint32_t TILE_K = Config::TILE_K;
    static constexpr uint32_t STAGES = Config::STAGES;
    static constexpr uint32_t AIV_PER_AIC = Config::AIV_PER_AIC;
    static constexpr bool HAS_RAB = Config::HAS_RAB;
    static constexpr bool HAS_MASK = Config::HAS_MASK;
    static constexpr bool USE_PHYSICAL_NZ = !HAS_RAB && !HAS_MASK;

    // Components are initialized as independent Same AB assembly aliases.
    // Modules not implemented yet continue to delegate to the legacy builder.
    using KernelResource = SameAB::KernelResource<ArchTag, L1TileShape, L0TileShape, ElementType, ElementAccumulator,
                                                  STAGES, HAS_RAB, HAS_MASK>;
    template <SameAB::BufferTag Tag>
    using TileBufferType = SameAB::TileBuffer<KernelResource, Tag>;
    using CopyL0CToUB = SameAB::BackwardSameABCopyL0CToUB<KernelResource>;

    // no-RAB/no-mask keeps the physical zN order through the vector epilogue,
    // so Prob/Grab can be copied to the downstream nZ L1 tiles without a
    // per-C0 RowMajor scatter. RAB/mask still require logical row addressing.
    using ScoreLayout = std::conditional_t<USE_PHYSICAL_NZ, layout::zN, layout::RowMajor>;
    using QKTileCopy =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementType, layout::zN, ElementType, layout::nZ, ElementAccumulator,
                                          ScoreLayout, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false,
                                          Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using QKTileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementType, typename QKTileCopy::LayoutTagL1A>;
    using BlockMmadQK = SameAB::BackwardSameABBlockMmadQK<ArchTag, L1TileShape, L0TileShape, ElementType,
                                                          TileBufferType<SameAB::BufferTag::QK_MMAD>, QKTileCopy,
                                                          QKTileMmad, CopyL0CToUB>;
    using BlockEpilogueQK = SameAB::BackwardSameABScoreGrad<ArchTag, ElementType, ElementAccumulator,
                                                            TileBufferType<SameAB::BufferTag::SCORE_GRAD_EPILOGUE>,
                                                            L1TileShape, HAS_RAB, HAS_MASK>;

    // GV is QK-like GEMM with different buffers. Same AB Fixpipe remains
    // NO_QUANT; scale * alpha is applied by the AIV RabGrad stage.
    using BlockMmadGV = SameAB::BackwardSameABBlockMmadQK<ArchTag, L1TileShape, L0TileShape, ElementType,
                                                          TileBufferType<SameAB::BufferTag::GV_MMAD>, QKTileCopy,
                                                          QKTileMmad, CopyL0CToUB>;
    using BlockEpilogueGV =
        SameAB::BackwardSameABRabGrad<BlockEpilogueQK, TileBufferType<SameAB::BufferTag::RAB_GRAD_EPILOGUE>>;

    using VGradTileCopy =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementType, layout::nZ, ElementType, layout::zN, ElementType,
                                          layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false,
                                          Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using VGradTileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementType, typename VGradTileCopy::LayoutTagL1A>;
    using BlockMmadVGrad =
        SameAB::BackwardSameABBlockMmadKVGrad<ArchTag, L1TileShape, L0TileShape, ElementType, ElementType, ElementType,
                                              TileBufferType<SameAB::BufferTag::V_GRAD_MMAD>, VGradTileCopy,
                                              VGradTileMmad>;

    using KGradTileCopy =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementType, layout::nZ, ElementType, layout::zN, ElementType,
                                          layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false,
                                          Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using KGradTileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementType, typename KGradTileCopy::LayoutTagL1A>;
    using BlockMmadKGrad =
        SameAB::BackwardSameABBlockMmadKVGrad<ArchTag, L1TileShape, L0TileShape, ElementType, ElementType, ElementType,
                                              TileBufferType<SameAB::BufferTag::K_GRAD_MMAD>, KGradTileCopy,
                                              KGradTileMmad>;

    using BlockEpilogueTransOut =
        SameAB::BackwardSameABTransOut<ArchTag, TileBufferType<SameAB::BufferTag::TRANS_KV_GRAD_EPILOGUE>, ElementType,
                                       ElementAccumulator>;
    using BlockEpilogueVGrad = BlockEpilogueTransOut;
    using BlockEpilogueKGrad = BlockEpilogueTransOut;

    // dQ consumes the merged L1 Grab tile after dK, then atomically adds each
    // K-block contribution to FP32 QShare. The final QShare -> dQ conversion
    // reuses the proven, lifetime-exclusive transpose epilogue.
    using QGradTileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementType, layout::zN, ElementType, layout::zN,
                                                        ElementAccumulator, layout::RowMajor>;
    using QGradTileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementType, typename QGradTileCopy::LayoutTagL1A>;
    using BlockMmadQGrad =
        SameAB::BackwardSameABBlockMmadDQ<ArchTag, L1TileShape, L0TileShape, ElementType,
                                          TileBufferType<SameAB::BufferTag::Q_GRAD_MMAD>, QGradTileCopy, QGradTileMmad>;
    using BlockEpilogueQGrad =
        Epilogue::Block::BlockEpilogueTransOut<ArchTag, TileBufferType<SameAB::BufferTag::TRANS_Q_GRAD_EPILOGUE>,
                                               Epilogue::Block::TransTag::GM_TO_GM, ElementAccumulator, ElementType,
                                               false>;

    using SchedulerBuilder = BlockSchedulerBuilder<ElementOffset, L1TileShape>;
    using QBlockScheduler = typename SchedulerBuilder::QBlockScheduler;
    using KBlockScheduler = typename SchedulerBuilder::KBlockScheduler;

    CATLASS_DEVICE static constexpr void InitComponents()
    {
        static_assert(AIV_PER_AIC == 2, "Same AB backward requires MIX AIC:AIV = 1:2");
        static_assert(STAGES == 2, "Step 1 must preserve the legacy pipeline depth");
        static_assert(KernelResource::L1_BYTES <= ArchTag::L1_SIZE, "Same AB backward resource exceeds L1 capacity");
        static_assert(KernelResource::L0A_BYTES <= ArchTag::L0A_SIZE, "Same AB backward resource exceeds L0A capacity");
        static_assert(KernelResource::L0B_BYTES <= ArchTag::L0B_SIZE, "Same AB backward resource exceeds L0B capacity");
        static_assert(KernelResource::L0C_BYTES <= ArchTag::L0C_SIZE, "Same AB backward resource exceeds L0C capacity");
        static_assert(KernelResource::UB_EPILOGUE_BYTES <= ArchTag::UB_SIZE,
                      "Same AB backward resource exceeds UB epilogue capacity");
        static_assert(KernelResource::UB_Q_GRAD_BYTES <= ArchTag::UB_SIZE,
                      "Same AB backward resource exceeds UB Q-grad capacity");
        static_assert(CopyL0CToUB::AIV_M * CopyL0CToUB::AIV_PER_AIC == CopyL0CToUB::LOGICAL_M,
                      "Same AB Fixpipe M ownership is inconsistent");
    }
};

}  // namespace Catlass::Kernel::SameAB
