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

#ifndef HSTU_CATLASS_FORWARD_KERNEL_BUILDER_HPP
#define HSTU_CATLASS_FORWARD_KERNEL_BUILDER_HPP

#include "../../../tla_hstu/layout.hpp"
#include "../../../catlass_hstu/gemm/tile/copy_gm_to_l1_a5.hpp"
#include "../../../catlass_hstu/gemm/tile/copy_l0c_to_ub_a5.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_qk_infer.hpp"
#include "../../../catlass_hstu/gemm/block/block_mmad_pv_infer.hpp"
#include "../../../catlass_hstu/gemm/block/block_scheduler.hpp"
#include "../../../catlass_hstu/gemm/block/block_scheduler_interleaved.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_score_infer.hpp"
#include "../../../catlass_hstu/epilogue/block/block_epilogue_trans_infer.hpp"
#include "../../../catlass_hstu/kernel/fwd/forward_kernel_resource.hpp"

// =============================================================================
// 分核策略切换 (硬编码，重新编译生效)
//   注释掉 → 原始连续分块 (RowBlockScheduler)
//   取消注释 → 交替分块 (InterleavedRowBlockScheduler)，提高 L2 Cache 跨核共享率
// =============================================================================
#define USE_INTERLEAVED_Q_SCHEDULER

// =============================================================================
// K/V Cluster 数量：调参开关
//   数值越大 → L2 冲突越分散，但跨核共享率越低
//   数值越小 → 共享率越高，但冲突可能增加
//   可通过编译选项覆盖：-DHSTU_FWD_KV_CLUSTER_NUM=2
// =============================================================================
#ifndef HSTU_FWD_KV_CLUSTER_NUM
#define HSTU_FWD_KV_CLUSTER_NUM 4
#endif

using namespace Catlass;
using namespace tla;
using namespace utils;

namespace Catlass::Kernel {

template <class Element, uint32_t TILE_K>
struct TileSelector {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupport TileSelector.!");
};

template <class Element>
struct TileSelector<Element, 128> {
    using L1TileShape = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShape = Shape<Int<128>, Int<128>, Int<128>>;
};

template <class Element>
struct TileSelector<Element, 256> {
    using L1TileShape = Shape<Int<64>, Int<64>, Int<256>>;
    using L0TileShape = Shape<Int<64>, Int<64>, Int<256>>;
};

template <typename ArchTag_, typename ElementType_, typename ElementOffset_, uint32_t TILE_K_, bool HAS_RAB_,
          bool HAS_MASK_>
struct ForwardKernelConfig {
    using ElementType = ElementType_;
    using ElementOffset = ElementOffset_;
    using ArchTag = ArchTag_;

    static constexpr uint32_t TILE_K = TILE_K_;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;

    using L1TileShape = typename TileSelector<ElementType, TILE_K>::L1TileShape;
    using L0TileShape = typename TileSelector<ElementType, TILE_K>::L0TileShape;
    static constexpr uint32_t STAGES = 2;
};

template <typename ArchTag_, typename ElementType_, template <BufferTag> class TileBufferType_, typename L1TileShape_,
          typename L0TileShape_, bool HAS_RAB_, bool HAS_MASK_>
struct QKBlockBuilder {
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using ElementQ = ElementType;
    using LayoutQ = layout::zN;
    using ElementK = ElementType;
    using LayoutK = layout::nZ;
    using ElementS = typename Gemm::helper::ElementAccumulatorSelector<ElementQ, ElementK>::ElementAccumulator;
    using LayoutS = std::conditional_t<HAS_RAB, layout::RowMajor, layout::zN>;

    using TileCopyTlaQK =
        std::conditional_t<HAS_RAB,
                           Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                             LayoutS, void, Gemm::Tile::CopyL0CToUBMode::RESERVED>,
                           Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                             LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>>;

    using DispatchPolicy = Gemm::MmadHSTUQK<ArchTag, false, false>;

    using MmadTileBuffer = TileBufferType_<BufferTag::QK_MMAD>;
    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS,
                                                MmadTileBuffer, TileCopyTlaQK>;

    using EpilogueTileBuffer = TileBufferType_<BufferTag::SCORE_EPILOGUE>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueScore<ArchTag, ElementQ, ElementS, EpilogueTileBuffer,
                                                              L0TileShape, HAS_RAB, HAS_MASK>;
};

template <typename ArchTag_, typename ElementType_, template <BufferTag> class TileBufferType_, typename L1TileShape_,
          typename L0TileShape_, bool HAS_RAB_, bool HAS_MASK_>
struct PVBlockBuilder {
    using ArchTag = ArchTag_;
    using ElementType = ElementType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;

    using ElementP = ElementType;
    using LayoutP = std::conditional_t<HAS_RAB, layout::RowMajor, layout::zN>;
    using ElementV = ElementType;
    using LayoutV = layout::zN;
    using ElementO = typename Gemm::helper::ElementAccumulatorSelector<ElementP, ElementV>::ElementAccumulator;
    using LayoutO = layout::RowMajor;

    using TileCopyTlaPV = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutP, ElementV, LayoutV, ElementO,
                                                            LayoutO, void, Gemm::Tile::CopyL0CToUBMode::RESERVED>;

    using DispatchPolicy = Gemm::MmadHSTUPV<ArchTag, false, false>;

    using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementP, ElementV, ElementO,
                                                TileBufferType_<BufferTag::PV_MMAD>, TileCopyTlaPV>;

    using BlockEpilogue =
        Epilogue::Block::BlockEpilogueTransOut<ArchTag, TileBufferType_<BufferTag::TRANS_PV_EPILOGUE>,
                                               Epilogue::Block::TransTag::UB_TO_GM, ElementO, ElementV, HAS_RAB>;
};

template <typename ArchTag_, typename ElementOffset_, typename L1TileShape_>
struct BlockSchedulerBuilder {
    using ArchTag = ArchTag_;
    using ElementOffset = ElementOffset_;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});

#ifdef USE_INTERLEAVED_Q_SCHEDULER
    using QBlockScheduler = Gemm::Block::InterleavedRowBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N>;
#else
    using QBlockScheduler = Gemm::Block::RowBlockScheduler<ElementOffset, BLOCK_M, BLOCK_N>;
#endif
    using KBlockScheduler =
        Gemm::Block::ColumnBlockScheduler<QBlockScheduler, BLOCK_M, BLOCK_N, false, false, HSTU_FWD_KV_CLUSTER_NUM>;
};

template <typename KernelConfig_>
struct ForwardKernelBuilder {
    using Config = KernelConfig_;

    using ElementType = typename Config::ElementType;
    using ElementOffset = typename Config::ElementOffset;
    using ArchTag = typename Config::ArchTag;

    static constexpr bool HAS_RAB = Config::HAS_RAB;
    static constexpr bool HAS_MASK = Config::HAS_MASK;
    static constexpr int32_t TILE_K = Config::TILE_K;
    static constexpr uint32_t STAGES = Config::STAGES;

    using AccumulatorElementType =
        typename Gemm::helper::ElementAccumulatorSelector<ElementType, ElementType>::ElementAccumulator;

    // Tile shapes
    using L1TileShape = typename Config::L1TileShape;
    using L0TileShape = typename Config::L0TileShape;

    // Tile Buffer
    using KernelResource = ForwardKernelResource<ArchTag, L1TileShape, L0TileShape, ElementType, AccumulatorElementType,
                                                 STAGES, HAS_RAB, HAS_MASK>;

    // Helper alias
    template <BufferTag Tag>
    using TileBufferType = TileBuffer<KernelResource, Tag>;

    // Block QK Builder
    using QKBlockBuilder =
        QKBlockBuilder<ArchTag, ElementType, TileBufferType, L1TileShape, L0TileShape, HAS_RAB, HAS_MASK>;
    using BlockMmadQK = typename QKBlockBuilder::BlockMmad;
    using BlockEpilogueQK = typename QKBlockBuilder::BlockEpilogue;

    // Block PV Builder
    using PVBlockBuilder =
        PVBlockBuilder<ArchTag, ElementType, TileBufferType, L1TileShape, L0TileShape, HAS_RAB, HAS_MASK>;
    using BlockMmadPV = typename PVBlockBuilder::BlockMmad;
    using BlockEpiloguePV = typename PVBlockBuilder::BlockEpilogue;

    using BlockSchedulerBuilder = BlockSchedulerBuilder<ArchTag, ElementOffset, L1TileShape>;
    using QBlockScheduler = typename BlockSchedulerBuilder::QBlockScheduler;
    using KBlockScheduler = typename BlockSchedulerBuilder::KBlockScheduler;
};

}  // namespace Catlass::Kernel

#endif
