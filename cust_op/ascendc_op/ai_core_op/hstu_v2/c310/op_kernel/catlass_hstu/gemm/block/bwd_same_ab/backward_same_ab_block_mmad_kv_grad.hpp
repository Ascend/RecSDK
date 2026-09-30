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

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Kernel::SameAB {

// Shared dV/dK block. One complete Same AB input tile is consumed per call;
// Fixpipe SPLIT_M sends one output half to each AIV.
template <class ArchTag_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
          class TileBuffer_, class TileCopy_, class TileMmad_>
class BackwardSameABBlockMmadKVGrad {
public:
    using ArchTag = ArchTag_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using TileBuffer = TileBuffer_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;
    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t AIV_PER_AIC = 2;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});

    static_assert(STAGES == 2, "Same AB KVGrad requires two input stages");
    static_assert(L1_TILE_M == L0_TILE_M, "Current Same AB KVGrad implements the TILE_K=128 single-L0-M path only");

    CATLASS_DEVICE BackwardSameABBlockMmadKVGrad(Arch::Resource<ArchTag>& resource, uint32_t const (&vecFlag)[STAGES],
                                                 uint32_t cubeFlag, uint32_t ubFreeFlag, uint32_t accumulatorEventId,
                                                 uint32_t const (&l1BEventId)[STAGES])
        : accumulatorEventId_(accumulatorEventId)
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1A_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1A[i]);
            l1B_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1B[i]);
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[i]);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[i]);
            // 跨核 flag 的 id 由调用方按 stage 显式给出：Same AB 使用模式 0x2，
            // flagId 必须落在 0-15 内，不能再用 i * SYNC_FLAG_ID_MAX 展开。
            vecReady_[i] = Arch::CrossCoreFlag(vecFlag[i]);
            l1BEventId_[i] = l1BEventId[i];
        }
        l0C_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C);
        transOut_ = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST);
        cubeReady_ = Arch::CrossCoreFlag(cubeFlag);
        ubFree_ = Arch::CrossCoreFlag(ubFreeFlag);
    }

    CATLASS_DEVICE void SetDeqScalar(ElementAccumulator deqScalar)
    {
        // Same AB Fixpipe only performs NO_QUANT movement. dV scale has
        // already been folded into Prob by ScoreGrad.
        (void)deqScalar;
    }

    CATLASS_DEVICE void operator()(GemmCoord& blockShape, uint32_t& pingPongFlag, uint32_t& l0bFlag,
                                   bool isInit = false, bool isFlush = false, uint32_t l1ReuseSlot = 0)
    {
        uint32_t stage = consumeStage_ % STAGES;
        uint32_t mReal = blockShape.m();
        uint32_t nReal = blockShape.n();
        uint32_t kReal = blockShape.k();
        uint32_t l0Stage = pingPongFlag % STAGES;

        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE1>(vecReady_[stage].id);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Stage);

        uint32_t zero = 0;
        auto coord = tla::MakeCoord(zero, zero);
        auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(nReal, mReal);
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(nReal, mReal);
        copyL1ToL0A_(tla::MakeTensor(l0A_[l0Stage], l0aLayout, coord, Arch::PositionL0A{}),
                     tla::MakeTensor(l1A_[stage], l1aLayout, coord, Arch::PositionL1{}));

        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(mReal, kReal);
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(mReal, kReal);
        auto l1SlotOffset = l1ReuseSlot * L0_TILE_M * kReal;
        copyL1ToL0B_(tla::MakeTensor(l0B_[l0Stage], l0bLayout, coord, Arch::PositionL0B{}),
                     tla::MakeTensor(l1B_[stage][l1SlotOffset], l1bLayout, coord, Arch::PositionL1{}));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId_[stage]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Stage);

        auto tensorL0A = tla::MakeTensor(l0A_[l0Stage], l0aLayout, coord, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B_[l0Stage], l0bLayout, coord, Arch::PositionL0B{});
        auto l0cLayout = tla::MakeLayoutL0C(nReal, kReal);
        auto tensorL0C = tla::MakeTensor(l0C_, l0cLayout, coord, Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Stage);
        // L0A/L0B are ping-pong buffers selected by l0Stage, whereas l0C_ is
        // one fixed accumulator (gradVAcc or gradKAcc) for every invocation.
        // A fixed event must therefore protect the fixed L0C allocation.
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(accumulatorEventId_);
        tileMmad_(tensorL0C, tensorL0A, tensorL0B, nReal, kReal, mReal, isInit, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(accumulatorEventId_);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Stage);

        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(accumulatorEventId_);
        if (isFlush) {
            Flush(nReal, kReal);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(accumulatorEventId_);

        ++consumeStage_;
        pingPongFlag = (l0Stage + 1) % STAGES;
        l0bFlag = pingPongFlag;
    }

private:
    CATLASS_DEVICE void Flush(uint32_t nReal, uint32_t kReal)
    {
        // Fixpipe requires its L0C and UB tensor views to describe the same
        // physical M/N region. Rebuild both views with the aligned M extent,
        // matching the proven forward SPLIT_M implementation.
        uint32_t nAligned = RoundUp<L1AAlignHelper::M_ALIGNED>(nReal);
        uint32_t zero = 0;
        auto coord = tla::MakeCoord(zero, zero);
        auto l0cLayout = tla::MakeLayoutL0C(nAligned, kReal);
        auto tensorL0C = tla::MakeTensor(l0C_, l0cLayout, coord, Arch::PositionL0C{});
        auto dstLayout = tla::MakeLayout<ElementAccumulator, LayoutTagDST>(nAligned, kReal);
        auto tensorDst = tla::MakeTensor(transOut_, dstLayout, coord, Arch::PositionUB{});
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(tensorDst)>;
        CopyL0CToDst copy;

        // Publish the Fixpipe result, then keep Cube blocked until both AIVs
        // have consumed their FP32 transIn halves. The GM writeback may still
        // be in flight because it reads the independent low-precision
        // transOut buffer. This prevents a later Cube producer from
        // overwriting the aliased score/transIn region.
        copy(tensorDst, tensorL0C, 0);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeReady_.id);
        AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(ubFree_.id);
    }

    Arch::CrossCoreFlag vecReady_[STAGES];
    Arch::CrossCoreFlag cubeReady_;
    Arch::CrossCoreFlag ubFree_;
    AscendC::LocalTensor<ElementA> l1A_[STAGES];
    AscendC::LocalTensor<ElementB> l1B_[STAGES];
    AscendC::LocalTensor<ElementA> l0A_[STAGES];
    AscendC::LocalTensor<ElementB> l0B_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0C_;
    AscendC::LocalTensor<ElementAccumulator> transOut_;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
    uint32_t l1BEventId_[STAGES]{0};
    uint32_t accumulatorEventId_{0};
    uint32_t consumeStage_{0};
};

}  // namespace Catlass::Kernel::SameAB
