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
#include "../../dispatch_policy.hpp"
#include "../../tile/bwd_same_ab/backward_same_ab_copy_l0c_to_ub.hpp"

namespace Catlass::Kernel::SameAB {

// One logical QK call produces one complete 128-row tile. Consecutive calls
// alternate all stage-owned Q/L0/score buffers, allowing QK0 and QK1 to be in
// flight before the pipeline advances to GV.
template <class ArchTag_, class L1TileShape_, class L0TileShape_, class Element_, class TileBuffer_, class TileCopy_,
          class TileMmad_, class CopyL0CToUB_>
class BackwardSameABBlockMmadQK {
public:
    using ArchTag = ArchTag_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = Element_;
    using ElementB = Element_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using TileBuffer = TileBuffer_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using CopyL0CToUB = CopyL0CToUB_;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;
    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});

    static_assert(STAGES == 2, "Same AB QK requires two stages");
    static_assert(L1_TILE_M % L1AAlignHelper::M_ALIGNED == 0,
                  "Same AB QK requires a fractal aligned logical M tile so one padded SPLIT_M half fits one AIV");
    static_assert(L0_TILE_M == L1_TILE_M, "Same AB QK requires one L0 tile to cover the logical M tile");

    CATLASS_DEVICE BackwardSameABBlockMmadQK(Arch::Resource<ArchTag>& resource, uint32_t headNum, uint32_t headDim,
                                             uint32_t const (&cubeFlag)[STAGES], uint32_t keyEventId,
                                             uint32_t const (&queryEventId)[STAGES])
        : keyEventId_(keyEventId),
          headNum_(headNum),
          stride_(headNum * headDim)
    {
        keyL1_ = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1B);
        for (uint32_t i = 0; i < STAGES; ++i) {
            queryL1_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1A[i]);
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[i]);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[i]);
            l0C_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[i]);
            scoreUB_[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[i]);
            queryEventId_[i] = queryEventId[i];
            // 跨核 flag 的 id 由调用方按 stage 显式给出：Same AB 使用模式 0x2，
            // flagId 必须落在 0-15 内，不能再用 i * SYNC_FLAG_ID_MAX 展开。
            cubeReady_[i] = Arch::CrossCoreFlag(cubeFlag[i]);
        }
    }

    CATLASS_DEVICE void SetDeqScalar(ElementAccumulator deqScalar)
    {
        // Preserve the legacy mainloop-facing API while the Same AB branch is
        // assembled. Same AB Fixpipe is NO_QUANT; alpha/scale belongs to AIV.
        (void)deqScalar;
    }

    template <class TensorK>
    CATLASS_DEVICE void AcquireTensor(TensorK& tensorK)
    {
        uint32_t nReal = tla::get<0>(tensorK.shape());
        uint32_t kReal = tla::get<1>(tensorK.shape());
        uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nReal);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(keyEventId_);
        auto layout = tla::MakeLayout<ElementB, LayoutTagL1B>(nRound, kReal);
        uint32_t zero = 0;
        auto coord = tla::MakeCoord(zero, zero);
        using Copy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorK, typename TileCopy::TensorL1B>;
        Copy{}(tla::MakeTensor(keyL1_, layout, coord, Arch::PositionL1{}), tensorK, nReal, kReal, stride_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(keyEventId_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(keyEventId_);
    }

    CATLASS_DEVICE void ReleaseTensor()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keyEventId_);
    }

    template <class TensorQ, class TensorK>
    CATLASS_DEVICE void operator()(TensorQ& tensorQ, TensorK& tensorK, uint32_t& pingPongFlag, uint32_t& l0bFlag,
                                   bool reuseQuery, uint32_t l1ReuseSlot = 0)
    {
        // The logical producer stage belongs to this QK/GV component.  The
        // shared pingPongFlag only selects the physical L0 ping-pong buffer;
        // QK and GV are called alternately and therefore cannot derive their
        // independent score/gs stage from that shared flag.
        uint32_t stage = produceStage_ % STAGES;
        uint32_t l0Stage = pingPongFlag % STAGES;
        uint32_t mReal = tla::get<0>(tensorQ.shape());
        uint32_t nReal = tla::get<0>(tensorK.shape());
        uint32_t kReal = tla::get<1>(tensorQ.shape());
        uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nReal);
        uint32_t zero = 0;
        auto coord = tla::MakeCoord(zero, zero);

        // K is shared by QK0/QK1. It is reloaded from the single L1 K tile
        // into the current L0B stage while Q uses independent L1 stages.
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Stage);
        auto keyL1Layout = tla::MakeLayout<ElementB, LayoutTagL1B>(nRound, kReal);
        auto keyL0Layout = tla::MakeLayout<ElementB, LayoutTagL0B>(nRound, kReal);
        copyL1ToL0B_(tla::MakeTensor(l0B_[l0Stage], keyL0Layout, coord, Arch::PositionL0B{}),
                     tla::MakeTensor(keyL1_, keyL1Layout, coord, Arch::PositionL1{}));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(queryEventId_[stage]);
        auto queryL1Layout = tla::MakeLayout<ElementA, LayoutTagL1A>(mReal, kReal);
        // Each stage owns a full L1_TILE_M x L1_TILE_K allocation. For narrow
        // head dimensions, pack multiple logical Q blocks into that allocation
        // and use a stage-local cache tag to safely reuse a resident slot when
        // the scheduler reverses direction on the next K block.
        auto l1SlotOffset = l1ReuseSlot * L0_TILE_M * kReal;
        auto queryL1Tensor = tla::MakeTensor(queryL1_[stage][l1SlotOffset], queryL1Layout, coord, Arch::PositionL1{});
        using Copy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorQ, typename TileCopy::TensorL1A>;
        if (!reuseQuery) {
            Copy{}(queryL1Tensor, tensorQ, mReal, kReal, stride_);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(queryEventId_[stage]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(queryEventId_[stage]);

        auto queryL0Layout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, kReal);
        copyL1ToL0A_(tla::MakeTensor(l0A_[l0Stage], queryL0Layout, coord, Arch::PositionL0A{}), queryL1Tensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Stage);

        auto l0cLayout = tla::MakeLayoutL0C(mReal, nRound);
        auto tensorL0C = tla::MakeTensor(l0C_[l0Stage], l0cLayout, coord, Arch::PositionL0C{});
        auto tensorL0A = tla::MakeTensor(l0A_[l0Stage], queryL0Layout, coord, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B_[l0Stage], keyL0Layout, coord, Arch::PositionL0B{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Stage);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0Stage);
        tileMmad_(tensorL0C, tensorL0A, tensorL0B, mReal, nReal, kReal, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0Stage);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Stage);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0Stage);

        // Keep the physical NZ contract used by the proven SameAB path:
        // although the destination tensor stores FP32 accumulators, its NZ
        // fractal geometry is described with ElementA so one C0 contains the
        // same 16 logical columns consumed by the vector epilogue. Keep this
        // kernel's established padded-M SPLIT_M ownership for both layouts;
        // only the NZ C0 geometry differs from RowMajor.
        using DstLayoutElement =
            std::conditional_t<std::is_same_v<LayoutTagDST, layout::zN>, ElementA, ElementAccumulator>;
        uint32_t dstM = RoundUp<L1AAlignHelper::M_ALIGNED>(mReal);
        auto dstLayout = tla::MakeLayout<DstLayoutElement, LayoutTagDST>(dstM, nRound);
        auto tensorDst = tla::MakeTensor(scoreUB_[stage], dstLayout, coord, Arch::PositionUB{});
        copyL0CToUB_(tensorDst, tensorL0C, 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0Stage);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeReady_[stage].id);

        // query/grad L1 remains live for the later dK/dV consumer. That
        // consumer releases queryEventId_[stage] after its L1B -> L0B copy.
        ++produceStage_;
        pingPongFlag = (l0Stage + 1) % STAGES;
        l0bFlag = pingPongFlag;
    }

private:
    Arch::CrossCoreFlag cubeReady_[STAGES];
    AscendC::LocalTensor<ElementA> queryL1_[STAGES];
    AscendC::LocalTensor<ElementB> keyL1_;
    AscendC::LocalTensor<ElementA> l0A_[STAGES];
    AscendC::LocalTensor<ElementB> l0B_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0C_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> scoreUB_[STAGES];
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
    CopyL0CToUB copyL0CToUB_;
    uint32_t keyEventId_{0};
    uint32_t queryEventId_[STAGES]{0};
    uint32_t headNum_{0};
    int64_t stride_{0};
    uint32_t produceStage_{0};
};

}  // namespace Catlass::Kernel::SameAB
