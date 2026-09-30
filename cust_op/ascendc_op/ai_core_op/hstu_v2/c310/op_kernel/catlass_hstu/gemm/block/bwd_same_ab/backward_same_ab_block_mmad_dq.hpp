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

#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Kernel::SameAB {

// Same AB dQ: Grab[M, N] * K[N, K] -> QShare[M, K].  QShare is FP32 GM
// storage shared by K blocks, so each complete MMAD result uses atomic add.
// dK is the preceding consumer of GRAB_READY; this block deliberately reads
// the same L1 grab stage without waiting on that cross-core token a second
// time.
template <class ArchTag_, class L1TileShape_, class L0TileShape_, class Element_, class TileBuffer_, class TileCopy_,
          class TileMmad_>
class BackwardSameABBlockMmadDQ {
public:
    using ArchTag = ArchTag_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using Element = Element_;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<Element, Element>::ElementAccumulator;
    using TileBuffer = TileBuffer_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});

    static_assert(STAGES == 2, "Same AB dQ requires two Grab stages");
    static_assert(L1_TILE_M == L0_TILE_M, "Current Same AB dQ implements the TILE_K=128 single-L0-M path only");

    CATLASS_DEVICE explicit BackwardSameABBlockMmadDQ(Arch::Resource<ArchTag>& resource)
    {
        keyL1_ = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::L1B);
        for (uint32_t i = 0; i < STAGES; ++i) {
            grabL1_[i] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::L1A[i]);
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<Element>(TileBuffer::L0A[i]);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<Element>(TileBuffer::L0B[i]);
            l0C_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[i]);
        }
    }

    template <class TensorQShare>
    CATLASS_DEVICE void operator()(TensorQShare& tensorQShare, GemmCoord& blockShape, uint32_t& pingPongFlag,
                                   uint32_t& l0bFlag)
    {
        uint32_t stage = consumeStage_ % STAGES;
        uint32_t l0Stage = pingPongFlag % STAGES;
        uint32_t mReal = blockShape.m();
        uint32_t nReal = blockShape.n();
        uint32_t kReal = blockShape.k();
        uint32_t zero = 0;
        auto coord = tla::MakeCoord(zero, zero);

        // Key is held by the outer K-block loop. Grab readiness was consumed
        // by dK immediately before this call; MTE ordering keeps its L1 write
        // visible while both Cube consumers execute.
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Stage);
        auto l1bLayout = tla::MakeLayout<Element, LayoutTagL1B>(nReal, kReal);
        auto l0bLayout = tla::MakeLayout<Element, LayoutTagL0B>(nReal, kReal);
        copyL1ToL0B_(tla::MakeTensor(l0B_[l0Stage], l0bLayout, coord, Arch::PositionL0B{}),
                     tla::MakeTensor(keyL1_, l1bLayout, coord, Arch::PositionL1{}));

        auto l1aLayout = tla::MakeLayout<Element, LayoutTagL1A>(mReal, nReal);
        auto l0aLayout = tla::MakeLayout<Element, LayoutTagL0A>(mReal, nReal);
        copyL1ToL0A_(tla::MakeTensor(l0A_[l0Stage], l0aLayout, coord, Arch::PositionL0A{}),
                     tla::MakeTensor(grabL1_[stage], l1aLayout, coord, Arch::PositionL1{}));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Stage);

        auto tensorL0A = tla::MakeTensor(l0A_[l0Stage], l0aLayout, coord, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B_[l0Stage], l0bLayout, coord, Arch::PositionL0B{});
        auto l0cLayout = tla::MakeLayoutL0C(mReal, kReal);
        auto tensorL0C = tla::MakeTensor(l0C_[l0Stage], l0cLayout, coord, Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Stage);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0Stage);
        tileMmad_(tensorL0C, tensorL0A, tensorL0B, mReal, kReal, nReal, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0Stage);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Stage);

        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0Stage);
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorQShare>;
        CopyL0CToDst copy;
        AscendC::SetAtomicAdd<ElementAccumulator>();
        copy(tensorQShare, tensorL0C, 0);
        AscendC::SetAtomicNone();
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0Stage);

        ++consumeStage_;
        pingPongFlag = (l0Stage + 1) % STAGES;
        l0bFlag = pingPongFlag;
    }

private:
    AscendC::LocalTensor<Element> grabL1_[STAGES];
    AscendC::LocalTensor<Element> keyL1_;
    AscendC::LocalTensor<Element> l0A_[STAGES];
    AscendC::LocalTensor<Element> l0B_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0C_[STAGES];
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
    uint32_t consumeStage_{0};
};

}  // namespace Catlass::Kernel::SameAB
