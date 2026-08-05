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

#ifndef HSTU_GEMM_BLOCK_BLOCK_MMAD_QK_HPP
#define HSTU_GEMM_BLOCK_BLOCK_MMAD_QK_HPP

#include "../../../catlass_hstu/gemm/dispatch_policy.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Gemm::Block {

template <class ArchTag_, bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_, class ElementC_, class TileBuffer_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadHSTUQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_, ElementA_,
                    ElementB_, ElementC_, TileBuffer_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadHSTUQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using TileBuffer = TileBuffer_;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    static constexpr uint32_t PROB_STAGES = TileBuffer::STAGES;

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t headNum, uint32_t headDim,
                 uint32_t const (&K_L1_EVENT_ID_)[2], uint32_t const (&Q_L1_EVENT_ID_)[2],
                 uint32_t const (&CUBE_CROSS_EVENT_ID_)[PROB_STAGES], uint32_t L0_HANDOFF_ID_, uint32_t L0C_FIX_ID_)
    {
        for (uint32_t i = 0; i < 2; i++) {
            l1QTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1Q[i]);
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1K[i]);
        }

        l0ATensor = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[0]);
        l0BTensor = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[0]);
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[0]);

        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            dstTensor[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[i]);
        }

        this->stride = headNum * headDim;
        L1_K_EVENT_ID[0] = K_L1_EVENT_ID_[0];
        L1_K_EVENT_ID[1] = K_L1_EVENT_ID_[1];
        L1_Q_EVENT_ID[0] = Q_L1_EVENT_ID_[0];
        L1_Q_EVENT_ID[1] = Q_L1_EVENT_ID_[1];

        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            cubeReady[i] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[i]);
        }

        L0_HANDOFF_ID = L0_HANDOFF_ID_;
        L0C_FIX_ID = L0C_FIX_ID_;
    }

    CATLASS_DEVICE
    ~BlockMmadTla() = default;

    template <bool IS_SIDE_A, class TensorSrc>
    CATLASS_DEVICE void LoadGMtoL1(TensorSrc& src, bool reload = true)
    {
        LoadGMtoL1Impl(std::bool_constant<IS_SIDE_A>{}, src, reload);
    }

private:
    template <class TensorSrc>
    CATLASS_DEVICE void LoadGMtoL1Impl(std::true_type, TensorSrc& src, bool reload)
    {
        if (reload) {
            auto mReal = tla::get<0>(src.shape());
            auto kReal = tla::get<1>(src.shape());
            uint32_t rows = RoundUp<L1AAlignHelper::M_ALIGNED>(mReal);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_Q_EVENT_ID[qW]);
            auto l1Layout = tla::MakeLayout<ElementA, LayoutTagL1A>(rows, kReal);
            using Copy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorSrc, typename TileCopy_::TensorL1A>;
            Copy{}(tla::MakeTensor(l1QTensor[qW], l1Layout, Arch::PositionL1{}), src, mReal, kReal, stride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_Q_EVENT_ID[qW]);
            qW ^= 1;
            if (!qFirst)
                qPending = true;
            qFirst = false;
        }
    }

    template <class TensorSrc>
    CATLASS_DEVICE void LoadGMtoL1Impl(std::false_type, TensorSrc& src, bool /*reload*/)
    {
        auto nReal = tla::get<0>(src.shape());
        auto kReal = tla::get<1>(src.shape());
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_K_EVENT_ID[kW]);
        auto l1Layout = tla::MakeLayout<ElementB, LayoutTagL1B>(nReal, kReal);
        using Copy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorSrc, typename TileCopy_::TensorL1B>;
        Copy{}(tla::MakeTensor(l1KTensor[kW], l1Layout, Arch::PositionL1{}), src, nReal, kReal, stride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_K_EVENT_ID[kW]);
        kW ^= 1;
    }

public:
    CATLASS_DEVICE void LoadL0Q(uint32_t mReal, uint32_t kReal)
    {
        if (!qReady) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_Q_EVENT_ID[qR]);
            qReady = true;
        }
        auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(mReal, kReal);
        auto tensorL1a = tla::MakeTensor(l1QTensor[qR], l1aLayout, tla::MakeCoord(0, 0), Arch::PositionL1{});
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, kReal);
        auto tensorL0a = tla::MakeTensor(l0ATensor, l0aLayout, tla::MakeCoord(0, 0), Arch::PositionL0A{});
        copyL1ToL0A(tensorL0a, tensorL1a);
    }

    CATLASS_DEVICE void LoadL0K(uint32_t nReal, uint32_t kReal)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_K_EVENT_ID[kR]);
        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(nReal, kReal);
        auto tensorL1b = tla::MakeTensor(l1KTensor[kR], l1bLayout, tla::MakeCoord(0, 0), Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(nReal, kReal);
        auto tensorL0b = tla::MakeTensor(l0BTensor, l0bLayout, tla::MakeCoord(0, 0), Arch::PositionL0B{});
        copyL1ToL0B(tensorL0b, tensorL1b);
    }

    CATLASS_DEVICE void LoadL0KRelease()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_K_EVENT_ID[kR]);
        kR ^= 1;
    }

    CATLASS_DEVICE void LoadL0QRelease()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_Q_EVENT_ID[qR]);
        if (qPending) {
            qR ^= 1;
            qPending = false;
            qReady = false;
        }
    }

    template <class Coord>
    CATLASS_DEVICE void LoadL0(Coord const& mn, uint32_t kReal, bool isLast = false)
    {
        uint32_t mReal = tla::get<0>(mn), nReal = tla::get<1>(mn);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID);

        LoadL0Q(mReal, kReal);
        LoadL0K(nReal, kReal);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_HANDOFF_ID);

        LoadL0KRelease();
        if (isLast) {
            LoadL0QRelease();
        }
    }

    template <class Coord>
    CATLASS_DEVICE void operator()(Coord const& mn, uint32_t kReal)
    {
        uint32_t mReal = tla::get<0>(mn), nReal = tla::get<1>(mn);
        uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nReal);

        auto coord = tla::MakeCoord(0, 0);
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, kReal);
        auto tensorL0a = tla::MakeTensor(l0ATensor, l0aLayout, coord, Arch::PositionL0A{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(nReal, kReal);
        auto tensorL0b = tla::MakeTensor(l0BTensor, l0bLayout, coord, Arch::PositionL0B{});
        auto l0cLayout = tla::MakeLayoutL0C(mReal, nRound);
        auto tensorL0c = tla::MakeTensor(l0CTensor, l0cLayout, coord, Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_HANDOFF_ID);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_ID);
        tileMmad(tensorL0c, tensorL0a, tensorL0b, mReal, nReal, kReal, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_FIX_ID);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID);

        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_FIX_ID);
        FlushL0CToDstQK_SplitM(mReal, nRound);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_ID);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeReady[UBflag % PROB_STAGES].id);

        UBflag++;
    }

private:
    CATLASS_DEVICE void FlushL0CToDstQK_SplitM(uint32_t mReal, uint32_t nSize)
    {
        mReal = AlignDstM(mReal);
        using TensorL0C = typename TileCopy::TensorL0C;
        auto coord = tla::MakeCoord(0, 0);
        auto l0cLayout = tla::MakeLayoutL0C(mReal, nSize);
        auto tensorL0c = tla::MakeTensor(l0CTensor, l0cLayout, coord, Arch::PositionL0C{});
        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(mReal, nSize);
        auto tensorDst = tla::MakeTensor(dstTensor[UBflag % PROB_STAGES], dstLayout, coord, Arch::PositionUB{});

        using CopySplitM =
            Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, decltype(tensorDst), Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
                                       Tile::ScaleGranularity::NO_QUANT, false>;
        CopySplitM{}(tensorDst, tensorL0c, 0);
    }

    static constexpr uint32_t AlignDstM(uint32_t m)
    {
        if constexpr (std::is_same_v<LayoutTagDST, layout::zN>) {
            return RoundUp<L1AAlignHelper::M_ALIGNED>(m);
        }
        return m;
    }

protected:
    Arch::CrossCoreFlag cubeReady[PROB_STAGES];

    AscendC::LocalTensor<ElementA> l1QTensor[2];
    AscendC::LocalTensor<ElementB> l1KTensor[2];
    AscendC::LocalTensor<ElementA> l0ATensor;
    AscendC::LocalTensor<ElementB> l0BTensor;
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    AscendC::LocalTensor<ElementAccumulator> dstTensor[PROB_STAGES];

    int64_t stride{0};

    // Internal pingpong state
    uint32_t kW = 0, kR = 0;
    uint32_t qW = 0, qR = 0;
    bool qPending = false;
    bool qFirst = true;
    bool qReady = false;
    uint32_t UBflag = 0;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t L1_Q_EVENT_ID[2];
    uint32_t L1_K_EVENT_ID[2];
    uint32_t L0_HANDOFF_ID;
    uint32_t L0C_FIX_ID;
};

}  // namespace Catlass::Gemm::Block

#endif
