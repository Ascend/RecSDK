/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy_::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy_::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy_::LayoutC;
    using TileCopy = TileCopy_;
    using TileCopyPV = typename TileCopy_::Secondary;
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
    using LayoutTagDST_PV = typename TileCopyPV::LayoutTagC;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    template <class TensorDst>
    using CopyL0CToDstQK_T = typename TileCopy_::template CopyL0CToDst<TensorDst>;

    template <class TensorDst>
    using CopyL0CToDstPV_T = typename TileCopyPV::template CopyL0CToDst<TensorDst>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1_MK_BYTES = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1_NK_BYTES = L1_TILE_N * L1_TILE_K * sizeof(ElementA);

    static constexpr uint32_t L1_Q_OFFSET = 0;
    static constexpr uint32_t L1_K_OFFSET = L1_MK_BYTES;
    static constexpr uint32_t L1_V_OFFSET = L1_MK_BYTES + L1_NK_BYTES;
    static constexpr uint32_t L1_P_OFFSET = L1_MK_BYTES + 2 * L1_NK_BYTES;

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t headNum, uint32_t headDim, uint32_t QKL1Flag,
                 uint32_t PVL1Flag, uint32_t const (&CUBE_CROSS_EVENT_ID_)[5], uint32_t const (&L0_EVENT_ID_)[2],
                 ElementAccumulator alpha_ = 1.0f, ElementAccumulator scale_ = 1.0f)
        : alpha(alpha_),
          scale(scale_)
    {
        l1QTensor = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1Q);
        for (uint32_t i = 0; i < 5; i++) {
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1K[i]);
            l1VTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1V[i]);
        }

        l1PTensor = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1P);

        l0ATensor[0] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[0]);
        l0ATensor[1] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[1]);
        l0BTensor[0] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[0]);
        l0BTensor[1] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[1]);
        l0CTensor[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[0]);
        l0CTensor[1] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[1]);

        AccTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::VACC);

        dstTensor[0] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[0]);
        dstTensor[1] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[1]);
        dstTensor[2] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[2]);
        dstTensor[3] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[3]);
        dstTensor[4] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[4]);
        OutTensor = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::TRANS_OUT);

        this->headNum = headNum;
        this->stride = headNum * headDim;
        L1_QK_EVENT_ID = QKL1Flag;
        L1_PV_EVENT_ID = PVL1Flag;

        cubeReady[0] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[0]);
        cubeReady[1] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[1]);
        cubeReady[2] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[2]);
        cubeReady[3] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[3]);
        cubeReady[4] = Arch::CrossCoreFlag(CUBE_CROSS_EVENT_ID_[4]);

        L0_EVENT_ID[0] = L0_EVENT_ID_[0];
        L0_EVENT_ID[1] = L0_EVENT_ID_[1];
    }

    CATLASS_DEVICE
    ~BlockMmadTla() = default;

    CATLASS_DEVICE void LoadL0Q(uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(mReal, kReal);
        auto tensorL1a = tla::MakeTensor(l1QTensor, l1aLayout, l1Coord, Arch::PositionL1{});
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, L0_TILE_K);
        auto tensorL0a = tla::MakeTensor(l0ATensor[pingpongflag], l0aLayout, l0Coord, Arch::PositionL0A{});
        copyL1ToL0A(tensorL0a, tensorL1a);
    }

    CATLASS_DEVICE void LoadL0K(uint32_t pipeKflag, uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(L0_TILE_N, L0_TILE_K);
        auto tensorL1b = tla::MakeTensor(l1KTensor[pipeKflag], l1bLayout, l1Coord, Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(L0_TILE_N, L0_TILE_K);
        auto tensorL0b = tla::MakeTensor(l0BTensor[pingpongflag], l0bLayout, l0Coord, Arch::PositionL0B{});
        copyL1ToL0B(tensorL0b, tensorL1b);
    }

    CATLASS_DEVICE void LoadL0P(uint32_t pipeSflag, uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1aLayout = tla::MakeLayout<ElementA, typename TileCopyPV::LayoutTagL1A>(mReal, L0_TILE_N);
        auto tensorL1a = tla::MakeTensor(l1PTensor, l1aLayout, l1Coord, Arch::PositionL1{});
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, L0_TILE_N);
        auto tensorL0a = tla::MakeTensor(l0ATensor[pingpongflag], l0aLayout, l0Coord, Arch::PositionL0A{});
        typename TileCopyPV::CopyL1ToL0A{}(tensorL0a, tensorL1a);
    }

    CATLASS_DEVICE void LoadL0V(uint32_t pipeVflag, uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1bLayout = tla::MakeLayout<ElementB, typename TileCopyPV::LayoutTagL1B>(L0_TILE_N, L0_TILE_K);
        auto tensorL1b = tla::MakeTensor(l1VTensor[pipeVflag], l1bLayout, l1Coord, Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(L0_TILE_N, L0_TILE_K);
        auto tensorL0b = tla::MakeTensor(l0BTensor[pingpongflag], l0bLayout, l0Coord, Arch::PositionL0B{});
        typename TileCopyPV::CopyL1ToL0B{}(tensorL0b, tensorL1b);
    }

    template <bool IS_SIDE_A, class TensorSrc>
    CATLASS_DEVICE void LoadGMtoL1(TensorSrc& src)
    {
        using TensorL1 = std::conditional_t<IS_SIDE_A, typename TileCopy_::TensorL1A, typename TileCopy_::TensorL1B>;
        using Element = std::conditional_t<IS_SIDE_A, ElementA, ElementB>;
        using LayoutTag = std::conditional_t<IS_SIDE_A, LayoutTagL1A, LayoutTagL1B>;

        using TileCopy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorSrc, TensorL1>;
        TileCopy tileCopy;

        if constexpr (IS_SIDE_A) {
            mReal = tla::get<0>(src.shape());
            kReal = tla::get<1>(src.shape());
            uint32_t rows = RoundUp<L1AAlignHelper::M_ALIGNED>(mReal);

            auto l1Layout = tla::MakeLayout<Element, LayoutTag>(rows, kReal);
            tileCopy(tla::MakeTensor(l1QTensor, l1Layout, Arch::PositionL1{}), src, mReal, kReal, stride);
        } else {
            nReal = tla::get<0>(src.shape());
            kReal = tla::get<1>(src.shape());

            uint32_t nBlocks = CeilDiv<L0_TILE_N>(nReal);
            auto l1Layout = tla::MakeLayout<Element, LayoutTag>(L0_TILE_N, L0_TILE_K);

            for (uint32_t i = 0; i < nBlocks; i++) {
                uint32_t blockRows = (i == nBlocks - 1) ? nReal - i * L0_TILE_N : L0_TILE_N;
                auto srcCoord = tla::MakeCoord(i * L0_TILE_N * this->headNum, 0);
                auto srcTile = tla::GetTile(src, srcCoord, tla::MakeShape(blockRows, kReal));
                tileCopy(tla::MakeTensor(l1KTensor[i], l1Layout, Arch::PositionL1{}), srcTile, blockRows, kReal,
                         stride);
            }
        }
    }

    CATLASS_DEVICE auto GetCopyL0CToDstQK()
    {
        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(0, 0);
        auto tensorC = tla::MakeTensor(dstTensor[0], dstLayout, Arch::PositionUB{});
        return CopyL0CToDstQK_T<decltype(tensorC)>{};
    }

    CATLASS_DEVICE void FlushL0CToDstQK(uint32_t L0CFlag, uint32_t UBFlag, uint32_t totalM, uint32_t copyM,
                                        uint32_t nSize, uint32_t mOffset, uint32_t subBlockId)
    {
        auto copyL0CToDstQK = GetCopyL0CToDstQK();
        auto l0cLayout = tla::MakeLayoutL0C(totalM, nSize);
        auto tensorL0c =
            tla::MakeTensor(l0CTensor[L0CFlag], l0cLayout, tla::MakeCoord(mOffset, 0), Arch::PositionL0C{});
        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(copyM, nSize);
        auto tensorDst = tla::MakeTensor(dstTensor[UBFlag % 5], dstLayout, tla::MakeCoord(0, 0), Arch::PositionUB{});
        copyL0CToDstQK(tensorDst, tensorL0c, 0, subBlockId);
    }

    CATLASS_DEVICE void FlushL0CToDstQK_SplitM(uint32_t L0CFlag, uint32_t UBFlag, uint32_t mSize, uint32_t nSize)
    {
        using TensorL0C = typename TileCopy::TensorL0C;
        auto coord = tla::MakeCoord(0, 0);
        auto l0cLayout = tla::MakeLayoutL0C(mSize, nSize);
        auto tensorL0c = tla::MakeTensor(l0CTensor[L0CFlag], l0cLayout, coord, Arch::PositionL0C{});

        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(mSize, nSize);
        auto tensorDst = tla::MakeTensor(dstTensor[UBFlag % 5], dstLayout, coord, Arch::PositionUB{});

        using CopySplitM =
            Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, decltype(tensorDst), Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
                                       Tile::ScaleGranularity::NO_QUANT, false>;
        CopySplitM copySplitM;
        copySplitM(tensorDst, tensorL0c, 0);
    }

    CATLASS_DEVICE
    void operator()(uint32_t& L0ApingpongFlag, uint32_t& L0BpingpongFlag, uint32_t& L0CFlag, uint32_t& UBFlag,
                    uint32_t mReal_, uint32_t nReal_, uint32_t kReal_)
    {
        mReal = mReal_;
        nReal = nReal_;
        kReal = kReal_;

        // Note: mRound/kRound kept for potential alignment use, but actual data
        // dimensions (mReal/nReal/kReal) are used for all compute/layout parameters,
        // consistent with the backward implementation pattern.
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mReal);
        uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nReal);
        uint32_t kRound = RoundUp<L1AAlignHelper::K_ALIGNED>(kReal);

        auto nLoop = CeilDiv<L0_TILE_N>(nReal);
        auto nTail = nReal - (nLoop - 1) * L0_TILE_N;
        uint32_t nSize;
        uint32_t nSizeAligned;  // nSize rounded up to C0 block (8 for float32) for Fixpipe compatibility

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0BpingpongFlag);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

        for (auto n = 0; n < nLoop; n++) {
            if (n == nLoop - 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_PV_EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);
                nSize = nTail;
                LoadL0V(0, !L0BpingpongFlag);
                nSizeAligned = RoundUp(nSize, 16);
                // Last iteration: prefetch P/V for PV MMAD, help next MMAD L0C
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE1>(5);

                LoadL0P(0, !L0ApingpongFlag);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(!L0BpingpongFlag);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);
                nSize = L0_TILE_N;
                nSizeAligned = L0_TILE_N;  // already aligned
                // Prefetch next K block
                LoadL0K(n + 1, !L0BpingpongFlag);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(!L0BpingpongFlag);
            }

            auto coord = tla::MakeCoord(0, 0);
            auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, kReal);
            auto tensorL0a = tla::MakeTensor(l0ATensor[L0ApingpongFlag], l0aLayout, coord, Arch::PositionL0A{});

            // L0B layout uses L0_TILE_N (matching LoadL0K output), not nSize.
            // L0C and flush use nSize — MMAD output column count.
            // L0B and L0C N-dimensions differ: L0B=128 (hw write), L0C=nSize (MMAD output).
            // MMAD n parameter bridges them: only nSize rows of L0B are used.
            auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(L0_TILE_N, kReal);
            auto tensorL0b = tla::MakeTensor(l0BTensor[L0BpingpongFlag], l0bLayout, coord, Arch::PositionL0B{});

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0CFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BpingpongFlag);

            // L0C N uses nSizeAligned (C0-block multiple) for Fixpipe compatibility
            auto l0cLayout = tla::MakeLayoutL0C(mReal, nSizeAligned);
            auto tensorL0c = tla::MakeTensor(l0CTensor[L0CFlag], l0cLayout, coord, Arch::PositionL0C{});

            tileMmad(tensorL0c, tensorL0a, tensorL0b, mReal, nSize, kReal, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0CFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BpingpongFlag);

            if (n != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(!L0CFlag);
                FlushL0CToDstQK_SplitM(!L0CFlag, UBFlag, mReal, nSizeAligned);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeReady[UBFlag % 5].id);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(!L0CFlag);
                UBFlag++;
            }

            L0BpingpongFlag = !L0BpingpongFlag;
            L0CFlag = !L0CFlag;
        }

        L0ApingpongFlag = !L0ApingpongFlag;

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BpingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0CFlag);
    }

protected:
    Arch::CrossCoreFlag cubeReady[5];

    AscendC::LocalTensor<ElementA> l1QTensor, l1PTensor;
    AscendC::LocalTensor<ElementB> l1KTensor[5], l1VTensor[5];
    AscendC::LocalTensor<ElementA> l0ATensor[2];
    AscendC::LocalTensor<ElementB> l0BTensor[2];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[2], AccTensor;
    AscendC::LocalTensor<ElementAccumulator> OutTensor, dstTensor[5];

    int64_t stride{0};
    uint32_t headNum{0};
    uint32_t mReal{0};
    uint32_t nReal{0};
    uint32_t kReal;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t L1_QK_EVENT_ID;
    uint32_t L1_PV_EVENT_ID;

    uint32_t L0_EVENT_ID[2];

    ElementAccumulator alpha{1.0f};
    ElementAccumulator scale{1.0f};
};

}  // namespace Catlass::Gemm::Block

#endif
