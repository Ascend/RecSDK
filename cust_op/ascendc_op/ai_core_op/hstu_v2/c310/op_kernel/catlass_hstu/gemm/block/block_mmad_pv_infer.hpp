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

#ifndef HSTU_GEMM_BLOCK_BLOCK_MMAD_PV_INFER_HPP
#define HSTU_GEMM_BLOCK_BLOCK_MMAD_PV_INFER_HPP

#include "../../../catlass_hstu/gemm/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Gemm::Block {

template <class ArchTag_, bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, bool BIND_SUB_CORE_, uint32_t SUB_CORE_ID_,
          class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class TileBuffer_,
          class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadHSTUPV<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, BIND_SUB_CORE_, SUB_CORE_ID_>,
                    L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, TileBuffer_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadHSTUPV<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, BIND_SUB_CORE_, SUB_CORE_ID_>;
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
    using TileMmad = TileMmad_;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using TileBuffer = TileBuffer_;

    using TileCopyQK = typename TileCopy_::Secondary;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;
    using LayoutTagDST_QK = typename TileCopyQK::LayoutTagC;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    template <class TensorDst>
    using CopyL0CToDstPV_T = typename TileCopy_::template CopyL0CToDst<TensorDst>;

    template <class TensorDst>
    using CopyL0CToDstQK_T = typename TileCopyQK::template CopyL0CToDst<TensorDst>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t headNum, uint32_t headDim, uint32_t QKL1Flag,
                 uint32_t SVL1Flag, uint32_t const (&CUBE_CROSS_EVENT_ID_)[5], uint32_t const (&L0_EVENT_ID_)[2],
                 ElementAccumulator alpha_ = 1.0f, ElementAccumulator scale_ = 1.0f)
        : alpha(alpha_),
          scale(scale_)
    {
        l1QTensor = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1Q);
        for (uint32_t i = 0; i < 5; i++) {
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1K[i]);
            l1VTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1V[i]);
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1P[i]);
        }

        l0ATensor[0] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[0]);
        l0ATensor[1] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[1]);
        l0BTensor[0] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[0]);
        l0BTensor[1] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[1]);
        l0CTensor[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[0]);
        l0CTensor[1] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[1]);

        dstTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::VACC);

        dstUBTensor[0] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[0]);
        dstUBTensor[1] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[1]);
        dstUBTensor[2] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[2]);
        dstUBTensor[3] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[3]);
        dstUBTensor[4] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST[4]);
        OutTensor = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::TRANS_OUT);

        this->headNum = headNum;
        this->stride = headNum * headDim;
        L1_QK_EVENT_ID = QKL1Flag;
        L1_PV_EVENT_ID = SVL1Flag;

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
        auto l1aLayout = tla::MakeLayout<ElementA, typename TileCopyQK::LayoutTagL1A>(L0_TILE_M, L0_TILE_K);
        auto tensorL1a = tla::MakeTensor(l1QTensor, l1aLayout, l1Coord, Arch::PositionL1{});
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(L0_TILE_M, L0_TILE_K);
        auto tensorL0a = tla::MakeTensor(l0ATensor[pingpongflag], l0aLayout, l0Coord, Arch::PositionL0A{});
        typename TileCopyQK::CopyL1ToL0A{}(tensorL0a, tensorL1a);
    }

    CATLASS_DEVICE void LoadL0K(uint32_t pipeKflag, uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1bLayout = tla::MakeLayout<ElementB, typename TileCopyQK::LayoutTagL1B>(L0_TILE_N, L0_TILE_K);
        auto tensorL1b = tla::MakeTensor(l1KTensor[pipeKflag], l1bLayout, l1Coord, Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(L0_TILE_N, L0_TILE_K);
        auto tensorL0b = tla::MakeTensor(l0BTensor[pingpongflag], l0bLayout, l0Coord, Arch::PositionL0B{});
        typename TileCopyQK::CopyL1ToL0B{}(tensorL0b, tensorL1b);
    }

    CATLASS_DEVICE void LoadL0P(uint32_t pipeSflag, uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(mReal, L0_TILE_N);
        auto tensorL1a = tla::MakeTensor(l1PTensor[pipeSflag], l1aLayout, l1Coord, Arch::PositionL1{});
        auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, L0_TILE_N);
        auto tensorL0a = tla::MakeTensor(l0ATensor[pingpongflag], l0aLayout, l0Coord, Arch::PositionL0A{});
        copyL1ToL0A(tensorL0a, tensorL1a);
    }

    CATLASS_DEVICE void LoadL0V(uint32_t pipeVflag, uint32_t pingpongflag)
    {
        auto l1Coord = tla::MakeCoord(0, 0);
        auto l0Coord = tla::MakeCoord(0, 0);
        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(L0_TILE_N, L0_TILE_K);
        auto tensorL1b = tla::MakeTensor(l1VTensor[pipeVflag], l1bLayout, l1Coord, Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(L0_TILE_N, L0_TILE_K);
        auto tensorL0b = tla::MakeTensor(l0BTensor[pingpongflag], l0bLayout, l0Coord, Arch::PositionL0B{});
        copyL1ToL0B(tensorL0b, tensorL1b);
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

            uint32_t mBlocks = CeilDiv<L0_TILE_M>(mReal);
            auto l1Layout = tla::MakeLayout<Element, LayoutTag>(L0_TILE_M, L0_TILE_N);
            for (uint32_t i = 0; i < mBlocks; i++) {
                uint32_t blockRows = (i == mBlocks - 1) ? mReal - i * L0_TILE_M : L0_TILE_M;
                auto srcCoord = tla::MakeCoord(i * L0_TILE_M * this->headNum, 0);
                auto srcTile = tla::GetTile(src, srcCoord, tla::MakeShape(blockRows, kReal));
                tileCopy(tla::MakeTensor(l1PTensor[i], l1Layout, Arch::PositionL1{}), srcTile, blockRows, kReal,
                         stride);
            }
        } else {
            nReal = tla::get<0>(src.shape());
            kReal = tla::get<1>(src.shape());

            uint32_t nBlocks = CeilDiv<L0_TILE_N>(nReal);
            auto l1Layout = tla::MakeLayout<Element, LayoutTag>(L0_TILE_N, L0_TILE_K);
            for (uint32_t i = 0; i < nBlocks; i++) {
                uint32_t blockRows = (i == nBlocks - 1) ? nReal - i * L0_TILE_N : L0_TILE_N;
                auto srcCoord = tla::MakeCoord(i * L0_TILE_N * this->headNum, 0);
                auto srcTile = tla::GetTile(src, srcCoord, tla::MakeShape(blockRows, kReal));
                tileCopy(tla::MakeTensor(l1VTensor[i], l1Layout, Arch::PositionL1{}), srcTile, blockRows, kReal,
                         stride);
            }
        }
    }

    CATLASS_DEVICE void FlushL0CToDstQK_SplitM(uint32_t L0CFlag, uint32_t UBFlag, uint32_t mSize, uint32_t nSize)
    {
        using TensorL0C = typename TileCopy::TensorL0C;
        auto coord = tla::MakeCoord(0, 0);
        auto l0cLayout = tla::MakeLayoutL0C(mSize, nSize);
        auto tensorL0c = tla::MakeTensor(l0CTensor[L0CFlag], l0cLayout, coord, Arch::PositionL0C{});

        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(mSize, nSize);
        auto tensorDst = tla::MakeTensor(dstUBTensor[UBFlag % 5], dstLayout, coord, Arch::PositionUB{});

        using CopySplitM =
            Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, decltype(tensorDst), Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
                                       Tile::ScaleGranularity::NO_QUANT, false>;
        CopySplitM copySplitM;
        copySplitM(tensorDst, tensorL0c, 0);
    }

    CATLASS_DEVICE void FlushL0CToDstPV(uint32_t mSize, uint32_t nSize)
    {
        using TensorL0C = typename TileCopy::TensorL0C;
        auto coord = tla::MakeCoord(0, 0);
        auto l0cLayout = tla::MakeLayoutL0C(mSize, nSize);
        auto tensorL0c = tla::MakeTensor(dstTensor, l0cLayout, coord, Arch::PositionL0C{});
        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(mSize, nSize);
        auto tensorDst = tla::MakeTensor(OutTensor, dstLayout, coord, Arch::PositionUB{});

        using CopySplitM =
            Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, decltype(tensorDst), Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
                                       Tile::ScaleGranularity::NO_QUANT, false>;
        CopySplitM copySplitM;
        copySplitM(tensorDst, tensorL0c, 0);

        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(10);
    }

    CATLASS_DEVICE
    void operator()(uint32_t& L0ApingpongFlag, uint32_t& L0BpingpongFlag, uint32_t& L0CFlag, uint32_t& UBFlag,
                    bool isInit = false, bool isFlush = false, uint32_t mReal_ = 0, uint32_t nReal_ = 0,
                    uint32_t kReal_ = 0)
    {
        mReal = mReal_;
        nReal = nReal_;
        kReal = kReal_;

        auto mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mReal);
        auto nLoop = CeilDiv<L0_TILE_N>(nReal);
        auto nTail = nReal - (nLoop - 1) * L0_TILE_N;
        uint32_t nSize;

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0BpingpongFlag);

        for (auto n = 0; n < nLoop; n++) {
            if (n == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(!L0CFlag);

                FlushL0CToDstQK_SplitM(!L0CFlag, UBFlag, mReal, RoundUp<16>(nTail));
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(!L0CFlag);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(!L0CFlag);

                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeReady[nLoop - 1].id - 5);
                UBFlag++;
            }

            if (n == nLoop - 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_QK_EVENT_ID);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);

                nSize = nTail;
                // Last iteration: prefetch Q, K for next QK MMAD, help next MMAD L0C
                LoadL0Q(!L0ApingpongFlag);
                LoadL0K(0, !L0BpingpongFlag);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(!L0BpingpongFlag);

            } else {
                // Prefetch next P and V blocks
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);

                nSize = L0_TILE_N;
                LoadL0V(n + 1, !L0BpingpongFlag);
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE1>(cubeReady[n + 1].id);
                LoadL0P(n + 1, !L0ApingpongFlag);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(!L0BpingpongFlag);
            }

            auto l0Coord = tla::MakeCoord(0, 0);

            auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mReal, L0_TILE_N);
            auto tensorL0a = tla::MakeTensor(l0ATensor[L0ApingpongFlag], l0aLayout, l0Coord, Arch::PositionL0A{});

            auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(L0_TILE_N, kReal);
            auto tensorL0b = tla::MakeTensor(l0BTensor[L0BpingpongFlag], l0bLayout, l0Coord, Arch::PositionL0B{});

            auto l0cLayout = tla::MakeLayoutL0C(mReal, kReal);
            auto tensorL0c = tla::MakeTensor(dstTensor, l0cLayout, l0Coord, Arch::PositionL0C{});

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BpingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);

            tileMmad(tensorL0c, tensorL0a, tensorL0b, mReal, kReal, RoundUp<16>(nSize), isInit, 0);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BpingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID2);

            if (n == nLoop - 1 && isFlush) {
                // PV output: M=mReal (Q seqlen), N=kReal (V head dim)
                FlushL0CToDstPV(mReal, kReal);
            }

            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);

            L0ApingpongFlag = !L0ApingpongFlag;
            L0BpingpongFlag = !L0BpingpongFlag;
            isInit = false;
        }

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(!L0BpingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BpingpongFlag);
    }

protected:
    Arch::CrossCoreFlag cubeReady[5];

    AscendC::LocalTensor<ElementA> l1QTensor;
    AscendC::LocalTensor<ElementA> l1PTensor[5];
    AscendC::LocalTensor<ElementB> l1KTensor[5], l1VTensor[5];
    AscendC::LocalTensor<ElementA> l0ATensor[2];
    AscendC::LocalTensor<ElementB> l0BTensor[2];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[2], dstTensor;
    AscendC::LocalTensor<ElementAccumulator> OutTensor, dstUBTensor[5];

    int64_t stride{0};
    uint32_t headNum{0};
    uint32_t mReal;
    uint32_t nReal;
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
