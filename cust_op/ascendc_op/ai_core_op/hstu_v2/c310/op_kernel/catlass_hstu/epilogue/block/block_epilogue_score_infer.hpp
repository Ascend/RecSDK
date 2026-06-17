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

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "../../../catlass_hstu/epilogue/regbase/fast_silu_infer.hpp"

namespace Catlass::Epilogue::Block {

template <class ArchTag_, class Element_, class ElementAccumulator_, class TileBuffer_, class L0TileShape_,
          bool HAS_RAB_ = false, bool HAS_MASK_ = false>
struct BlockEpilogueScore {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
    using TileBuffer = TileBuffer_;
    using L0TileShape = L0TileShape_;

    static constexpr uint32_t ELEM_PER_BLOCK = Catlass::BYTE_PER_C0 / sizeof(Element);

    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});

    CATLASS_DEVICE
    BlockEpilogueScore(uint32_t const (&QK_CROSS_EVENT_ID_)[5], uint32_t const (&PV_CROSS_EVENT_ID_)[5],
                       Arch::Resource<ArchTag>& resource, ElementAccumulator alpha_ = 1.0f,
                       ElementAccumulator scale_ = 1.0f)
    {
        alpha = alpha_;
        scale = scale_;
        ubScoreTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::SCORE);
        ubProbTensor[0] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::PROB[0]);
        ubProbTensor[1] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::PROB[1]);
        ubProbTensor[2] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::PROB[2]);
        ubProbTensor[3] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::PROB[3]);
        ubProbTensor[4] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::PROB[4]);

        ubRabTensor[0] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB[0]);
        ubRabTensor[1] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB[1]);

        dstTensor[0] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::DST[0]);
        dstTensor[1] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::DST[1]);
        dstTensor[2] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::DST[2]);
        dstTensor[3] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::DST[3]);
        dstTensor[4] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::DST[4]);

        QKcubeReady[0] = Arch::CrossCoreFlag(QK_CROSS_EVENT_ID_[0]);
        QKcubeReady[1] = Arch::CrossCoreFlag(QK_CROSS_EVENT_ID_[1]);
        QKcubeReady[2] = Arch::CrossCoreFlag(QK_CROSS_EVENT_ID_[2]);
        QKcubeReady[3] = Arch::CrossCoreFlag(QK_CROSS_EVENT_ID_[3]);
        QKcubeReady[4] = Arch::CrossCoreFlag(QK_CROSS_EVENT_ID_[4]);

        PVcubeReady[0] = Arch::CrossCoreFlag(PV_CROSS_EVENT_ID_[0]);
        PVcubeReady[1] = Arch::CrossCoreFlag(PV_CROSS_EVENT_ID_[1]);
        PVcubeReady[2] = Arch::CrossCoreFlag(PV_CROSS_EVENT_ID_[2]);
        PVcubeReady[3] = Arch::CrossCoreFlag(PV_CROSS_EVENT_ID_[3]);
        PVcubeReady[4] = Arch::CrossCoreFlag(PV_CROSS_EVENT_ID_[4]);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    template <class TensorSrc, class Coord, class Shape>
    CATLASS_DEVICE void operator()(TensorSrc& tensorRab, Coord const& coord, Shape const& shape)
    {
        auto mReal = RoundUp(tla::get<0>(shape), 2) / 2;
        auto nReal = tla::get<1>(shape);

        auto nLoop = CeilDiv<L0_TILE_N>(nReal);

        auto nTail = nReal - (nLoop - 1) * L0_TILE_N;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);

        if constexpr (HAS_RAB) {
            // Copy first K strip: all Q rows × first K tile cols
            auto firstN = (nLoop == 1) ? nReal : L0_TILE_N;
            auto firstTileShape = tla::MakeShape(mReal, firstN);
            CopyRab(tensorRab, coord, firstTileShape, 0);
        }

        for (auto n = 0; n < nLoop; n++) {
            if constexpr (HAS_RAB) {
                if (n < nLoop - 1) {
                    // Preload next K strip: advance seqK coord (dim 3), not seqQ (dim 2)
                    auto nextN = (n + 1 == nLoop - 1) ? nTail : L0_TILE_N;
                    auto kOffset = (n + 1) * L0_TILE_N;
                    auto coordNew = tla::Add(coord, tla::MakeCoord(0, 0, 0, kOffset));
                    auto nextTileShape = tla::MakeShape(mReal, nextN);
                    CopyRab(tensorRab, coordNew, nextTileShape, !(n % 2));
                }
            }

            auto nSize = (n == nLoop - 1) ? nTail : L0_TILE_N;
            AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(QKcubeReady[UBFlag % 5].id);

            ComputeBlock(mReal, nSize, n);

            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(PVcubeReady[n].id);
            UBFlag++;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((nLoop - 1) % 2);
    }

    CATLASS_DEVICE void ComputeBlock(uint32_t mReal, uint32_t nSize, uint32_t blockIdx)
    {
        auto count = RoundUp<ELEM_PER_BLOCK>(mReal) * RoundUp<ELEM_PER_BLOCK>(nSize);

        auto repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));

        auto ubSPtr = (__ubuf__ Element*)ubScoreTensor.GetPhyAddr();
        auto ubRabPtr = (__ubuf__ Element*)ubRabTensor[UBFlag % 2].GetPhyAddr();
        auto ubMaskPtr = (__ubuf__ Element*)ubRabTensor[UBFlag % 2].GetPhyAddr();
        auto ubProbPtr = (__ubuf__ ElementAccumulator*)ubProbTensor[UBFlag % 5].GetPhyAddr();

        if constexpr (HAS_RAB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(blockIdx % 2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        AscendC::VF_CALL<catlass::Epilogue::RegBase::FastSiluScoreVf<Element, ElementAccumulator, HAS_RAB, HAS_MASK>>(
            ubProbPtr, ubRabPtr, ubMaskPtr, ubSPtr, alpha, scale, count, repeatTimes);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        CopyToDst(mReal, nSize, count, blockIdx % 5);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        if constexpr (HAS_RAB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(blockIdx % 2);
        }
    }

    CATLASS_DEVICE void CopyToDst(uint32_t mReal, uint32_t nSize, uint32_t count, uint32_t blockIdx)
    {
        if constexpr (HAS_RAB || HAS_MASK) {
            auto coreId = AscendC::GetSubBlockIdx();
            auto totalM = mReal * 2;  // SPLIT_M=2, full M dim
            auto coreRowOffset = coreId * mReal * ELEM_PER_BLOCK;

            auto colBlk = CeilDiv<ELEM_PER_BLOCK>(nSize);

            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = mReal;
            intriParams.blockLen = 1;
            intriParams.srcStride = (colBlk - 1);
            intriParams.dstStride = 0;

            auto totalMAligned = RoundUp<ELEM_PER_BLOCK>(totalM);
            for (auto i = 0; i < colBlk; i++) {
                auto dstBase = i * totalMAligned * ELEM_PER_BLOCK + coreRowOffset;
                AscendC::DataCopy(dstTensor[blockIdx][dstBase], ubScoreTensor[i * ELEM_PER_BLOCK], intriParams);
            }
        } else {
            auto coreId = AscendC::GetSubBlockIdx();
            auto coreOffset = coreId * mReal * nSize;
            AscendC::DataCopy(dstTensor[blockIdx][coreOffset], ubScoreTensor, count);
        }
    }

    template <class TensorSrc, class Coord, class Shape>
    CATLASS_DEVICE void CopyRab(TensorSrc& tensorRab, Coord const& coord, Shape const& shape, uint32_t blockIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(blockIdx);

        auto srcOffset = tensorRab.layout()(coord);

        AscendC::DataCopyParams intriParams;
        AscendC::DataCopyPadParams padParams;
        intriParams.blockCount = tla::get<0>(shape);
        intriParams.blockLen = tla::get<1>(shape) * sizeof(Element);
        intriParams.srcStride = (tla::get<2>(tensorRab.stride()) - tla::get<1>(shape)) * sizeof(Element);
        intriParams.dstStride = 0;

        padParams.isPad = false;
        AscendC::DataCopyPad(ubRabTensor[blockIdx], tensorRab.data()[srcOffset], intriParams, padParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(blockIdx);
    }

private:
    Arch::CrossCoreFlag QKcubeReady[5];
    Arch::CrossCoreFlag PVcubeReady[5];

    AscendC::LocalTensor<Element> ubScoreTensor;
    AscendC::LocalTensor<Element> ubRabTensor[2];
    AscendC::LocalTensor<Element> ubMaskTensor[2];
    AscendC::LocalTensor<ElementAccumulator> ubProbTensor[5];
    AscendC::LocalTensor<Element> dstTensor[5];

    ElementAccumulator alpha{1.0f};
    ElementAccumulator scale{1.0f};

    uint32_t UBFlag{0};
};

}  // namespace Catlass::Epilogue::Block
