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
#include "catlass/arch/resource.hpp"
#include "../../regbase/fast_rab_grad.hpp"

namespace Catlass::Kernel::SameAB {

template <class ScoreGrad_, class TileBuffer_>
class BackwardSameABRabGrad {
public:
    using ScoreGrad = ScoreGrad_;
    using ArchTag = typename ScoreGrad::ArchTag;
    using Element = typename ScoreGrad::Element;
    using ElementAccumulator = typename ScoreGrad::ElementAccumulator;
    using TileBuffer = TileBuffer_;

    static constexpr bool HAS_RAB = ScoreGrad::HAS_RAB;
    static constexpr bool HAS_MASK = ScoreGrad::HAS_MASK;
    static constexpr bool USE_PHYSICAL_NZ = ScoreGrad::USE_PHYSICAL_NZ;
    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t LOGICAL_M = TileBuffer::LOGICAL_M;
    static constexpr uint32_t AIV_M = TileBuffer::AIV_M;
    static constexpr uint32_t ELEM_PER_BLOCK = Catlass::BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE BackwardSameABRabGrad(uint32_t const (&cubeFlag)[STAGES], uint32_t const (&vecFlag)[STAGES],
                                         Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            gs_[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::GS[i]);
            grabPart_[i] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRABPART[i]);
            grabDst_[i] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::GRAB_DST[i]);
            // 跨核 flag 的 id 由调用方按 stage 显式给出：Same AB 使用模式 0x2，
            // flagId 必须落在 0-15 内，不能再用 i * SYNC_FLAG_ID_MAX 展开。
            cubeReady_[i] = Arch::CrossCoreFlag(cubeFlag[i]);
            vecReady_[i] = Arch::CrossCoreFlag(vecFlag[i]);
        }
        grab_ = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRAB);
    }

    // `scale * alpha` used to be performed by legacy PER_TENSOR Fixpipe.
    // Same AB keeps Fixpipe NO_QUANT, so its dedicated mainloop sets this
    // factor once on the AIV before consuming GV tiles.
    CATLASS_DEVICE void SetGvScale(ElementAccumulator scale)
    {
        gvScale_ = scale;
    }

    template <class TensorGrab, class Coord, class Shape>
    CATLASS_DEVICE void operator()(TensorGrab& tensorGrab, Coord const& coord, Shape const& shape)
    {
        uint32_t stage = consumeStage_ % STAGES;
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        uint32_t mReal = tla::get<0>(shape);
        uint32_t nReal = tla::get<1>(shape);
        // Match this kernel's align-full-M-then-split Fixpipe contract for
        // both RowMajor and physical NZ.
        uint32_t halfRows = RoundUp<ELEM_PER_BLOCK>(mReal) / 2;
        uint32_t coreRowOffset = subBlock * halfRows;
        uint32_t localM = 0;
        if (subBlock == 0) {
            localM = mReal < halfRows ? mReal : halfRows;
        } else if (mReal > halfRows) {
            localM = mReal - halfRows;
        }
        uint32_t processM = localM;
        if constexpr (USE_PHYSICAL_NZ) {
            processM = halfRows;
        }

        AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(cubeReady_[stage].id);
        if (processM != 0) {
            uint32_t rows = processM;
            if constexpr (HAS_RAB || HAS_MASK) {
                rows = RoundUp<ELEM_PER_BLOCK>(processM);
            }
            uint32_t cols = RoundUp<ELEM_PER_BLOCK>(nReal);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            Compute(stage, rows * cols);
            CopyGrabToL1(stage, RoundUp<ELEM_PER_BLOCK>(mReal), processM, cols, coreRowOffset);

            // dK/dQ only consume the merged L1 Grab tile. Publish readiness
            // as soon as that MTE3 copy has been issued, matching the legacy
            // path, so Cube can overlap its work with the independent dRAB GM
            // writeback below. PIPE_MTE3 preserves ordering after the L1 copy.
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecReady_[stage].id);

            if constexpr (HAS_RAB) {
                auto localCoord = tla::Add(coord, tla::MakeCoord(0, 0, coreRowOffset, 0));
                CopyOutGrab(tensorGrab, localCoord, tla::MakeShape(localM, nReal));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        } else {
            // Keep both AIVs' cross-core counters balanced for empty tails.
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecReady_[stage].id);
        }
        ++consumeStage_;
    }

private:
    CATLASS_DEVICE void Compute(uint32_t stage, uint32_t count)
    {
        uint32_t repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));
        auto gsPtr = (__ubuf__ ElementAccumulator*)gs_[stage].GetPhyAddr();
        auto grabPartPtr = (__ubuf__ Element*)grabPart_[stage].GetPhyAddr();
        auto grabPtr = (__ubuf__ Element*)grab_.GetPhyAddr();
        AscendC::VF_CALL<catlass::Epilogue::RegBase::FastRabGradScaledVf<Element, ElementAccumulator, Element>>(
            gsPtr, grabPartPtr, grabPtr, gvScale_, count, repeatTimes);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }

    CATLASS_DEVICE void CopyGrabToL1(uint32_t stage, uint32_t logicalRows, uint32_t localRows, uint32_t cols,
                                     uint32_t coreRowOffset)
    {
        uint32_t colBlocks = cols / ELEM_PER_BLOCK;
        AscendC::DataCopyParams params;
        if constexpr (USE_PHYSICAL_NZ) {
            params.blockCount = colBlocks;
            params.blockLen = localRows;
            params.srcStride = 0;
            params.dstStride = localRows;
            AscendC::DataCopy(grabDst_[stage][coreRowOffset * ELEM_PER_BLOCK], grab_, params);
        } else {
            params.blockCount = localRows;
            params.blockLen = 1;
            params.srcStride = colBlocks - 1;
            params.dstStride = 0;
            for (uint32_t i = 0; i < colBlocks; ++i) {
                uint32_t dstElem = (i * logicalRows + coreRowOffset) * ELEM_PER_BLOCK;
                AscendC::DataCopy(grabDst_[stage][dstElem], grab_[i * ELEM_PER_BLOCK], params);
            }
        }
    }

    template <class TensorGrab, class Coord, class Shape>
    CATLASS_DEVICE void CopyOutGrab(TensorGrab& tensorGrab, Coord const& coord, Shape const& shape)
    {
        auto dstOffset = tensorGrab.layout()(coord);
        AscendC::DataCopyParams params;
        params.blockCount = tla::get<0>(shape);
        params.blockLen = tla::get<1>(shape) * sizeof(Element);
        params.srcStride = 0;
        params.dstStride = (tla::get<2>(tensorGrab.stride()) - tla::get<1>(shape)) * sizeof(Element);
        AscendC::DataCopyPad(tensorGrab.data()[dstOffset], grab_, params);
    }

    Arch::CrossCoreFlag cubeReady_[STAGES];
    Arch::CrossCoreFlag vecReady_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> gs_[STAGES];
    AscendC::LocalTensor<Element> grabPart_[STAGES];
    AscendC::LocalTensor<Element> grab_;
    AscendC::LocalTensor<Element> grabDst_[STAGES];
    ElementAccumulator gvScale_{1};
    uint32_t consumeStage_{0};
};

}  // namespace Catlass::Kernel::SameAB
