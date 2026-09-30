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
#include "../../regbase/bwd_same_ab/backward_same_ab_fast_silu_grad.hpp"

namespace Catlass::Kernel::SameAB {

// Vector half of Same AB QK/ScoreGrad. Fixpipe places each AIV's M half at
// local UB row 0. A global row offset is only used for source coordinates and
// merging both halves into the shared L1 destination.
template <class ArchTag_, class Element_, class ElementAccumulator_, class TileBuffer_, class L1TileShape_,
          bool HAS_RAB_, bool HAS_MASK_>
class BackwardSameABScoreGrad {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
    using TileBuffer = TileBuffer_;
    using L1TileShape = L1TileShape_;

    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;
    static constexpr bool USE_PHYSICAL_NZ = !HAS_RAB && !HAS_MASK;
    static constexpr uint32_t STAGES = TileBuffer::STAGES;
    static constexpr uint32_t LOGICAL_M = TileBuffer::LOGICAL_M;
    static constexpr uint32_t AIV_M = TileBuffer::AIV_M;
    static constexpr uint32_t ELEM_PER_BLOCK = Catlass::BYTE_PER_C0 / sizeof(Element);

    static_assert(STAGES == 2, "Same AB ScoreGrad requires two score stages");
    static_assert(LOGICAL_M == AIV_M * 2, "Same AB ScoreGrad requires equal M ownership");

    CATLASS_DEVICE BackwardSameABScoreGrad(ElementAccumulator alpha, ElementAccumulator scale,
                                           uint32_t const (&cubeFlag)[STAGES], uint32_t const (&vecFlag)[STAGES],
                                           Arch::Resource<ArchTag>& resource)
        : alpha_(alpha),
          scale_(scale)
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            score_[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::SCORE[i]);
            probDst_[i] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::PROB_DST[i]);
            // 跨核 flag 的 id 由调用方按 stage 显式给出：Same AB 使用模式 0x2，
            // flagId 必须落在 0-15 内，不能再用 i * SYNC_FLAG_ID_MAX 展开。
            cubeReady_[i] = Arch::CrossCoreFlag(cubeFlag[i]);
            vecReady_[i] = Arch::CrossCoreFlag(vecFlag[i]);
        }
        if constexpr (HAS_RAB) {
            rab_ = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB);
        }
        if constexpr (HAS_MASK) {
            mask_ = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::MASK);
        }
        prob_ = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::PROB);
        for (uint32_t i = 0; i < STAGES; ++i) {
            grabPart_[i] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRABPART[i]);
        }
    }

    template <class TensorRab, class Coord, class Shape, class Predictor>
    CATLASS_DEVICE void operator()(TensorRab& tensorRab, Coord const& coord, Shape const& shape, Predictor& predictor)
    {
        uint32_t stage = consumeStage_ % STAGES;
        uint32_t subBlock = AscendC::GetSubBlockIdx();
        uint32_t mReal = tla::get<0>(shape);
        uint32_t nReal = tla::get<1>(shape);
        // This dedicated SameAB kernel aligns the complete M tile before
        // Fixpipe SPLIT_M. Keep that ownership rule for both RowMajor and NZ;
        // it differs from the reference kernel's pre-existing split-first
        // protocol and must not be changed while porting its NZ DMA pattern.
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

        if (processM != 0) {
            // score_/rab_/mask_/prob_/grabPart_ all start from local UB row 0.
            // Only the GM coordinate carries this AIV's global row offset.
            auto localCoord = tla::Add(coord, tla::MakeCoord(0, 0, coreRowOffset, 0));
            auto localShape = tla::MakeShape(localM, nReal);
            uint32_t rows = RoundUp<ELEM_PER_BLOCK>(localM);
            uint32_t cols = RoundUp<ELEM_PER_BLOCK>(nReal);

            if constexpr (HAS_RAB) {
                CopyRab(tensorRab, localCoord, localShape);
            }
            if constexpr (HAS_MASK) {
                predictor.ApplyMask(mask_, localCoord, localShape, rows, cols);
            }
        }

        // RAB/mask do not alias the stage-owned Score input. Prefetch them
        // while Cube is still producing QK, then rendezvous immediately
        // before Vector consumes Score, as in the legacy pipeline. Both AIVs
        // must consume the token even for an empty tail.
        AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(cubeReady_[stage].id);
        if (processM != 0) {
            Compute(stage, mReal, processM, nReal, predictor.needMask, coreRowOffset);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecReady_[stage].id);
        ++consumeStage_;
    }

private:
    CATLASS_DEVICE void CallVectorFunction(uint32_t stage, int64_t offset, uint32_t count, bool needMask)
    {
        uint32_t repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));
        auto scorePtr = (__ubuf__ ElementAccumulator*)score_[stage][offset].GetPhyAddr();
        auto rabPtr = (__ubuf__ Element*)rab_[offset].GetPhyAddr();
        auto maskPtr = (__ubuf__ Element*)mask_[offset].GetPhyAddr();
        auto probPtr = (__ubuf__ Element*)prob_[offset].GetPhyAddr();
        auto grabPartPtr = (__ubuf__ Element*)grabPart_[stage][offset].GetPhyAddr();
        AscendC::VF_CALL<
            catlass::Epilogue::RegBase::SameAB::FastSiluGradVf<Element, ElementAccumulator, Element, HAS_RAB>>(
            scorePtr, rabPtr, maskPtr, probPtr, grabPartPtr, alpha_, scale_, count, repeatTimes, needMask);
    }

    CATLASS_DEVICE void Compute(uint32_t stage, uint32_t logicalM, uint32_t localM, uint32_t nReal, bool needMask,
                                uint32_t coreRowOffset)
    {
        // no-RAB/no-mask follows forward Same AB and consumes the exact
        // Fixpipe-owned half (which may be 8-row aligned rather than 16-row
        // aligned). RAB/mask paths retain their existing vector alignment.
        uint32_t rows = localM;
        if constexpr (HAS_RAB || HAS_MASK) {
            rows = RoundUp<ELEM_PER_BLOCK>(localM);
        }
        uint32_t cols = RoundUp<ELEM_PER_BLOCK>(nReal);
        if constexpr (HAS_RAB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        CallVectorFunction(stage, 0, rows * cols, needMask);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyProbToL1(stage, RoundUp<ELEM_PER_BLOCK>(logicalM), localM, cols, coreRowOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        if constexpr (HAS_RAB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE void CopyProbToL1(uint32_t stage, uint32_t logicalRows, uint32_t localRows, uint32_t cols,
                                     uint32_t coreRowOffset)
    {
        uint32_t colBlocks = cols / ELEM_PER_BLOCK;
        AscendC::DataCopyParams params;
        if constexpr (USE_PHYSICAL_NZ) {
            // The local SPLIT_M half is already contiguous in physical zN
            // order. Copy every column fractal with one strided DMA command;
            // dstStride skips the other AIV's M half in the merged L1 tile.
            params.blockCount = colBlocks;
            params.blockLen = localRows;
            params.srcStride = 0;
            params.dstStride = localRows;
            AscendC::DataCopy(probDst_[stage][coreRowOffset * ELEM_PER_BLOCK], prob_, params);
        } else {
            params.blockCount = localRows;
            params.blockLen = 1;
            params.srcStride = colBlocks - 1;
            params.dstStride = 0;
            for (uint32_t i = 0; i < colBlocks; ++i) {
                uint32_t dstElem = (i * logicalRows + coreRowOffset) * ELEM_PER_BLOCK;
                AscendC::DataCopy(probDst_[stage][dstElem], prob_[i * ELEM_PER_BLOCK], params);
            }
        }
    }

    template <class TensorRab, class Coord, class Shape>
    CATLASS_DEVICE void CopyRab(TensorRab& tensorRab, Coord const& coord, Shape const& shape)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        auto srcOffset = tensorRab.layout()(coord);
        AscendC::DataCopyParams params;
        AscendC::DataCopyPadParams pad;
        params.blockCount = tla::get<0>(shape);
        params.blockLen = tla::get<1>(shape) * sizeof(Element);
        params.srcStride = (tla::get<2>(tensorRab.stride()) - tla::get<1>(shape)) * sizeof(Element);
        params.dstStride = 0;
        pad.isPad = false;
        AscendC::DataCopyPad(rab_, tensorRab.data()[srcOffset], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    Arch::CrossCoreFlag cubeReady_[STAGES];
    Arch::CrossCoreFlag vecReady_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> score_[STAGES];
    AscendC::LocalTensor<Element> rab_;
    AscendC::LocalTensor<Element> mask_;
    AscendC::LocalTensor<Element> prob_;
    AscendC::LocalTensor<Element> grabPart_[STAGES];
    AscendC::LocalTensor<Element> probDst_[STAGES];
    ElementAccumulator alpha_{0};
    ElementAccumulator scale_{0};
    uint32_t consumeStage_{0};
};

}  // namespace Catlass::Kernel::SameAB
