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
    static constexpr uint32_t PROB_STAGES = TileBuffer::STAGES;  // Pipeline depth for prob tensors

    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;
    static constexpr uint32_t SCORE_BUF_CNT = HAS_RAB ? 2 : 3;  // UB score pingpong depth: 3-way in non-RAB

    static constexpr uint32_t RAB_MTE2_V_ID[3] = {EVENT_ID6, EVENT_ID7, EVENT_ID6};
    static constexpr uint32_t SCORE_MTE3_V_ID[3] = {EVENT_ID6, EVENT_ID7, EVENT_ID6};

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});

    CATLASS_DEVICE
    BlockEpilogueScore(uint32_t const (&QK_CROSS_EVENT_ID_)[PROB_STAGES],
                       uint32_t const (&PV_CROSS_EVENT_ID_)[PROB_STAGES], Arch::Resource<ArchTag>& resource,
                       float alpha_ = 1.0f, float scale_ = 1.0f)
    {
        alpha = alpha_;
        scale = scale_;
        ubScoreTensor[0] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::SCORE[0]);
        ubScoreTensor[1] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::SCORE[1]);
        if constexpr (!HAS_RAB) {
            ubScoreTensor[2] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::SCORE[2]);
        }
        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            ubProbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::PROB[i]);
        }

        ubRabTensor[0] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB[0]);
        ubRabTensor[1] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB[1]);
        if constexpr (!HAS_RAB) {
            ubRabTensor[2] = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB[0]);  // dummy, never used
        }

        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            dstTensor[i] = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::DST[i]);
        }

        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            QKcubeReady[i] = Arch::CrossCoreFlag(QK_CROSS_EVENT_ID_[i]);
        }

        for (uint32_t i = 0; i < PROB_STAGES; i++) {
            PVcubeReady[i] = Arch::CrossCoreFlag(PV_CROSS_EVENT_ID_[i]);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCORE_MTE3_V_ID[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(SCORE_MTE3_V_ID[1]);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(RAB_MTE2_V_ID[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(RAB_MTE2_V_ID[1]);
    }

    template <class TensorSrc, class Coord, class Shape>
    CATLASS_DEVICE void operator()(TensorSrc& tensorRab, Coord const& coord, Shape const& shape, Coord const& coordNext,
                                   Shape const& shapeNext, bool hasNext)
    {
        // SPLIT_M: 每 AIC 的 AIV 子核各处理一半行
        auto mReal = CeilDiv<TileBuffer::AIV_PER_AIC>(tla::get<0>(shape));
        auto nReal = tla::get<1>(shape);

        if constexpr (!HAS_RAB) {
            mReal = CeilDiv<TileBuffer::AIV_PER_AIC>(RoundUp<ELEM_PER_BLOCK>(tla::get<0>(shape)));
            nReal = RoundUp<ELEM_PER_BLOCK>(nReal);
        }

        uint32_t slot = UBFlag % PROB_STAGES;
        AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(QKcubeReady[slot].id);

        ComputeBlock(mReal, nReal, 0, tensorRab, coordNext, shapeNext, hasNext);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(PVcubeReady[slot].id);
        UBFlag++;

        rabidx = UBFlag % SCORE_BUF_CNT;
    }

    template <class TensorSrc, class Coord, class Shape>
    CATLASS_DEVICE void ComputeBlock(uint32_t mReal, uint32_t nSize, uint32_t blockIdx, TensorSrc& tensorRab,
                                     Coord const& coordNext, Shape const& shapeNext, bool hasNext)
    {
        auto count = RoundUp<ELEM_PER_BLOCK>(mReal) * RoundUp<ELEM_PER_BLOCK>(nSize);

        if constexpr (!HAS_RAB) {
            count = RoundUp<8>(mReal) * RoundUp<ELEM_PER_BLOCK>(nSize);
        }

        auto repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));

        uint32_t scoreIdx = UBFlag % SCORE_BUF_CNT;
        uint32_t rabIdx = UBFlag % SCORE_BUF_CNT;  // current tile consumes the buffer prefetched for it

        auto scoreEventId = SCORE_MTE3_V_ID[scoreIdx];
        auto rabEventId = RAB_MTE2_V_ID[rabIdx];

        auto ubSPtr = (__ubuf__ Element*)ubScoreTensor[scoreIdx].GetPhyAddr();
        auto ubRabPtr = (__ubuf__ Element*)ubRabTensor[rabIdx].GetPhyAddr();
        auto ubProbPtr = (__ubuf__ ElementAccumulator*)ubProbTensor[UBFlag % PROB_STAGES].GetPhyAddr();

        if constexpr (HAS_RAB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rabEventId);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scoreEventId);

        AscendC::VF_CALL<catlass::Epilogue::RegBase::FastSiluScoreVf<Element, ElementAccumulator, HAS_RAB, HAS_MASK>>(
            ubProbPtr, ubRabPtr, ubRabPtr, ubSPtr, alpha, scale, count, repeatTimes);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(scoreEventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(scoreEventId);

        if constexpr (HAS_RAB) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rabEventId);
        }

        if constexpr (HAS_RAB) {
            if (hasNext) {
                LoadRab(tensorRab, coordNext, shapeNext, (UBFlag + 1) & 1);
            }
        }

        CopyToDst(mReal, nSize, count, UBFlag % PROB_STAGES, scoreIdx);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(scoreEventId);
    }

    CATLASS_DEVICE void CopyToDst(uint32_t mReal, uint32_t nSize, uint32_t count, uint32_t blockIdx, uint32_t scoreIdx)
    {
        if constexpr (HAS_RAB || HAS_MASK) {
            auto coreId = AscendC::GetSubBlockIdx();
            auto totalM = mReal * 2;
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
                AscendC::DataCopy(dstTensor[blockIdx][dstBase], ubScoreTensor[scoreIdx][i * ELEM_PER_BLOCK],
                                  intriParams);
            }

        } else {
            auto coreId = AscendC::GetSubBlockIdx();
            auto mRealAligned = RoundUp<ELEM_PER_BLOCK>(mReal);
            auto numKBlocks = CeilDiv<ELEM_PER_BLOCK>(nSize);
            auto coreOffset = coreId * mReal * ELEM_PER_BLOCK;

            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = numKBlocks;
            intriParams.blockLen = mReal;
            intriParams.srcStride = 0;
            intriParams.dstStride = mReal;

            AscendC::DataCopy(dstTensor[blockIdx][coreOffset], ubScoreTensor[scoreIdx], intriParams);
        }
    }

    template <class TensorSrc, class Coord, class Shape>
    CATLASS_DEVICE void LoadRab(TensorSrc& tensorRab, Coord const& coord, Shape const& shape, int bufIdxOverride = -1)
    {
        if constexpr (!HAS_RAB)
            return;
        uint32_t bufIdx = (bufIdxOverride >= 0) ? static_cast<uint32_t>(bufIdxOverride) : rabidx;
        auto rabEventId = RAB_MTE2_V_ID[bufIdx];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rabEventId);

        auto coreId = AscendC::GetSubBlockIdx();
        auto blockCountTotal = tla::get<0>(shape);

        auto blockLenTotal = tla::get<1>(shape) > L0_TILE_N ? L0_TILE_N : tla::get<1>(shape);
        auto halfBlockCount = RoundUp(blockCountTotal, 2) / 2;

        auto rabCoord = tla::MakeCoord(tla::get<0>(coord), tla::get<1>(coord), tla::get<3>(coord), tla::get<2>(coord));
        auto srcOffset = tensorRab.layout()(rabCoord);

        AscendC::DataCopyParams intriParams;
        AscendC::DataCopyPadParams padParams;

        // coreId == 0: blockCount = halfBlockCount, offset unchanged
        // coreId == 1: blockCount = blockCountTotal - halfBlockCount, offset += halfBlockCount * stride
        intriParams.blockCount = halfBlockCount + coreId * (blockCountTotal - 2 * halfBlockCount);
        srcOffset += coreId * halfBlockCount * tla::get<2>(tensorRab.stride());

        intriParams.blockLen = blockLenTotal * sizeof(Element);
        intriParams.srcStride = (tla::get<2>(tensorRab.stride()) - blockLenTotal) * sizeof(Element);
        intriParams.dstStride = 0;

        if (intriParams.blockCount > 0) {
            padParams.isPad = false;
            AscendC::DataCopyPad(ubRabTensor[bufIdx], tensorRab.data()[srcOffset], intriParams, padParams);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rabEventId);

        if (bufIdxOverride < 0) {
            rabidx = !rabidx;
        }
    }

private:
    Arch::CrossCoreFlag QKcubeReady[PROB_STAGES];
    Arch::CrossCoreFlag PVcubeReady[PROB_STAGES];

    AscendC::LocalTensor<Element> ubScoreTensor[3];
    AscendC::LocalTensor<Element> ubRabTensor[3];
    AscendC::LocalTensor<ElementAccumulator> ubProbTensor[PROB_STAGES];
    AscendC::LocalTensor<Element> dstTensor[PROB_STAGES];

    float alpha{1.0f};
    float scale{1.0f};

    uint32_t UBFlag{0};
    uint32_t rabidx{0};
};

}  // namespace Catlass::Epilogue::Block
