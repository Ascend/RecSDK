/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

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

/**
 • @file block_epilogue_rab_grad.hpp

 • @brief HSTU RAB 梯度 Epilogue Block 实现

 • @description 计算相对位置注意力偏置 (RAB) 的梯度，使用 FastRabGradVf 函数

 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "../../../catlass_hstu/epilogue/regbase/fast_rab_grad.hpp"

namespace Catlass::Epilogue::Block {

/**
 • @brief RAB 梯度 Epilogue Block

 • @tparam EpilogueScoreGrad_ Score 梯度 Epilogue 类型

 • @tparam TileBuffer_ Tile 缓冲区类型

 • @description 执行 RAB 梯度的计算，将 SiLU Score 的梯度传递到 RAB 参数

 */
template <class EpilogueScoreGrad_, class TileBuffer_>
struct BlockEpilogueRabGrad {
public:
    using EpilogueScoreGrad = EpilogueScoreGrad_;
    using ArchTag = typename EpilogueScoreGrad::ArchTag;
    using Element = typename EpilogueScoreGrad::Element;
    using ElementAccumulator = typename EpilogueScoreGrad::ElementAccumulator;
    using L0TileShape = typename EpilogueScoreGrad::L0TileShape;
    using TileBuffer = TileBuffer_;

    static constexpr uint32_t ELEM_PER_BLOCK = Catlass::BYTE_PER_C0 / sizeof(Element);
    static constexpr bool HAS_RAB = EpilogueScoreGrad::HAS_RAB;
    static constexpr bool HAS_MASK = EpilogueScoreGrad::HAS_MASK;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t PING_PONG_ROW = L0_TILE_M / 2;

    CATLASS_DEVICE
    BlockEpilogueRabGrad(uint32_t cubeFlag, uint32_t vecFlag, Arch::Resource<ArchTag> &resource)
    {
        ubGsTensor = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::GS);
        ubGRabPartTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRABPART);
        ubGRabTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRAB);
        dstTensor = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::GRAB_DST[AscendC::GetSubBlockIdx()]);

        cubeReady = Arch::CrossCoreFlag(cubeFlag);
        vecReady = Arch::CrossCoreFlag(vecFlag);
    }

    template <class TensorDst, class Coord, class Shape>
    CATLASS_DEVICE void operator()(TensorDst &tensorGrab, Coord const &coord, Shape const &shape)
    {
        auto mReal = tla::get<0>(shape);
        auto nReal = tla::get<1>(shape);

        auto mLoop = CeilDiv<L0_TILE_M>(mReal);
        auto mTail = mReal - (mLoop - 1) * L0_TILE_M;

        for (auto m = 0; m < mLoop; m++) {
            bool isLast = (m == mLoop - 1);
            auto mSize = isLast ? mTail : L0_TILE_M;

            auto coordNew = tla::Add(coord, tla::MakeCoord(0, 0, m * L0_TILE_M, 0));
            if ((m % 2) == AscendC::GetSubBlockIdx()) {
                auto tileShape = tla::MakeShape(mSize, nReal);
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(cubeReady.id);
                Compute(tileShape);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(vecReady.id);

                if constexpr (HAS_RAB) {
                    CopyOutGrab(tensorGrab, coordNew, tileShape);
                }
            }
        }
    }

    CATLASS_DEVICE void CallVectorFunction(int64_t offset, uint32_t count)
    {
        auto repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));

        auto ubGSPtr = (__ubuf__ ElementAccumulator *)ubGsTensor[offset].GetPhyAddr();
        auto ubGRadPartPtr = (__ubuf__ Element *)ubGRabPartTensor[offset].GetPhyAddr();
        auto ubGRabPtr = (__ubuf__ Element *)ubGRabTensor[offset].GetPhyAddr();

        AscendC::VF_CALL<catlass::Epilogue::RegBase::FastRabGradVf<Element, ElementAccumulator, Element>>(
            ubGSPtr, ubGRadPartPtr, ubGRabPtr, count, repeatTimes);
    }

    template <class Shape>
    CATLASS_DEVICE void Compute(Shape const &shape)
    {
        auto rows = RoundUp<ELEM_PER_BLOCK>(tla::get<0>(shape));
        auto cols = RoundUp<ELEM_PER_BLOCK>(tla::get<1>(shape));

        auto loop = CeilDiv(rows, PING_PONG_ROW);
        for (auto i = 0; i < loop; i++) {
            int64_t offset = i * PING_PONG_ROW * cols;
            uint32_t count = (i == loop - 1) ? (rows * cols - offset) : PING_PONG_ROW * cols;
            uint32_t pingPongFlag = i % 2;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
            CallVectorFunction(offset, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
            CopyToDst(offset, count, rows, cols);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
        }
    }

    CATLASS_DEVICE void CopyToDst(int64_t offset, uint32_t count, uint32_t rows, uint32_t cols)
    {
        if constexpr (HAS_RAB || HAS_MASK) {
            auto rowOffset = offset / cols;
            auto colBlk = cols / ELEM_PER_BLOCK;

            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = count / cols;
            intriParams.blockLen = 1;
            intriParams.srcStride = (colBlk - 1);
            intriParams.dstStride = 0;
            for (auto i = 0; i < colBlk; i++) {
                AscendC::DataCopy(dstTensor[(i * rows + rowOffset) * ELEM_PER_BLOCK],
                    ubGRabTensor[offset + i * ELEM_PER_BLOCK], intriParams);
            }
        } else {
            AscendC::DataCopy(dstTensor[offset], ubGRabTensor[offset], count);
        }
    }

    template <class TensorDst, class Coord, class Shape>
    CATLASS_DEVICE void CopyOutGrab(TensorDst &tensorGrab, Coord const &coord, Shape const &shape)
    {
        auto dstOffset = tensorGrab.layout()(coord);

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = tla::get<0>(shape);
        intriParams.blockLen = tla::get<1>(shape) * sizeof(Element);
        intriParams.srcStride = 0;
        intriParams.dstStride = (tla::get<2>(tensorGrab.stride()) - tla::get<1>(shape)) * sizeof(Element);

        AscendC::DataCopyPad(tensorGrab.data()[dstOffset], ubGRabTensor, intriParams);
    }

private:
    Arch::CrossCoreFlag vecReady;
    Arch::CrossCoreFlag cubeReady;

    AscendC::LocalTensor<ElementAccumulator> ubGsTensor;
    AscendC::LocalTensor<Element> ubGRabTensor;
    AscendC::LocalTensor<Element> ubGRabPartTensor;
    AscendC::LocalTensor<Element> dstTensor;
};

}
