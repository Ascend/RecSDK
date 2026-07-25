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
 • @file block_epilogue_score_grad.hpp

 • @brief HSTU Score 梯度 Epilogue Block 实现

 • @description 计算 SiLU 激活函数的反向传播梯度，使用 FastSiluGradVf 函数

 */
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "../../../catlass_hstu/epilogue/regbase/fast_silu_grad.hpp"

namespace Catlass::Epilogue::Block {

/**
 • @brief Score 梯度 Epilogue Block

 • @tparam ArchTag_ 架构标签

 • @tparam Element_ 数据类型

 • @tparam ElementAccumulator_ 累加器类型

 • @tparam TileBuffer_ Tile 缓冲区类型

 • @tparam L0TileShape_ L0 Tile 形状

 • @tparam HAS_RAB_ 是否有相对位置偏置

 • @tparam HAS_MASK_ 是否有掩码

 • @description 执行 SiLU 激活函数的梯度计算: d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

 */
template <class ArchTag_, class Element_, class ElementAccumulator_, class TileBuffer_, class L0TileShape_,
          class L1TileShape_, bool HAS_RAB_ = false, bool HAS_MASK_ = false>
struct BlockEpilogueScoreGrad {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
    using TileBuffer = TileBuffer_;
    using L0TileShape = L0TileShape_;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t ELEM_PER_BLOCK = Catlass::BYTE_PER_C0 / sizeof(Element);

    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t PING_PONG_ROW = L0_TILE_M / 2;

    CATLASS_DEVICE
    BlockEpilogueScoreGrad(ElementAccumulator alpha_, ElementAccumulator scale_, uint32_t cubeFlag, uint32_t vecFlag,
                           Arch::Resource<ArchTag>& resource)
        : alpha(alpha_),
          scale(scale_)
    {
        if constexpr (HAS_RAB) {
            ubRabTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::RAB);
        }

        if constexpr (HAS_MASK) {
            ubMaskTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRABPART);
        }

        ubScoreTensor = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::SCORE);
        ubProbTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::PROB);
        ubGRabPartTensor = resource.ubBuf.template GetBufferByByte<Element>(TileBuffer::GRABPART);
        dstTensor = resource.l1Buf.template GetBufferByByte<Element>(TileBuffer::PROB_DST[AscendC::GetSubBlockIdx()]);

        cubeReady = Arch::CrossCoreFlag(cubeFlag);
        vecReady = Arch::CrossCoreFlag(vecFlag);
    }

    template <class TensorSrc, class Coord, class Shape, class Predictor>
    CATLASS_DEVICE void operator()(TensorSrc& tensorRab, Coord const& coord, Shape const& shape, Predictor& predictor)
    {
        bool needMask = predictor.needMask;
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
                auto rows = RoundUp<ELEM_PER_BLOCK>(mSize);
                auto cols = RoundUp<ELEM_PER_BLOCK>(nReal);

                if constexpr (HAS_RAB) {
                    CopyRab(tensorRab, coordNew, tileShape);
                }

                if constexpr (HAS_MASK) {
                    predictor.ApplyMask(ubMaskTensor, coordNew, tileShape, rows, cols);
                }

                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(cubeReady.id);
                Compute(tileShape, needMask);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(vecReady.id);
            }
        }
    }

    CATLASS_DEVICE void CallVectorFunction(int64_t offset, uint32_t count, bool needMask)
    {
        auto repeatTimes = CeilDiv(count, AscendC::GetVecLen() / sizeof(ElementAccumulator));

        auto ubSPtr = (__ubuf__ ElementAccumulator*)ubScoreTensor[offset].GetPhyAddr();
        auto ubRabPtr = (__ubuf__ Element*)ubRabTensor[offset].GetPhyAddr();
        auto ubMaskPtr = (__ubuf__ Element*)ubMaskTensor[offset].GetPhyAddr();
        auto ubProbPtr = (__ubuf__ Element*)ubProbTensor[offset].GetPhyAddr();
        auto ubGradRabPartPtr = (__ubuf__ Element*)ubGRabPartTensor[offset].GetPhyAddr();

        AscendC::VF_CALL<catlass::Epilogue::RegBase::FastSiluGradVf<Element, ElementAccumulator, Element, HAS_RAB>>(
            ubSPtr, ubRabPtr, ubMaskPtr, ubProbPtr, ubGradRabPartPtr, alpha, count, repeatTimes, needMask);
    }

    template <class Shape>
    CATLASS_DEVICE void Compute(Shape& shape, bool needMask)
    {
        auto rows = RoundUp<ELEM_PER_BLOCK>(tla::get<0>(shape));
        auto cols = RoundUp<ELEM_PER_BLOCK>(tla::get<1>(shape));

        if constexpr (HAS_RAB) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        auto loop = CeilDiv(rows, PING_PONG_ROW);
        for (auto i = 0; i < loop; i++) {
            int64_t offset = i * PING_PONG_ROW * cols;
            uint32_t count = (i == loop - 1) ? (rows * cols - offset) : PING_PONG_ROW * cols;
            uint32_t pingPongFlag = i % 2;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
            CallVectorFunction(offset, count, needMask);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
            CopyToDst(offset, count, rows, cols);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
        }

        if constexpr (HAS_RAB) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
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
                                  ubProbTensor[offset + i * ELEM_PER_BLOCK], intriParams);
            }
        } else {
            AscendC::DataCopy(dstTensor[offset], ubProbTensor[offset], count);
        }
    }

    template <class TensorSrc, class Coord, class Shape>
    CATLASS_DEVICE void CopyRab(TensorSrc& tensorRab, Coord const& coord, Shape const& shape)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        auto srcOffset = tensorRab.layout()(coord);

        AscendC::DataCopyParams intriParams;
        AscendC::DataCopyPadParams padParams;
        intriParams.blockCount = tla::get<0>(shape);
        intriParams.blockLen = tla::get<1>(shape) * sizeof(Element);
        intriParams.srcStride = (tla::get<2>(tensorRab.stride()) - tla::get<1>(shape)) * sizeof(Element);
        intriParams.dstStride = 0;

        padParams.isPad = false;
        AscendC::DataCopyPad(ubRabTensor, tensorRab.data()[srcOffset], intriParams, padParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

private:
    Arch::CrossCoreFlag vecReady;
    Arch::CrossCoreFlag cubeReady;

    AscendC::LocalTensor<ElementAccumulator> ubScoreTensor;
    AscendC::LocalTensor<Element> ubRabTensor;
    AscendC::LocalTensor<Element> ubMaskTensor;
    AscendC::LocalTensor<Element> ubProbTensor;
    AscendC::LocalTensor<Element> ubGRabPartTensor;
    AscendC::LocalTensor<Element> dstTensor;

    ElementAccumulator alpha{0.0f};
    ElementAccumulator scale{0.0f};
};

}  // namespace Catlass::Epilogue::Block
