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
 • @file block_mmad_dq.hpp

 • @brief HSTU dQ (Query 梯度) 矩阵乘法的 Block MMAD 实现

 • @description 实现 dOutput * K 的矩阵乘法来计算 Query 的梯度: dQ = dO * K^T

 •              用于反向传播中计算 Query 的梯度

 */

#pragma once

#include "../../../catlass_hstu/gemm/dispatch_policy.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Gemm::Block {

/**
 • @brief HSTU dQ 矩阵乘法的 Block MMAD 特化实现

 • @tparam ArchTag_ 架构标签

 • @tparam PAGED_CACHE_FLAG_ 是否使用分页缓存

 • @tparam ENABLE_UNIT_FLAG_ 是否启用单元标志

 • @tparam L1TileShape_ L1 缓存的 Tile 形状

 • @tparam L0TileShape_ L0 缓存的 Tile 形状

 • @tparam ElementA_ 矩阵 A 的数据类型 (dOutput)

 • @tparam ElementB_ 矩阵 B 的数据类型 (Key)

 • @tparam ElementC_ 矩阵 C 的数据类型 (dQ)

 • @tparam TileBuffer_ Tile 缓冲区类型

 • @tparam TileCopy_ Tile 拷贝类型

 • @tparam TileMmad_ Tile MMAD 类型

 • @description 实现 dO * K^T 的矩阵乘法，用于计算 Query 的梯度

 •              使用原子加法操作支持多核并行计算

 */
template <class ArchTag_, bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_, class ElementC_, class TileBuffer_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadHSTUDQ<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>, L1TileShape_, L0TileShape_, ElementA_,
                    ElementB_, ElementC_, TileBuffer_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadHSTUDQ<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
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

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    /**
     ◦ @brief 初始化缓冲区

     ◦ @param resource 架构资源对象

     ◦ @description 从资源中分配和初始化 L1、L0、UB 缓冲区

     */
    CATLASS_DEVICE
    void InitBuffer(Arch::Resource<ArchTag> &resource)
    {
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1B);
        for (auto i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1A[i]);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[i]);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[i]);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C[i]);
        }
        dstTensor = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::DST);
    }

    /**
     ◦ @brief 构造函数

     ◦ @param resource 架构资源对象

     ◦ @description 调用 InitBuffer 初始化所有缓冲区

     */
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource)
    {
        InitBuffer(resource);
    }

    CATLASS_DEVICE
    ~BlockMmadTla() = default;

    /**
     ◦ @brief 执行 dQ 矩阵乘法

     ◦ @tparam TensorC 输出张量类型

     ◦ @param tensorC 输出张量 (dQ)

     ◦ @param blockShape GEMM 块形状 (M, N, K)

     ◦ @param pingPongFlag 双缓冲标志

     ◦ @param l0bFlag L0 B 缓冲区标志

     ◦ @description 执行 dO * K^T 矩阵乘法来计算 dQ，包含以下步骤:

     ◦              1. 等待 L1 B 数据就绪，拷贝 L1 B -> L0 B

     ◦              2. 循环处理 M 维度:

     ◦                 - 等待 A 数据就绪

     ◦                 - 拷贝 L1 A -> L0 A

     ◦                 - 执行 Cube 矩阵乘法

     ◦              3. 使用原子加法将结果累加到输出缓冲区 (支持多核并行)

     */
    template <class TensorC>
    CATLASS_DEVICE void operator()(TensorC &tensorC, GemmCoord &blockShape, uint32_t &pingPongFlag, uint32_t &l0bFlag)
    {
        uint32_t mReal = blockShape.m();
        uint32_t nReal = blockShape.n();
        uint32_t kReal = blockShape.k();

        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;

        auto coord = tla::MakeCoord(0, 0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0bFlag + 2);
        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(nReal, kReal);
        auto tensorL1b = tla::MakeTensor(l1BTensor, l1bLayout, coord, Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(nReal, kReal);
        auto tensorL0b = tla::MakeTensor(l0BTensor[l0bFlag], l0bLayout, coord, Arch::PositionL0B{});
        copyL1ToL0B(tensorL0b, tensorL1b);

        auto mLoop = CeilDiv<L0_TILE_M>(mReal);
        auto mTail = mReal - (mLoop - 1) * L0_TILE_M;

        for (auto m = 0; m < mLoop; ++m) {
            bool isLast = (m == mLoop - 1);
            auto mSize = isLast ? mTail : L0_TILE_M;

            auto coordDst = tla::MakeCoord(m * L0_TILE_M, 0);
            auto tensorCTile = tla::GetTile(tensorC, coordDst, tla::MakeShape(mSize, kReal));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingPongFlag);
            auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(mSize, nReal);
            auto tensorL1a = tla::MakeTensor(l1ATensor[m % STAGES], l1aLayout, coord, Arch::PositionL1{});
            auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mSize, nReal);
            auto tensorL0a = tla::MakeTensor(l0ATensor[pingPongFlag], l0aLayout, coord, Arch::PositionL0A{});
            copyL1ToL0A(tensorL0a, tensorL1a);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingPongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(pingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingPongFlag);
            auto l0cLayout = tla::MakeLayoutL0C(mSize, kReal);
            auto tensorL0c = tla::MakeTensor(l0CTensor[pingPongFlag], l0cLayout, coord, Arch::PositionL0C{});
            tileMmad(tensorL0c, tensorL0a, tensorL0b, mSize, kReal, nReal, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(pingPongFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingPongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(pingPongFlag);

            AscendC::SetAtomicAdd<ElementAccumulator>();
            copyL0CToDst(tensorCTile, tensorL0c, 0);
            AscendC::SetAtomicNone();

            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pingPongFlag);

            pingPongFlag = (pingPongFlag + 1) % 2;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0bFlag + 2);
        l0bFlag = (l0bFlag + 1) % 2;
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> dstTensor;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

}
