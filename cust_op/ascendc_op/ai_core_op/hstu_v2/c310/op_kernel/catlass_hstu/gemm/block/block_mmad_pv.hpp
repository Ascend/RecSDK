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
 • @file block_mmad_pv.hpp

 • @brief HSTU PV (Probability × Value) 矩阵乘法的 Block MMAD 实现

 • @description 实现 Attention Score 和 Value 的矩阵乘法: P * V

 •              用于计算注意力机制的输出，支持子核绑定和 Swizzle 优化

 */
#pragma once

#include "../../../catlass_hstu/gemm/dispatch_policy.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Gemm::Block {

/**
 • @brief HSTU PV 矩阵乘法的 Block MMAD 特化实现

 • @tparam ArchTag_ 架构标签

 • @tparam PAGED_CACHE_FLAG_ 是否使用分页缓存

 • @tparam ENABLE_UNIT_FLAG_ 是否启用单元标志

 • @tparam BIND_SUB_CORE_ 是否绑定子核

 • @tparam SUB_CORE_ID_ 子核 ID

 • @tparam L1TileShape_ L1 缓存的 Tile 形状

 • @tparam L0TileShape_ L0 缓存的 Tile 形状

 • @tparam ElementA_ 矩阵 A 的数据类型 (Attention Score)

 • @tparam ElementB_ 矩阵 B 的数据类型 (Value)

 • @tparam ElementC_ 矩阵 C 的数据类型

 • @tparam TileBuffer_ Tile 缓冲区类型

 • @tparam TileCopy_ Tile 拷贝类型

 • @tparam TileMmad_ Tile MMAD 类型

 • @description 实现 P * V 的矩阵乘法，用于计算注意力输出

 •              支持子核绑定 (BIND_SUB_CORE) 可将计算绑定到特定子核

 */
template <class ArchTag_, bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, bool BIND_SUB_CORE_, bool SUB_CORE_ID_,
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

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using LayoutTagDST = typename TileCopy::LayoutTagC;

    using L1AAlignHelper = typename TileCopy::L1AAlignHelper;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr bool BIND_SUB_CORE = DispatchPolicy::BIND_SUB_CORE;
    static constexpr uint32_t SUB_CORE_ID = DispatchPolicy::SUB_CORE_ID;

    /**
     ◦ @brief 构造函数

     ◦ @param resource 架构资源对象

     ◦ @param vecFlag Vector 核同步标志

     ◦ @param cubeFlag Cube 核同步标志

     ◦ @param L1B_EVENT_ID_ L1 B 缓冲区事件 ID 数组

     ◦ @description 初始化 PV Block MMAD 核，包括缓冲区分配和事件标志设置

     */
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t vecFlag, uint32_t cubeFlag,
                 uint32_t const (&L1B_EVENT_ID_)[2])
    {
        for (auto i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(TileBuffer::L1A[i]);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(TileBuffer::L1B[i]);

            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(TileBuffer::L0A[i]);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(TileBuffer::L0B[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(TileBuffer::L0C);
        dstTensor = resource.ubBuf.template GetBufferByByte<ElementC>(TileBuffer::DST);

        for (auto i = 0; i < STAGES; i++) {
            vecReady[i] = Arch::CrossCoreFlag(i * AscendC::SYNC_FLAG_ID_MAX + vecFlag);
        }

        cubeReady = Arch::CrossCoreFlag(cubeFlag);

        L1B_EVENT_ID[0] = L1B_EVENT_ID_[0];  // q or grad
        L1B_EVENT_ID[1] = L1B_EVENT_ID_[1];  // q or grad
    }

    /**
     ◦ @brief 触发数据刷新到目标缓冲区

     ◦ @tparam TensorDst 目标张量类型

     ◦ @tparam TensorSrc 源张量类型

     ◦ @tparam TileCopy Tile 拷贝类型

     ◦ @param dst 目标张量

     ◦ @param src 源张量

     ◦ @param isFlush 是否执行刷新

     ◦ @param tileCopy Tile 拷贝对象

     ◦ @description 当计算完成时，将结果从 L0C 刷新到目标缓冲区 (UB)

     ◦              如果启用了子核绑定，还需要设置 Cross Core 同步标志

     */
    template <class TensorDst, class TensorSrc, class TileCopy>
    CATLASS_DEVICE void TriggerFlushToDst(TensorDst &dst, TensorSrc &src, bool isFlush, TileCopy &tileCopy)
    {
        if (isFlush) {
            if constexpr (BIND_SUB_CORE) {
                tileCopy(dst, src, 0, SUB_CORE_ID);
            }
            AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(cubeReady.id);
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla() = default;

    /**
     ◦ @brief 执行 PV 矩阵乘法

     ◦ @param blockShape GEMM 块形状 (M, N, K)

     ◦ @param pingPongFlag 双缓冲标志

     ◦ @param l0bFlag L0 B 缓冲区标志

     ◦ @param isInit 是否首次执行

     ◦ @param isFlush 是否刷新结果到目标

     ◦ @description 执行 P * V 矩阵乘法，包含以下步骤:

     ◦              1. 等待 Vector 核数据就绪

     ◦              2. 拷贝 L1 B -> L0 B

     ◦              3. 循环处理 M 维度:

     ◦                 - 等待 A 数据就绪

     ◦                 - 拷贝 L1 A -> L0 A

     ◦                 - 执行 Cube 矩阵乘法

     ◦              4. 如果是最后一块，触发数据刷新到目标缓冲区

     */
    CATLASS_DEVICE
    void operator()(GemmCoord &blockShape, uint32_t &pingPongFlag, uint32_t &l0bFlag, bool isInit = false,
                    bool isFlush = false)
    {
        uint32_t mReal = blockShape.m();
        uint32_t nReal = blockShape.n();
        uint32_t kReal = blockShape.k();

        auto dstLayout = tla::MakeLayout<ElementC, LayoutTagDST>(nReal, kReal);
        auto tensorC = tla::MakeTensor(dstTensor, dstLayout, Arch::PositionUB{});

        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<decltype(tensorC)>;
        CopyL0CToDst copyL0CToDst;

        auto coord = tla::MakeCoord(0, 0);

        auto mLoop = CeilDiv<L0_TILE_M>(mReal);
        auto mTail = mReal - (mLoop - 1) * L0_TILE_M;

        for (auto m = 0; m < mLoop; ++m) {
            bool isLast = (m == mLoop - 1);
            auto mSize = isLast ? mTail : L0_TILE_M;

            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(vecReady[m % STAGES].id);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0bFlag + 2);  // 2 means pingPong
            auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(mSize, kReal);
            auto tensorL1b = tla::MakeTensor(l1BTensor[m % STAGES], l1bLayout, coord, Arch::PositionL1{});
            auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(mSize, kReal);
            auto tensorL0b = tla::MakeTensor(l0BTensor[l0bFlag], l0bLayout, coord, Arch::PositionL0B{});
            copyL1ToL0B(tensorL0b, tensorL1b);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B_EVENT_ID[m % STAGES]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingPongFlag);
            auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(nReal, mSize);
            auto tensorL1a = tla::MakeTensor(l1ATensor[m % STAGES], l1aLayout, coord, Arch::PositionL1{});
            auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(nReal, mSize);
            auto tensorL0a = tla::MakeTensor(l0ATensor[pingPongFlag], l0aLayout, coord, Arch::PositionL0A{});
            copyL1ToL0A(tensorL0a, tensorL1a);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingPongFlag);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(pingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingPongFlag);
            auto l0cLayout = tla::MakeLayoutL0C(nReal, kReal);
            auto tensorL0c = tla::MakeTensor(l0CTensor, l0cLayout, coord, Arch::PositionL0C{});
            tileMmad(tensorL0c, tensorL0a, tensorL0b, nReal, kReal, mSize, isInit, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(pingPongFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingPongFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0bFlag + 2); // 2 means pingPong

            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(pingPongFlag);
            if (isLast) {
                TriggerFlushToDst(tensorC, tensorL0c, isFlush, copyL0CToDst);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pingPongFlag);

            pingPongFlag = (pingPongFlag + 1) % 2; // 2 means pingPong
            l0bFlag = (l0bFlag + 1) % 2;  // 2 means pingPong
            isInit = false;
        }
    }

protected:
    Arch::CrossCoreFlag vecReady[STAGES];
    Arch::CrossCoreFlag cubeReady;

    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    AscendC::LocalTensor<ElementC> dstTensor;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t L1B_EVENT_ID[2]{0};
};

}
