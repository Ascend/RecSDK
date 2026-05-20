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

/**
 • @file block_mmad_qk.hpp

 • @brief HSTU QK^T 矩阵乘法的 Block MMAD 实现

 • @description 实现 Query 和 Key 的注意力分数计算: Q * K^T

 •              使用 CATLASS 框架的 BlockMmadTla 模板，支持双缓冲和流水线调度

 */

#pragma once

#include "../../../catlass_hstu/gemm/dispatch_policy.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

namespace Catlass::Gemm::Block {

/**
 • @brief HSTU QK^T 矩阵乘法的 Block MMAD 特化实现

 • @tparam ArchTag_ 架构标签 (如 Ascend950)

 • @tparam PAGED_CACHE_FLAG_ 是否使用分页缓存

 • @tparam ENABLE_UNIT_FLAG_ 是否启用单元标志

 • @tparam L1TileShape_ L1 缓存的 Tile 形状

 • @tparam L0TileShape_ L0 缓存的 Tile 形状

 • @tparam ElementA_ 矩阵 A 的数据类型

 • @tparam ElementB_ 矩阵 B 的数据类型

 • @tparam ElementC_ 矩阵 C (输出) 的数据类型

 • @tparam TileBuffer_ Tile 缓冲区类型

 • @tparam TileCopy_ Tile 拷贝类型

 • @tparam TileMmad_ Tile MMAD 类型

 • @description 实现 Q * K^T 的矩阵乘法，用于计算注意力分数

 •              支持多阶段流水线，包含 L1/L0 拷贝和 Cube/Vector 混合计算

 */
template <class ArchTag_, bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SCALAR_QUANT_,
          class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
          class TileBuffer_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadHSTUQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, ENABLE_SCALAR_QUANT_>,
                    L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_,
                    TileBuffer_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadHSTUQK<ArchTag_, PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, ENABLE_SCALAR_QUANT_>;
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
    static constexpr bool ENABLE_SCALAR_QUANT = DispatchPolicy::ENABLE_SCALAR_QUANT;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // 提取 CopyGmToL1A 和 CopyL0CToDst 的通用模板
    template <class TensorSrc>
    using CopyGmToL1A_T = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorSrc, typename TileCopy_::TensorL1A>;
    
    template <class TensorDst>
    using CopyL0CToDst_T = typename TileCopy_::template CopyL0CToDst<TensorDst>;

    /**
     ◦ @brief 获取 CopyL0CToDst 的具体实例

     ◦ @tparam M 实际的 M 维度大小

     ◦ @tparam N 实际的 N 维度大小

     ◦ @description 根据给定的形状返回 CopyL0CToDst 实例

     */
    CATLASS_DEVICE auto GetCopyL0CToDst() {
        auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(0, 0);
        auto tensorC = tla::MakeTensor(dstTensor, dstLayout, Arch::PositionUB{});
        return CopyL0CToDst_T<decltype(tensorC)>{};
    }

    /**
     ◦ @brief 获取 CopyGmToL1A 的具体实例

     ◦ @tparam TensorA 输入张量类型

     ◦ @description 根据输入张量类型返回 CopyGmToL1A 实例

     */
    template <class TensorA>
    CATLASS_DEVICE auto GetCopyGmToL1A() {
        return CopyGmToL1A_T<TensorA>{};
    }

    /**
     ◦ @brief 初始化缓冲区

     ◦ @param resource 架构资源对象，包含 L1/L0/UB 缓冲区

     ◦ @description 从资源中分配和初始化 L1、L0、UB 缓冲区，用于存储矩阵数据

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

     ◦ @param headNum 注意力头数

     ◦ @param headDim 头维度

     ◦ @param cubeFlag Cube 核同步标志

     ◦ @param L1B_EVENT_ID_ K/V 数据的 L1 事件 ID

     ◦ @param L1A_EVENT_ID_ Q/Grad 数据的 L1 事件 ID 数组

     ◦ @description 初始化 Block MMAD 核，包括缓冲区初始化、事件标志设置、头数和步长设置

     */
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t headNum, uint32_t headDim, uint32_t cubeFlag,
                 uint32_t L1B_EVENT_ID_, uint32_t const (&L1A_EVENT_ID_)[2])
    {
        InitBuffer(resource);
        L1B_EVENT_ID = L1B_EVENT_ID_;        // k or v
        L1A_EVENT_ID[0] = L1A_EVENT_ID_[0];  // q or grad
        L1A_EVENT_ID[1] = L1A_EVENT_ID_[1];  // q or grad

        for (auto i = 0; i < STAGES; i++) {
            cubeReady[i] = Arch::CrossCoreFlag(i * AscendC::SYNC_FLAG_ID_MAX + cubeFlag);
        }

        this->headNum = headNum;
        this->stride = headNum * headDim;
    }

    CATLASS_DEVICE
    void SetDeqScalar(ElementAccumulator deqScalar)
    {
        this->deqScalar = deqScalar;
    }

    CATLASS_DEVICE
    ~BlockMmadTla() = default;

    /**
     ◦ @brief 获取源张量数据

     ◦ @param src 源张量引用

     ◦ @description 从 Global Memory 拷贝数据到 L1 缓冲区，等待 K/V 数据就绪后执行

     */
    template <class TensorSrc>
    CATLASS_DEVICE void AcquireTensor(TensorSrc &src)
    {
        using TileCopy = Gemm::Tile::TileCopyTNDTla<ArchTag, TensorSrc, typename TileCopy_::TensorL1B>;
        TileCopy tileCopy;

        uint32_t nReal = tla::get<0>(src.shape());
        uint32_t cols = tla::get<1>(src.shape());
        uint32_t rows = RoundUp<L1AAlignHelper::M_ALIGNED>(nReal);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1B_EVENT_ID);
        auto l1Layout = tla::MakeLayout<ElementB, LayoutTagL1B>(rows, cols);
        auto tensorL1 = tla::MakeTensor(l1BTensor, l1Layout, Arch::PositionL1{});
        tileCopy(tensorL1, src, nReal, cols, stride);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    }

    /**
     ◦ @brief 释放张量

     ◦ @description 标记 L1 B 数据已使用完成，触发后续数据拷贝

     */
    CATLASS_DEVICE
    void ReleaseTensor()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1B_EVENT_ID);
    }

    /**
     ◦ @brief 执行 QK^T 矩阵乘法

     ◦ @tparam TensorA 矩阵 A 的张量类型 (Query)

     ◦ @tparam TensorB 矩阵 B 的张量类型 (Key)

     ◦ @param tensorA 输入张量 A

     ◦ @param tensorB 输入张量 B

     ◦ @param pingPongFlag 双缓冲标志，切换 A 数据的 ping-pong 缓冲区

     ◦ @param l0bFlag L0 B 缓冲区标志

     ◦ @param triggerSwizzle 是否触发 Swizzle 优化

     ◦ @description 执行完整的 Q * K^T 矩阵乘法，包含以下步骤:

     ◦              1. 等待 L1 B 数据就绪

     ◦              2. 拷贝 L1 B -> L0 B

     ◦              3. 循环处理 M 维度:

     ◦                 - 拷贝 GM -> L1 A

     ◦                 - 拷贝 L1 A -> L0 A

     ◦                 - 执行 Cube 矩阵乘法 L0A * L0B -> L0C

     ◦                 - 拷贝 L0C -> UB

     ◦              4. 触发 Cross Core 同步

     */
    template <class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, uint32_t &pingPongFlag, uint32_t &l0bFlag,
                                   bool triggerSwizzle)
    {
        uint32_t mReal = tla::get<0>(tensorA.shape());
        uint32_t nReal = tla::get<0>(tensorB.shape());
        uint32_t kReal = tla::get<1>(tensorA.shape());
        uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nReal);

        auto copyGmToL1A = GetCopyGmToL1A<TensorA>();
        auto copyL0CToDst = GetCopyL0CToDst();

        auto coord = tla::MakeCoord(0, 0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0bFlag + 2);
        auto l1bLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(nRound, kReal);
        auto tensorL1b = tla::MakeTensor(l1BTensor, l1bLayout, coord, Arch::PositionL1{});
        auto l0bLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(nRound, kReal);
        auto tensorL0b = tla::MakeTensor(l0BTensor[l0bFlag], l0bLayout, coord, Arch::PositionL0B{});
        copyL1ToL0B(tensorL0b, tensorL1b);

        auto mLoop = CeilDiv<L0_TILE_M>(mReal);
        auto mTail = mReal - (mLoop - 1) * L0_TILE_M;
        for (auto m = 0; m < mLoop; m++) {
            auto mSize = (m == mLoop - 1) ? mTail : L0_TILE_M;

            auto coordSrc = tla::MakeCoord(m * L0_TILE_M * this->headNum, 0);
            auto tensorATile = tla::GetTile(tensorA, coordSrc, tla::MakeShape(mSize, kReal));

            // copy ga -> l1a
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_EVENT_ID[m % STAGES]);
            auto l1aLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(mSize, kReal);
            auto tensorL1a = tla::MakeTensor(l1ATensor[m % STAGES], l1aLayout, coord, Arch::PositionL1{});
            if (!triggerSwizzle) {copyGmToL1A(tensorL1a, tensorATile, mSize, kReal, stride);}
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingPongFlag);

            // copy l1a -> l0a
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingPongFlag);
            auto l0aLayout = tla::MakeLayout<ElementA, LayoutTagL0A>(mSize, kReal);
            auto tensorL0a = tla::MakeTensor(l0ATensor[pingPongFlag], l0aLayout, coord, Arch::PositionL0A{});
            copyL1ToL0A(tensorL0a, tensorL1a);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingPongFlag);

            // matmul
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(pingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingPongFlag);
            auto l0cLayout = tla::MakeLayoutL0C(mSize, nRound);
            auto tensorL0c = tla::MakeTensor(l0CTensor[pingPongFlag], l0cLayout, coord, Arch::PositionL0C{});
            tileMmad(tensorL0c, tensorL0a, tensorL0b, mSize, nReal, kReal, true, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(pingPongFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingPongFlag);

            // copy l0c -> ub
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(pingPongFlag);
            auto dstLayout = tla::MakeLayout<ElementA, LayoutTagDST>(mSize, nRound);
            auto tensorC = tla::MakeTensor(dstTensor, dstLayout, Arch::PositionUB{});
            if constexpr (ENABLE_SCALAR_QUANT) {
                copyL0CToDst(tensorC, tensorL0c, 0, deqScalar, m % STAGES);
            } else {
                copyL0CToDst(tensorC, tensorL0c, 0, m % STAGES);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pingPongFlag);

            AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(cubeReady[m % STAGES].id);

            pingPongFlag = (pingPongFlag + 1) % STAGES;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0bFlag + 2);
        l0bFlag = (l0bFlag + 1) % STAGES;
    }

protected:
    Arch::CrossCoreFlag cubeReady[STAGES];

    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> dstTensor;

    int64_t stride{0};
    uint32_t headNum{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t L1B_EVENT_ID{0};
    uint32_t L1A_EVENT_ID[2]{0};

    ElementAccumulator deqScalar{0.0f};
};
}
