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
 • @file backward_kernel_mmad_mainloop.hpp

 • @brief HSTU Backward 算子 MMAD 主循环实现

 • @description 实现 Cube 核上的矩阵乘法主循环，包含 QK、GV、各梯度计算的调度和执行

 */

#pragma once

#include "catlass/detail/macros.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Kernel {

/**
 • @brief Backward MMAD 主循环结构体

 • @tparam BlockMmadQK_ QK 矩阵乘法 Block

 • @tparam BlockMmadGV_ GV 矩阵乘法 Block

 • @tparam BlockMmadVGrad_ dV 梯度 Block

 • @tparam BlockMmadKGrad_ dK 梯度 Block

 • @tparam BlockMmadQGrad_ dQ 梯度 Block

 • @tparam QBlockScheduler_ Q 方向块调度器

 • @tparam KBlockScheduler_ K 方向块调度器

 • @description 负责在 AI Core 的 Cube 核上执行所有矩阵乘法操作，

 •              包括前向的 Q*K^T、Score*V 以及反向的 dQ、dK、dV 计算

 */
template <class BlockMmadQK_, class BlockMmadGV_, class BlockMmadVGrad_, class BlockMmadKGrad_, class BlockMmadQGrad_,
          class QBlockScheduler_, class KBlockScheduler_>
struct BackwardMmadMainloop {
    using BlockMmadQK = BlockMmadQK_;
    using BlockMmadGV = BlockMmadGV_;
    using BlockMmadVGrad = BlockMmadVGrad_;
    using BlockMmadKGrad = BlockMmadKGrad_;
    using BlockMmadQGrad = BlockMmadQGrad_;
    using QBlockScheduler = QBlockScheduler_;
    using KBlockScheduler = KBlockScheduler_;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ArchTag = typename BlockMmadQK::ArchTag;
    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementG = typename BlockMmadGV::ElementA;
    using ElementV = typename BlockMmadGV::ElementB;
    using ElementACC = typename BlockMmadGV::ElementAccumulator;

    static constexpr uint32_t EVENT_V_ID = EVENT_ID0;
    static constexpr uint32_t EVENT_GRAD0_ID = EVENT_ID1;
    static constexpr uint32_t EVENT_GRAD1_ID = EVENT_ID2;
    static constexpr uint32_t EVENT_Q0_ID = EVENT_ID3;
    static constexpr uint32_t EVENT_Q1_ID = EVENT_ID4;
    static constexpr uint32_t EVENT_K_ID = EVENT_ID5;

    // Get BLOCK_M by indices 0
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    // Get BLOCK_N by indices 1
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    // Get BLOCK_K by indices 2
    static constexpr uint32_t BLOCK_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t QK_READY_ID = 0;
    static constexpr uint32_t GV_READY_ID = 1;
    static constexpr uint32_t PROB_READY_ID = 2;
    static constexpr uint32_t GRAB_READY_ID = 3;
    static constexpr uint32_t TRANS_READY_ID = 4;
    static constexpr uint32_t V_TRANS_READY_ID = TRANS_READY_ID;
    static constexpr uint32_t K_TRANS_READY_ID = AscendC::SYNC_FLAG_ID_MAX + V_TRANS_READY_ID;
    static constexpr uint32_t Q_TRANS_READY_ID = 5;

    struct Params {
        GM_ADDR ptrGrad;
        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrSeqOffsetQ;
        GM_ADDR ptrSeqOffsetK;
        GM_ADDR ptrQShare;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrGrad_, GM_ADDR ptrQ_, GM_ADDR ptrK_, GM_ADDR ptrV_, GM_ADDR ptrSeqOffsetQ_,
               GM_ADDR ptrSeqOffsetK_, GM_ADDR ptrQShare_)
            : ptrGrad(ptrGrad_),
              ptrQ(ptrQ_),
              ptrK(ptrK_),
              ptrV(ptrV_),
              ptrSeqOffsetQ(ptrSeqOffsetQ_),
              ptrSeqOffsetK(ptrSeqOffsetK_),
              ptrQShare(ptrQShare_)
        {
        }
    };

    CATLASS_DEVICE
    BackwardMmadMainloop(GM_ADDR ptrTiling_)
    {
        GET_TILING_DATA(tilingData, ptrTiling_);
        batch = tilingData.batch;
        heads = tilingData.heads;
        dimQK = tilingData.dimQK;
        dimGV = tilingData.dimGV;
        totalSeqLenQ = tilingData.totalSeqLenQ;
        totalSeqLenK = tilingData.totalSeqLenK;
        targetGroupSize = tilingData.targetGroupSize;
        alpha = tilingData.alpha;
        scale = tilingData.scale;
    }

    struct PipeEventGuard {
        CATLASS_DEVICE
        PipeEventGuard()
        {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_ID);      // gK
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q0_ID);     // gQ0
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q1_ID);     // gQ1
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_V_ID);      // gV
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD0_ID);  // gR0
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD1_ID);  // gR1
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        }

        CATLASS_DEVICE
        ~PipeEventGuard()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q0_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q1_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_V_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD0_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD1_ID);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        }
    };

    CATLASS_DEVICE
    void InitGlobalTensors(const Params &params,
                           AscendC::GlobalTensor<ElementQ> &gQ,
                           AscendC::GlobalTensor<ElementK> &gK,
                           AscendC::GlobalTensor<ElementG> &gGrad,
                           AscendC::GlobalTensor<ElementV> &gV,
                           AscendC::GlobalTensor<ElementACC> &gQShare)
    {
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.ptrQ);
        gK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK);
        gGrad.SetGlobalBuffer((__gm__ ElementG *)params.ptrGrad);
        gV.SetGlobalBuffer((__gm__ ElementV *)params.ptrV);
        gQShare.SetGlobalBuffer((__gm__ ElementACC *)params.ptrQShare);
        gK.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gV.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    template <typename ElementT>
    CATLASS_DEVICE
    auto MakeTNDTensor(AscendC::GlobalTensor<ElementT> &gt, uint32_t seqLen, uint32_t dim)
    {
        auto layout = tla::MakeLayout(tla::MakeShape((int64_t)seqLen * heads, dim),
                                      tla::MakeStride(dim, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    CATLASS_DEVICE
    void operator()(const Params &params)
    {
        PipeEventGuard pipeEventGuard;

        AscendC::GlobalTensor<ElementQ> gQ;
        AscendC::GlobalTensor<ElementK> gK;
        AscendC::GlobalTensor<ElementG> gGrad;
        AscendC::GlobalTensor<ElementV> gV;
        AscendC::GlobalTensor<ElementACC> gQShare;
        InitGlobalTensors(params, gQ, gK, gGrad, gV, gQShare);

        auto tensorQ = MakeTNDTensor(gQ, totalSeqLenQ, dimQK);
        auto tensorK = MakeTNDTensor(gK, totalSeqLenK, dimQK);
        auto tensorGrad = MakeTNDTensor(gGrad, totalSeqLenQ, dimGV);
        auto tensorV = MakeTNDTensor(gV, totalSeqLenK, dimGV);
        auto tensorQShare = MakeTNDTensor(gQShare, totalSeqLenQ, dimQK);

        BlockMmadQK blockMmadQK(resource, heads, dimQK, QK_READY_ID, EVENT_K_ID, {EVENT_Q0_ID, EVENT_Q1_ID});
        BlockMmadGV blockMmadGV(resource, heads, dimGV, GV_READY_ID, EVENT_V_ID, {EVENT_GRAD0_ID, EVENT_GRAD1_ID});
        BlockMmadVGrad blockMmadVGrad(resource, PROB_READY_ID, V_TRANS_READY_ID, {EVENT_GRAD0_ID, EVENT_GRAD1_ID});
        BlockMmadKGrad blockMmadKGrad(resource, GRAB_READY_ID, K_TRANS_READY_ID, {EVENT_Q0_ID, EVENT_Q1_ID});
        BlockMmadQGrad blockMmadQGrad(resource);

        blockMmadQK.SetDeqScalar(alpha);
        blockMmadGV.SetDeqScalar(scale * alpha);
        blockMmadVGrad.SetDeqScalar(scale);

        QBlockScheduler qBlockScheduler(batch, heads, params.ptrSeqOffsetQ);
        KBlockScheduler kBlockScheduler(batch, heads, params.ptrSeqOffsetK, params.ptrSeqOffsetQ);
        uint32_t pingPongFlag = 0;
        uint32_t l0bFlag = 0;

        kBlockScheduler.Init();
        for (; kBlockScheduler.IsValid(); ++kBlockScheduler) {
            auto tK = kBlockScheduler.GetTile(tensorK);
            auto tV = kBlockScheduler.GetTile(tensorV);
            blockMmadQK.AcquireTensor(tK);
            blockMmadGV.AcquireTensor(tV);

            qBlockScheduler.Init(kBlockScheduler);
            for (; qBlockScheduler.IsValid(); ++qBlockScheduler) {
                auto triggerSwizzle = qBlockScheduler.GetTriggerSwizzle();
                auto tQ = qBlockScheduler.GetTile(tensorQ);
                blockMmadQK(tQ, tK, pingPongFlag, l0bFlag, triggerSwizzle);

                auto tG = qBlockScheduler.GetTile(tensorGrad);
                blockMmadGV(tG, tV, pingPongFlag, l0bFlag, triggerSwizzle);

                GemmCoord blockQActualShape(tla::get<0>(tQ.shape()), tla::get<0>(tK.shape()), dimQK);
                GemmCoord blockGActualShape(tla::get<0>(tQ.shape()), tla::get<0>(tK.shape()), dimGV);

                auto isFirstQBlock = qBlockScheduler.IsFirst();
                auto isLastQBlock = qBlockScheduler.IsLast();
                blockMmadVGrad(blockGActualShape, pingPongFlag, l0bFlag, isFirstQBlock, isLastQBlock);
                blockMmadKGrad(blockQActualShape, pingPongFlag, l0bFlag, isFirstQBlock, isLastQBlock);

                auto tQS = qBlockScheduler.GetShareTile(tensorQShare, totalSeqLenQ);
                blockMmadQGrad(tQS, blockQActualShape, pingPongFlag, l0bFlag);
            }
            blockMmadQK.ReleaseTensor();
            blockMmadGV.ReleaseTensor();
        }

        Arch::CrossCoreFlag qTransReady{Arch::CrossCoreFlag(Q_TRANS_READY_ID)};
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(qTransReady.id);
    }

    Arch::Resource<ArchTag> resource;
    uint32_t batch{0};
    uint32_t heads{0};
    uint32_t dimQK{0};
    uint32_t dimGV{0};
    uint32_t totalSeqLenQ{0};
    uint32_t totalSeqLenK{0};
    int32_t targetGroupSize{0};
    ElementACC alpha{0.0f};
    ElementACC scale{0.0f};
};

}
