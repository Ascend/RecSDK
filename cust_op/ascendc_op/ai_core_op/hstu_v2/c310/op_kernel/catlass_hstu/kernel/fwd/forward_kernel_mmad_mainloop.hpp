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
 * @file forward_kernel_mmad_mainloop.hpp
 * @brief HSTU Forward 算子 MMAD 主循环实现
 * @description 实现 Cube 核上的矩阵乘法主循环，包含 QK 和 PV 的调度和执行
 */

#pragma once

#include "catlass/detail/macros.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Kernel {

/**
 * @brief Forward MMAD 主循环结构体
 * @tparam BlockMmadQK_ QK 矩阵乘法 Block
 * @tparam BlockMmadPV_ PV 矩阵乘法 Block
 * @tparam QBlockScheduler_ Q 方向块调度器
 * @tparam KBlockScheduler_ K 方向块调度器
 * @description 负责在 AI Core 的 Cube 核上执行 Q*K^T 和 P*V 矩阵乘法
 */
template <class BlockMmadQK_, class BlockMmadPV_, class QBlockScheduler_, class KBlockScheduler_>
struct ForwardMmadMainloop {
    using BlockMmadQK = BlockMmadQK_;
    using BlockMmadPV = BlockMmadPV_;
    using QBlockScheduler = QBlockScheduler_;
    using KBlockScheduler = KBlockScheduler_;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ArchTag = typename BlockMmadQK::ArchTag;
    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementACC = typename BlockMmadPV::ElementAccumulator;

    static constexpr uint32_t EVENT_QK_L1_ID = EVENT_ID0;
    static constexpr uint32_t EVENT_PV_L1_ID = EVENT_ID1;

    static constexpr uint32_t EVENT_L0_ID0 = EVENT_ID0;
    static constexpr uint32_t EVENT_L0_ID1 = EVENT_ID1;

    static constexpr uint32_t EVENT_Q0_ID = EVENT_ID3;
    static constexpr uint32_t EVENT_Q1_ID = EVENT_ID4;
    static constexpr uint32_t EVENT_K_ID = EVENT_ID5;

    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t BLOCK_K = tla::get<2>(L1TileShape{});

    static constexpr uint32_t QK_READY_ID0 = EVENT_ID0;
    static constexpr uint32_t QK_READY_ID1 = EVENT_ID1;
    static constexpr uint32_t QK_READY_ID2 = EVENT_ID2;
    static constexpr uint32_t QK_READY_ID3 = EVENT_ID3;
    static constexpr uint32_t QK_READY_ID4 = EVENT_ID4;
    static constexpr uint32_t PV_READY_ID0 = EVENT_ID5;
    static constexpr uint32_t PV_READY_ID1 = EVENT_ID6;
    static constexpr uint32_t PV_READY_ID2 = EVENT_ID7;
    static constexpr uint32_t PV_READY_ID3 = 8;
    static constexpr uint32_t PV_READY_ID4 = 9;

    struct Params {
        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrSeqOffsetQ;
        GM_ADDR ptrSeqOffsetK;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrQ_, GM_ADDR ptrK_, GM_ADDR ptrV_, GM_ADDR ptrSeqOffsetQ_, GM_ADDR ptrSeqOffsetK_)
            : ptrQ(ptrQ_),
              ptrK(ptrK_),
              ptrV(ptrV_),
              ptrSeqOffsetQ(ptrSeqOffsetQ_),
              ptrSeqOffsetK(ptrSeqOffsetK_)
        {
        }
    };

    CATLASS_DEVICE
    ForwardMmadMainloop(GM_ADDR ptrTiling_)
    {
        GET_TILING_DATA(tilingData, ptrTiling_);
        batch = tilingData.batch;
        heads = tilingData.heads;
        dimQK = tilingData.dimQK;
        dimV = tilingData.dimV;
        totalSeqLenQ = tilingData.totalSeqLenQ;
        totalSeqLenK = tilingData.totalSeqLenK;
        alpha = tilingData.alpha;
        scale = tilingData.scale;
    }

    template <typename ElementT>
    CATLASS_DEVICE auto MakeTNDTensor(AscendC::GlobalTensor<ElementT>& gt, uint32_t seqLen, uint32_t dim)
    {
        auto layout =
            tla::MakeLayout(tla::MakeShape((int64_t)seqLen * heads, dim), tla::MakeStride(dim, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    CATLASS_DEVICE
    void operator()(const Params& params)
    {
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.ptrQ);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.ptrK);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV*)params.ptrV);

        gK.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gV.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);

        auto tensorQ = MakeTNDTensor(gQ, totalSeqLenQ, dimQK);
        auto tensorK = MakeTNDTensor(gK, totalSeqLenK, dimQK);
        auto tensorV = MakeTNDTensor(gV, totalSeqLenK, dimV);

        BlockMmadQK blockMmadQK(resource, heads, dimQK, EVENT_QK_L1_ID, EVENT_PV_L1_ID,
                                {QK_READY_ID0, QK_READY_ID1, QK_READY_ID2, QK_READY_ID3, QK_READY_ID4},
                                {EVENT_L0_ID0, EVENT_L0_ID1}, alpha, scale);

        BlockMmadPV blockMmadPV(resource, heads, dimV, EVENT_QK_L1_ID, EVENT_PV_L1_ID,
                                {PV_READY_ID0, PV_READY_ID1, PV_READY_ID2, PV_READY_ID3, PV_READY_ID4},
                                {EVENT_L0_ID0, EVENT_L0_ID1}, alpha, scale);

        QBlockScheduler qBlockScheduler(batch, heads, params.ptrSeqOffsetQ, params.ptrSeqOffsetK);
        KBlockScheduler kBlockScheduler(batch, heads, params.ptrSeqOffsetK);

        uint32_t l0Apos = 0, l0Bpos = 0, l0Cpos = 0, UBflag = 0;

        qBlockScheduler.Init();
        kBlockScheduler.Init(qBlockScheduler);

        auto tQ = qBlockScheduler.GetTile(tensorQ);
        auto tK = kBlockScheduler.GetTile(tensorK);

        blockMmadQK.template LoadGMtoL1<true>(tQ);
        blockMmadQK.template LoadGMtoL1<false>(tK);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_QK_L1_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_QK_L1_ID);

        blockMmadQK.LoadL0Q(0);
        blockMmadQK.LoadL0K(0, l0Bpos);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Bpos);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Bpos);

        while (qBlockScheduler.IsValid()) {
            for (; kBlockScheduler.IsValid();) {
                bool isFirst = kBlockScheduler.IsFirst();

                auto tV = kBlockScheduler.GetTile(tensorV);
                blockMmadPV.template LoadGMtoL1<false>(tV);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_PV_L1_ID);

                uint32_t mQK = tla::get<0>(tQ.shape());
                uint32_t nQK = tla::get<0>(tK.shape());
                uint32_t kQK = tla::get<1>(tK.shape());
                blockMmadQK(l0Apos, l0Bpos, l0Cpos, UBflag, mQK, nQK, kQK);

                uint32_t mPV = tla::get<0>(tQ.shape());
                uint32_t nPV = tla::get<0>(tK.shape());
                uint32_t kPV = tla::get<1>(tV.shape());

                bool Isfinal = kBlockScheduler.IsLast();
                ++kBlockScheduler;
                bool IsValidK = kBlockScheduler.IsValid();

                if (IsValidK) {
                    auto tK = kBlockScheduler.GetTile(tensorK);
                    blockMmadQK.template LoadGMtoL1<false>(tK);
                } else {
                    ++qBlockScheduler;
                    if (qBlockScheduler.IsValid()) {
                        tQ = qBlockScheduler.GetTile(tensorQ);
                        kBlockScheduler.Init(qBlockScheduler);
                        tK = kBlockScheduler.GetTile(tensorK);
                        blockMmadQK.template LoadGMtoL1<true>(tQ);
                        blockMmadQK.template LoadGMtoL1<false>(tK);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_QK_L1_ID);
                blockMmadPV(l0Apos, l0Bpos, l0Cpos, UBflag, isFirst, Isfinal, mPV, nPV, kPV);
            }
        }
    }

    Arch::Resource<ArchTag> resource;

    uint32_t batch{0};
    uint32_t heads{0};
    uint32_t dimQK{0};
    uint32_t dimV{0};
    uint32_t totalSeqLenQ{0};
    uint32_t totalSeqLenK{0};
    float alpha{0.0f};
    float scale{0.0f};
};

}  // namespace Catlass::Kernel
