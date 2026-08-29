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

/**
 * @file forward_kernel_mmad_mainloop.hpp
 * @brief HSTU Forward 算子 MMAD 主循环实现
 * @description 实现 Cube 核上的矩阵乘法主循环，包含 QK 和 PV 的调度和执行
 */

#pragma once

#include "catlass/detail/macros.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "../../../catlass_hstu/gemm/block/metadata_row_block_scheduler.hpp"

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
    using ArchTag = typename BlockMmadQK::ArchTag;
    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementV = typename BlockMmadPV::ElementB;

    // ===== HardEvent: MTE2_MTE1 / MTE1_MTE2 (L1 buffer pingpong) =====
    // K:0-1  Q:6-7  V:4-5
    static constexpr uint32_t K_L1_EVENT_ID0 = EVENT_ID0;
    static constexpr uint32_t K_L1_EVENT_ID1 = EVENT_ID1;
    static constexpr uint32_t V_L1_EVENT_ID0 = EVENT_ID4;
    static constexpr uint32_t V_L1_EVENT_ID1 = EVENT_ID5;
    static constexpr uint32_t L1_Q_EVENT_ID0 = EVENT_ID6;
    static constexpr uint32_t L1_Q_EVENT_ID1 = EVENT_ID7;

    // ===== HardEvent: M_MTE1 / MTE1_M (L0 slot handoff) =====
    // L0-0(QK)=2  L0-1(PV)=3
    static constexpr uint32_t L0_HANDOFF_ID0 = EVENT_ID2;
    static constexpr uint32_t L0_HANDOFF_ID1 = EVENT_ID3;

    // ===== HardEvent: M_FIX / FIX_M (L0C/VACC drain) =====
    // QK=4  PV=5
    static constexpr uint32_t EVENT_QK_L0C_FIX_ID = EVENT_ID4;
    static constexpr uint32_t EVENT_PV_L0C_FIX_ID = EVENT_ID5;

    // ===== CrossCore =====
    // QK→PV score ready: 0-2  PV→AIV ready: 3-5
    static constexpr uint32_t QK_READY_ID0 = EVENT_ID0;
    static constexpr uint32_t QK_READY_ID1 = EVENT_ID1;
    static constexpr uint32_t QK_READY_ID2 = EVENT_ID2;
    static constexpr uint32_t PV_READY_ID0 = EVENT_ID3;
    static constexpr uint32_t PV_READY_ID1 = EVENT_ID4;
    static constexpr uint32_t PV_READY_ID2 = EVENT_ID5;
    static constexpr uint32_t TRANS_READY_ID = EVENT_ID6;

    struct Params {
        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrSeqOffsetQ;
        GM_ADDR ptrSeqOffsetK;
        GM_ADDR ptrMetadata;  // 可选: flash_attn_metadata 分核输出;nullptr → 旧设备现算路径

        CATLASS_DEVICE Params() {}
        CATLASS_DEVICE
        Params(GM_ADDR ptrQ_, GM_ADDR ptrK_, GM_ADDR ptrV_, GM_ADDR ptrSeqOffsetQ_, GM_ADDR ptrSeqOffsetK_,
               GM_ADDR ptrMetadata_ = nullptr)
            : ptrQ(ptrQ_),
              ptrK(ptrK_),
              ptrV(ptrV_),
              ptrSeqOffsetQ(ptrSeqOffsetQ_),
              ptrSeqOffsetK(ptrSeqOffsetK_),
              ptrMetadata(ptrMetadata_)
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
    }

    // ============ internal helpers ============

    template <typename ElementT>
    CATLASS_DEVICE auto MakeTNDTensor(AscendC::GlobalTensor<ElementT>& gt, uint32_t seqLen, uint32_t dim)
    {
        auto layout =
            tla::MakeLayout(tla::MakeShape((int64_t)seqLen * heads, dim), tla::MakeStride(dim, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    /// RAII guard: 构造 = SetFlag 释放所有 pipe slot，析构 = WaitFlag 回收
    struct PipeEventGuard {
        CATLASS_DEVICE PipeEventGuard()
        {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(K_L1_EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(K_L1_EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_Q_EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_Q_EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(V_L1_EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(V_L1_EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_QK_L0C_FIX_ID);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_PV_L0C_FIX_ID);
        }
        CATLASS_DEVICE ~PipeEventGuard()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(K_L1_EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(K_L1_EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_Q_EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_Q_EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(V_L1_EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(V_L1_EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_HANDOFF_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_QK_L0C_FIX_ID);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_PV_L0C_FIX_ID);
        }
    };

    CATLASS_DEVICE
    void InitGlobalTensors(const Params& params, AscendC::GlobalTensor<ElementQ>& gQ,
                           AscendC::GlobalTensor<ElementK>& gK, AscendC::GlobalTensor<ElementV>& gV)
    {
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.ptrQ);
        gK.SetGlobalBuffer((__gm__ ElementK*)params.ptrK);
        gV.SetGlobalBuffer((__gm__ ElementV*)params.ptrV);
        gQ.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    template <class TensorQ, class TensorK, class TensorV>
    CATLASS_DEVICE void RunPipeline(BlockMmadQK& blockMmadQK, BlockMmadPV& blockMmadPV,
                                    QBlockScheduler& qBlockScheduler, KBlockScheduler& kBlockScheduler,
                                    TensorQ& tensorQ, TensorK& tensorK, TensorV& tensorV)
    {
        auto tQ = qBlockScheduler.GetTile(tensorQ);
        auto tK = kBlockScheduler.GetTile(tensorK);
        auto tpreV = kBlockScheduler.GetTile(tensorV);
        auto tV = tpreV;

        auto initCoord = tla::MakeCoord(tla::get<0>(tQ.shape()), tla::get<0>(tK.shape()));

        struct StageCoord {
            decltype(initCoord) shape;
            bool isFirst = true;
            bool isLast = false;

            CATLASS_DEVICE StageCoord() = default;
            CATLASS_DEVICE StageCoord(decltype(initCoord) s, bool f, bool l) : shape(s), isFirst(f), isLast(l) {}
        };
        StageCoord stage[3] = {{initCoord, true, false}, {initCoord, true, false}, {initCoord, true, false}};

        auto newShape = initCoord;
        uint32_t transInSlot = 0;

        int endPipe = 0, cnt = 0;
        while (endPipe < 4) {
            bool isFirst = kBlockScheduler.IsFirst();
            bool isLast = kBlockScheduler.IsLast();

            // Stage 0 — MTE2: DMA Q + K from GM to L1
            if (endPipe == 0) {
                if (isFirst)
                    tQ = qBlockScheduler.GetTile(tensorQ);
                blockMmadQK.template LoadGMtoL1<true>(tQ, isFirst);
                tK = kBlockScheduler.GetTile(tensorK);
                blockMmadQK.template LoadGMtoL1<false>(tK);
                newShape = tla::MakeCoord(tla::get<0>(tQ.shape()), tla::get<0>(tK.shape()));
            }

            // Stage 2 — MTE2: DMA V from GM to L1
            if (cnt > 1 && endPipe < 3) {
                blockMmadPV.LoadGMtoL1(tpreV);
                tpreV = tV;
            }
            tV = kBlockScheduler.GetTile(tensorV);

            // Stage 3 — MTE1: Load P + V from L1 to L0
            if (cnt > 2) {
                blockMmadPV.LoadL0(stage[2].shape, dimV);
            }

            // Stage 2 — CUBE: Q*K^T MMAD
            if (cnt > 1 && endPipe < 3) {
                blockMmadQK(stage[1].shape, dimQK);
            }

            // Stage 1 — MTE1: Load Q + K from L1 to L0
            if (cnt > 0 && endPipe < 2) {
                blockMmadQK.LoadL0(stage[0].shape, dimQK, stage[0].isLast);
            }

            // Stage 3 — CUBE: P*V MMAD
            if (cnt > 2) {
                uint32_t curT = transInSlot;
                transInSlot = transInSlot ^ stage[2].isLast;
                blockMmadPV(stage[2].shape, dimV, stage[2].isFirst, stage[2].isLast, curT);
            }

            // Advance + rotate
            // 推进 Q 的前提是 Q 仍有效: Q 耗尽后不再 ++(已耗尽的行调度器 ++ 会下溢),
            // 直接进入排水计数; K 耗尽且 Q 有效时才切到下一 Q 行。
            ++kBlockScheduler;
            if (!kBlockScheduler.IsValid() && qBlockScheduler.IsValid()) {
                ++qBlockScheduler;
                if (qBlockScheduler.IsValid())
                    kBlockScheduler.Init(qBlockScheduler);
            }
            endPipe += !qBlockScheduler.IsValid();
            cnt++;

            stage[2] = stage[1];
            stage[1] = stage[0];
            stage[0] = {newShape, isFirst, isLast};
        }
    }

    CATLASS_DEVICE
    void operator()(const Params& params)
    {
        AscendC::GlobalTensor<ElementQ> gQ;
        AscendC::GlobalTensor<ElementK> gK;
        AscendC::GlobalTensor<ElementV> gV;
        InitGlobalTensors(params, gQ, gK, gV);

        auto tensorQ = MakeTNDTensor(gQ, totalSeqLenQ, dimQK);
        auto tensorK = MakeTNDTensor(gK, totalSeqLenK, dimQK);
        auto tensorV = MakeTNDTensor(gV, totalSeqLenK, dimV);

        static constexpr uint32_t K_L1_EVENT_ID[2] = {K_L1_EVENT_ID0, K_L1_EVENT_ID1};
        static constexpr uint32_t Q_L1_EVENT_ID[2] = {L1_Q_EVENT_ID0, L1_Q_EVENT_ID1};
        static constexpr uint32_t V_L1_EVENT_ID[2] = {V_L1_EVENT_ID0, V_L1_EVENT_ID1};
        BlockMmadQK blockMmadQK(resource, heads, dimQK, K_L1_EVENT_ID, Q_L1_EVENT_ID,
                                {QK_READY_ID0, QK_READY_ID1, QK_READY_ID2}, L0_HANDOFF_ID0, EVENT_QK_L0C_FIX_ID);
        BlockMmadPV blockMmadPV(resource, heads, dimV, V_L1_EVENT_ID, {PV_READY_ID0, PV_READY_ID1, PV_READY_ID2},
                                L0_HANDOFF_ID1, EVENT_PV_L0C_FIX_ID, TRANS_READY_ID);

        // 行(Q)调度器: 经工厂构造。QBlockScheduler 为 RowBlockScheduler/InterleavedRowBlockScheduler 时
        // 忽略 metadata(旧路);为 MetadataRowBlockScheduler 时用 metadata 驱动(新路)。构造代码对两种类型统一。
        QBlockScheduler qBlockScheduler = Gemm::Block::MakeRowScheduler<QBlockScheduler>(
            batch, heads, params.ptrSeqOffsetQ, params.ptrSeqOffsetK, params.ptrMetadata);
        KBlockScheduler kBlockScheduler(batch, heads, params.ptrSeqOffsetK);
        qBlockScheduler.Init();
        kBlockScheduler.Init(qBlockScheduler);
        if (!kBlockScheduler.IsValid() || !qBlockScheduler.IsValid())
            return;

        PipeEventGuard guard;
        RunPipeline(blockMmadQK, blockMmadPV, qBlockScheduler, kBlockScheduler, tensorQ, tensorK, tensorV);
    }

    Arch::Resource<ArchTag> resource;

    uint32_t batch{0};
    uint32_t heads{0};
    uint32_t dimQK{0};
    uint32_t dimV{0};
    uint32_t totalSeqLenQ{0};
    uint32_t totalSeqLenK{0};
};

}  // namespace Catlass::Kernel
