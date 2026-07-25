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
 • @file backward_kernel_epilogue_mainloop.hpp

 • @brief HSTU Backward 算子 Epilogue 主循环实现

 • @description 实现 Vector 核上的 Epilogue 操作，包括 SiLU 激活、RAB 相加、掩码应用、

 •              Score 梯度计算、RAB 梯度计算以及输出转置等

 */

#pragma once

#include "catlass/detail/macros.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "../../../catlass_hstu/kernel/mask/predictor_builder.hpp"
#include "../../../catlass_hstu/gemm/block/metadata_row_block_scheduler.hpp"

namespace Catlass::Kernel {

/**
 • @brief Backward Epilogue 主循环结构体

 • @tparam BlockEpilogueQK_ QK Epilogue Block

 • @tparam BlockEpilogueGV_ GV Epilogue Block

 • @tparam BlockEpilogueKVGrad_ KV 梯度 Epilogue Block

 • @tparam BlockEpilogueQGrad_ Q 梯度 Epilogue Block

 • @tparam QBlockScheduler_ Q 方向块调度器

 • @tparam KBlockScheduler_ K 方向块调度器

 • @description 负责在 AI Core 的 Vector 核上执行所有 Epilogue 操作，

 •              包括 SiLU 激活函数、梯度计算、RAB 处理、数据转置等

 */
template <class BlockEpilogueQK_, class BlockEpilogueGV_, class BlockEpilogueKVGrad_, class BlockEpilogueQGrad_,
          class QBlockScheduler_, class KBlockScheduler_, typename ElementOffset, bool IS_LOCAL, bool IS_CAUSAL,
          bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY, class Predictor>
struct BackwardEpilogueMainloop {
    using BlockEpilogueQK = BlockEpilogueQK_;
    using BlockEpilogueGV = BlockEpilogueGV_;
    using BlockEpilogueKGrad = BlockEpilogueKVGrad_;
    using BlockEpilogueVGrad = BlockEpilogueKVGrad_;
    using BlockEpilogueQGrad = BlockEpilogueQGrad_;
    using QBlockScheduler = QBlockScheduler_;
    using KBlockScheduler = KBlockScheduler_;
    using L1TileShape = typename BlockEpilogueQK_::L1TileShape;

    using ArchTag = typename BlockEpilogueQK::ArchTag;
    using ElementK = typename BlockEpilogueQK::Element;
    using ElementG = typename BlockEpilogueGV::Element;
    using ElementV = typename BlockEpilogueGV::Element;
    using ElementACC = typename BlockEpilogueQK::ElementAccumulator;

    static constexpr uint32_t QK_READY_ID = 0;
    static constexpr uint32_t GV_READY_ID = 1;
    static constexpr uint32_t PROB_READY_ID = 2;
    static constexpr uint32_t GRAB_READY_ID = 3;
    static constexpr uint32_t TRANS_READY_ID = 4;
    static constexpr uint32_t Q_TRANS_READY_ID = 5;

    static constexpr bool IS_LOCAL_V = IS_LOCAL;
    static constexpr bool IS_CAUSAL_V = IS_CAUSAL;
    static constexpr bool IS_CONTEXT_V = IS_CONTEXT;
    static constexpr bool IS_TARGET_V = IS_TARGET;
    static constexpr bool IS_ARBITRARY_V = IS_ARBITRARY;

    // Get BLOCK_M by indices 0
    static constexpr uint32_t BLOCK_M = tla::get<0>(L1TileShape{});
    // Get BLOCK_N by indices 1
    static constexpr uint32_t BLOCK_N = tla::get<1>(L1TileShape{});
    // Get BLOCK_K by indices 2
    static constexpr uint32_t BLOCK_K = tla::get<2>(L1TileShape{});

    static constexpr bool HAS_RAB = BlockEpilogueQK::HAS_RAB;

    struct Params {
        GM_ADDR ptrRab;
        GM_ADDR ptrSeqOffsetQ;
        GM_ADDR ptrSeqOffsetK;
        GM_ADDR ptrQGrad;
        GM_ADDR ptrKGrad;
        GM_ADDR ptrVGrad;
        GM_ADDR ptrRabGrad;
        GM_ADDR ptrQShare;
        GM_ADDR ptrNumContext;
        GM_ADDR ptrNumTarget;
        GM_ADDR ptrMetadata;  // 可选: flash_attn_metadata 分核输出;nullptr → 旧设备现算路径

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrRab_, GM_ADDR ptrSeqOffsetQ_, GM_ADDR ptrSeqOffsetK_, GM_ADDR ptrQGrad_, GM_ADDR ptrKGrad_,
               GM_ADDR ptrVGrad_, GM_ADDR ptrRabGrad_, GM_ADDR ptrQShare_, GM_ADDR ptrNumContext_,
               GM_ADDR ptrNumTarget_, GM_ADDR ptrMetadata_ = nullptr)
            : ptrRab(ptrRab_),
              ptrSeqOffsetQ(ptrSeqOffsetQ_),
              ptrSeqOffsetK(ptrSeqOffsetK_),
              ptrQGrad(ptrQGrad_),
              ptrKGrad(ptrKGrad_),
              ptrVGrad(ptrVGrad_),
              ptrRabGrad(ptrRabGrad_),
              ptrQShare(ptrQShare_),
              ptrNumContext(ptrNumContext_),
              ptrNumTarget(ptrNumTarget_),
              ptrMetadata(ptrMetadata_)
        {
        }
    };

    CATLASS_DEVICE
    BackwardEpilogueMainloop(GM_ADDR ptrTiling_)
    {
        GET_TILING_DATA(tilingData, ptrTiling_);
        batch = tilingData.batch;
        heads = tilingData.heads;
        dimQK = tilingData.dimQK;
        dimGV = tilingData.dimGV;
        maxSeqLenQ = tilingData.maxSeqLenQ;
        maxSeqLenK = tilingData.maxSeqLenK;
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
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }

        CATLASS_DEVICE
        ~PipeEventGuard()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
    };

    struct LoopPipeEventGuard {
        CATLASS_DEVICE
        LoopPipeEventGuard()
        {
            if constexpr (HAS_RAB) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }

        CATLASS_DEVICE
        ~LoopPipeEventGuard()
        {
            if constexpr (HAS_RAB) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }
    };

    CATLASS_DEVICE
    void InitGlobalTensors(const Params& params, AscendC::GlobalTensor<ElementV>& gVGrad,
                           AscendC::GlobalTensor<ElementK>& gKGrad, AscendC::GlobalTensor<ElementG>& gRab,
                           AscendC::GlobalTensor<ElementG>& gRabGrad)
    {
        gVGrad.SetGlobalBuffer((__gm__ ElementV*)params.ptrVGrad);
        gKGrad.SetGlobalBuffer((__gm__ ElementK*)params.ptrKGrad);
        gRab.SetGlobalBuffer((__gm__ ElementG*)params.ptrRab);
        gRabGrad.SetGlobalBuffer((__gm__ ElementG*)params.ptrRabGrad);

        gRab.template SetL2CacheHint<AscendC::CacheRwMode::READ>(AscendC::CacheMode::CACHE_MODE_DISABLE);

        if constexpr (IS_CONTEXT) {
            gNumContext.SetGlobalBuffer((__gm__ ElementOffset*)params.ptrNumContext);
        }
        if constexpr (IS_TARGET) {
            gNumTarget.SetGlobalBuffer((__gm__ ElementOffset*)params.ptrNumTarget);
        }
    }

    template <typename ElementT>
    CATLASS_DEVICE auto MakeTNDTensor(AscendC::GlobalTensor<ElementT>& gt, uint32_t seqLen, uint32_t dim)
    {
        auto layout =
            tla::MakeLayout(tla::MakeShape((int64_t)seqLen, dim), tla::MakeStride((int64_t)dim, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    template <typename ElementT>
    CATLASS_DEVICE auto MakeBNSSTensor(AscendC::GlobalTensor<ElementT>& gt)
    {
        auto stride0 = (int64_t)heads * maxSeqLenQ * maxSeqLenK;
        auto stride1 = (int64_t)maxSeqLenQ * maxSeqLenK;
        auto layout = tla::MakeLayout(tla::MakeShape(batch, heads, maxSeqLenQ, maxSeqLenK),
                                      tla::MakeStride(stride0, stride1, maxSeqLenK, tla::Int<1>{}));
        return tla::MakeTensor(gt, layout, Arch::PositionGM{});
    }

    CATLASS_DEVICE
    void operator()(const Params& params)
    {
        PipeEventGuard pipeEventGuard;

        AscendC::GlobalTensor<ElementV> gVGrad;
        AscendC::GlobalTensor<ElementK> gKGrad;
        AscendC::GlobalTensor<ElementG> gRab;
        AscendC::GlobalTensor<ElementG> gRabGrad;
        InitGlobalTensors(params, gVGrad, gKGrad, gRab, gRabGrad);
        // 使能单指令饱和模式
        if constexpr (IS_CAUSAL || IS_LOCAL || IS_ARBITRARY) {
            AscendC::SetCtrlSpr<60, 60>(0);
        }

        auto tensorVGrad = MakeTNDTensor(gVGrad, totalSeqLenK * heads, dimGV);
        auto tensorKGrad = MakeTNDTensor(gKGrad, totalSeqLenK * heads, dimQK);
        auto tensorRab = MakeBNSSTensor(gRab);
        auto tensorGrab = MakeBNSSTensor(gRabGrad);

        BlockEpilogueQK blockEpilogueQK(alpha, scale, QK_READY_ID, PROB_READY_ID, resource);
        BlockEpilogueGV blockEpilogueGV(GV_READY_ID, GRAB_READY_ID, resource);
        BlockEpilogueKGrad blockEpilogueKGrad(TRANS_READY_ID, (int64_t)heads * dimQK, resource);
        BlockEpilogueVGrad blockEpilogueVGrad(TRANS_READY_ID, (int64_t)heads * dimGV, resource);

        QBlockScheduler qBlockScheduler(batch, heads, params.ptrSeqOffsetQ);
        // 行(K)调度器: 经工厂构造,对 RowBlockScheduler / MetadataRowBlockScheduler 统一(见 mmad mainloop 注释)。
        KBlockScheduler kBlockScheduler = Gemm::Block::MakeRowScheduler<KBlockScheduler>(
            batch, heads, params.ptrSeqOffsetK, params.ptrSeqOffsetQ, params.ptrMetadata);

        kBlockScheduler.Init();
        Predictor predictor;
        for (; kBlockScheduler.IsValid(); ++kBlockScheduler) {
            auto tVg = kBlockScheduler.GetTile(tensorVGrad);
            auto tKg = kBlockScheduler.GetTile(tensorKGrad);

            LoopPipeEventGuard loopPipeEventGuard;

            qBlockScheduler.Init(kBlockScheduler);
            for (; qBlockScheduler.IsValid(); ++qBlockScheduler) {
                auto meta = kBlockScheduler.GetMeta();
                auto blockCoord = tla::MakeCoord(static_cast<uint32_t>(tla::get<0>(meta)),             // batchId
                                                 static_cast<uint32_t>(tla::get<1>(meta)),             // headId
                                                 static_cast<uint32_t>(qBlockScheduler.GetBlockId()),  // qSeqId
                                                 qBlockScheduler.GetRowBlockId());

                auto blockPredParams =
                    predictor.MakeBlockPredParams(blockCoord, this, qBlockScheduler.GetSeqLens(),
                                                  kBlockScheduler.GetCurrentSeqLen(), qBlockScheduler.GetSwizzleDir());
                predictor.Classifier(blockPredParams);
                if (predictor.IsSkip()) {
                    continue;
                }
                auto mapping = qBlockScheduler.GetTileMapping(tVg.coord(), tVg.shape());
                auto coord = tla::get<0>(mapping);
                auto shape = tla::get<1>(mapping);
                blockEpilogueQK(tensorRab, coord, shape, predictor);
                blockEpilogueGV(tensorGrab, coord, shape);
            }
            if (AscendC::GetSubBlockIdx() == 0) {
                blockEpilogueVGrad(tVg);
            } else {
                blockEpilogueKGrad(tKg);
            }
        }

        Arch::CrossCoreFlag qTransReady{Arch::CrossCoreFlag(Q_TRANS_READY_ID)};
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(qTransReady.id);
        AscendC::SyncAll();
        CopyQGrad(params);
    }

    CATLASS_DEVICE
    auto GetQShareTile(AscendC::GlobalTensor<ElementACC>& gQShare)
    {
        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum() * AscendC::GetTaskRatio();

        auto splitNextCore = totalSeqLenQ / coreNum;
        auto splitPrevCore = splitNextCore + 1;
        auto splitCoreIdx = totalSeqLenQ % coreNum;

        uint32_t totalProcSeqs = 0;
        uint32_t coreSeqsOffset = 0;
        if (coreId < splitCoreIdx) {
            totalProcSeqs = splitPrevCore;
            coreSeqsOffset = coreId * splitPrevCore;
        } else if (coreId < coreNum) {
            totalProcSeqs = splitNextCore;
            coreSeqsOffset = splitCoreIdx * splitPrevCore + (coreId - splitCoreIdx) * splitNextCore;
        }

        auto layout =
            tla::MakeLayout(tla::MakeShape((int64_t)totalProcSeqs, dimQK), tla::MakeStride(dimQK, tla::Int<1>{}));
        auto coord = tla::MakeCoord(coreSeqsOffset, 0);
        auto tensorQShareTile = tla::MakeTensor(gQShare, layout, coord, Arch::PositionGM{});
        return tensorQShareTile;
    }

    CATLASS_DEVICE
    void CopyQGrad(const Params& params)
    {
        AscendC::GlobalTensor<ElementG> gQGrad;
        gQGrad.SetGlobalBuffer((__gm__ ElementG*)params.ptrQGrad);
        AscendC::GlobalTensor<ElementACC> gQShare;
        gQShare.SetGlobalBuffer((__gm__ ElementACC*)params.ptrQShare);

        auto tensorQShare = GetQShareTile(gQShare);
        auto totalProcSeqs = tla::get<0>(tensorQShare.shape());
        auto seqOffset = tla::get<0>(tensorQShare.coord());

        auto layoutQGrad = tla::MakeLayout(tla::MakeShape(totalSeqLenQ, heads, dimQK),
                                           tla::MakeStride((int64_t)heads * dimQK, (int64_t)dimQK, tla::Int<1>{}));
        auto tensorQGrad = tla::MakeTensor(gQGrad, layoutQGrad, tla::MakeCoord(seqOffset, 0, 0), Arch::PositionGM{});

        if (totalProcSeqs != 0) {
            BlockEpilogueQGrad blockEpilogueQGrad(resource);
            blockEpilogueQGrad(tensorQGrad, tensorQShare);
        }
    }

    Arch::Resource<ArchTag> resource;

    AscendC::GlobalTensor<ElementOffset> gNumContext;
    AscendC::GlobalTensor<ElementOffset> gNumTarget;

    uint32_t batch{0};
    uint32_t heads{0};
    uint32_t dimQK{0};
    uint32_t dimGV{0};
    uint32_t maxSeqLenQ{0};
    uint32_t maxSeqLenK{0};
    uint32_t totalSeqLenQ{0};
    uint32_t totalSeqLenK{0};
    int32_t targetGroupSize{0};
    ElementACC alpha{0.0f};
    ElementACC scale{0.0f};
};

}  // namespace Catlass::Kernel
