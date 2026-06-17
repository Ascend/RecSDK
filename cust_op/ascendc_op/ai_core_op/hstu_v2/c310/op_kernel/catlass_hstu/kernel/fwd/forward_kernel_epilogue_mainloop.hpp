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
 * @file forward_kernel_epilogue_mainloop.hpp
 * @brief HSTU Forward 算子 Epilogue 主循环实现
 * @description 实现 Vector 核上的 Epilogue 操作，包括 SiLU 激活、RAB 相加、
 *              Mask 应用以及 PV 输出转置
 */

#pragma once

#include "catlass/detail/macros.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Kernel {

/**
 * @brief Forward Epilogue 主循环结构体
 * @tparam BlockEpilogueQK_ QK Epilogue Block
 * @tparam BlockEpiloguePV_ PV Epilogue Block
 * @tparam QBlockScheduler_ Q 方向块调度器
 * @tparam KBlockScheduler_ K 方向块调度器
 * @description 负责在 AI Core 的 Vector 核上执行所有 Epilogue 操作，
 *              包括 SiLU 激活函数、RAB 处理、输出转置等
 */
template <class BlockEpilogueQK_, class BlockEpiloguePV_, class QBlockScheduler_, class KBlockScheduler_>
struct ForwardEpilogueMainloop {
    using BlockEpilogueQK = BlockEpilogueQK_;
    using BlockEpiloguePV = BlockEpiloguePV_;
    using QBlockScheduler = QBlockScheduler_;
    using KBlockScheduler = KBlockScheduler_;

    using ArchTag = typename BlockEpilogueQK::ArchTag;
    using ElementQ = typename BlockEpilogueQK::Element;
    using ElementACC = typename BlockEpilogueQK::ElementAccumulator;

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

    static constexpr uint32_t TRANS_READY_ID = 10;

    struct Params {
        GM_ADDR ptrRab;
        GM_ADDR ptrSeqOffsetQ;
        GM_ADDR ptrSeqOffsetK;
        GM_ADDR ptrAttnOutput;

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR ptrRab_, GM_ADDR ptrSeqOffsetQ_, GM_ADDR ptrSeqOffsetK_, GM_ADDR ptrAttnOutput_)
            : ptrRab(ptrRab_),
              ptrSeqOffsetQ(ptrSeqOffsetQ_),
              ptrSeqOffsetK(ptrSeqOffsetK_),
              ptrAttnOutput(ptrAttnOutput_)
        {
        }
    };

    CATLASS_DEVICE
    ForwardEpilogueMainloop(GM_ADDR ptrTiling_)
    {
        GET_TILING_DATA(tilingData, ptrTiling_);
        batch = tilingData.batch;
        heads = tilingData.heads;
        dimQK = tilingData.dimQK;
        dimV = tilingData.dimV;
        maxSeqLenQ = tilingData.maxSeqLenQ;
        maxSeqLenK = tilingData.maxSeqLenK;
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
        AscendC::GlobalTensor<ElementQ> gRab;
        gRab.SetGlobalBuffer((__gm__ ElementQ*)params.ptrRab);

        auto tensorRab = MakeBNSSTensor(gRab);

        AscendC::GlobalTensor<ElementQ> gAttnOut;
        gAttnOut.SetGlobalBuffer((__gm__ ElementQ*)params.ptrAttnOutput);

        auto tensorAttnOut = MakeTNDTensor(gAttnOut, totalSeqLenQ, dimV);

        BlockEpilogueQK blockEpilogueQK({QK_READY_ID0, QK_READY_ID1, QK_READY_ID2, QK_READY_ID3, QK_READY_ID4},
                                        {PV_READY_ID0, PV_READY_ID1, PV_READY_ID2, PV_READY_ID3, PV_READY_ID4},
                                        resource, alpha, scale);
        BlockEpiloguePV blockEpiloguePV(TRANS_READY_ID, (int64_t)heads * dimQK, resource);

        QBlockScheduler qBlockScheduler(batch, heads, params.ptrSeqOffsetQ, params.ptrSeqOffsetK);
        KBlockScheduler kBlockScheduler(batch, heads, params.ptrSeqOffsetK);

        qBlockScheduler.Init();
        kBlockScheduler.Init(qBlockScheduler);

        for (; qBlockScheduler.IsValid(); ++qBlockScheduler) {
            kBlockScheduler.Init(qBlockScheduler);

            auto tAttnOutQ = qBlockScheduler.GetTile(tensorAttnOut);

            for (; kBlockScheduler.IsValid(); ++kBlockScheduler) {
                auto tAttnOutV = kBlockScheduler.GetTile(tensorAttnOut);
                auto mapping = kBlockScheduler.GetTileMapping(tAttnOutQ.coord(), tAttnOutQ.shape());
                auto coord = tla::get<0>(mapping);
                auto shape = tla::MakeShape(tla::get<0>(tAttnOutQ.shape()), tla::get<0>(tAttnOutV.shape()));
                blockEpilogueQK(tensorRab, coord, shape);
            }
            blockEpiloguePV(tAttnOutQ);
        }
    }

    Arch::Resource<ArchTag> resource;

    uint32_t batch{0};
    uint32_t heads{0};
    uint32_t dimQK{0};
    uint32_t dimV{0};
    uint32_t maxSeqLenQ{0};
    uint32_t maxSeqLenK{0};
    uint32_t totalSeqLenQ{0};
    uint32_t totalSeqLenK{0};
    float alpha{0.0f};
    float scale{0.0f};
};

}  // namespace Catlass::Kernel
