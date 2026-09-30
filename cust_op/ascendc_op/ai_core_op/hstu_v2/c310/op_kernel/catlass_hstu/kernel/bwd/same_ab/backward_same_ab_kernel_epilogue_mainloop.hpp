#pragma once

#include "../backward_kernel_epilogue_mainloop.hpp"
#include "backward_same_ab_kernel_resource.hpp"

namespace Catlass::Kernel::SameAB {

template <class BlockEpilogueQK_, class BlockEpilogueGV_, class BlockEpilogueTransOut_, class BlockEpilogueQGrad_,
          class QBlockScheduler_, class KBlockScheduler_, typename ElementOffset, bool IS_LOCAL, bool IS_CAUSAL,
          bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY, class Predictor>
struct BackwardSameABKernelEpilogueMainloop
    : public Catlass::Kernel::BackwardEpilogueMainloop<
          BlockEpilogueQK_, BlockEpilogueGV_, BlockEpilogueTransOut_, BlockEpilogueQGrad_, QBlockScheduler_,
          KBlockScheduler_, ElementOffset, IS_LOCAL, IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor> {
    using Base = Catlass::Kernel::BackwardEpilogueMainloop<
        BlockEpilogueQK_, BlockEpilogueGV_, BlockEpilogueTransOut_, BlockEpilogueQGrad_, QBlockScheduler_,
        KBlockScheduler_, ElementOffset, IS_LOCAL, IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor>;
    using Params = typename Base::Params;
    using ArchTag = typename Base::ArchTag;
    using ElementV = typename Base::ElementV;
    using ElementK = typename Base::ElementK;
    using ElementG = typename Base::ElementG;
    using TransTileBuffer = typename BlockEpilogueTransOut_::TileBuffer;
    static constexpr bool HAS_RAB = BlockEpilogueQK_::HAS_RAB;

    struct PipeEventGuard {
        CATLASS_DEVICE
        PipeEventGuard()
        {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(BlockEpilogueTransOut_::TRANS_EVENT_ID);
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
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(BlockEpilogueTransOut_::TRANS_EVENT_ID);
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

    CATLASS_DEVICE explicit BackwardSameABKernelEpilogueMainloop(GM_ADDR ptrTiling) : Base(ptrTiling) {}

    CATLASS_DEVICE void operator()(Params const& params)
    {
        PipeEventGuard pipeEventGuard;
        AscendC::GlobalTensor<ElementV> gVGrad;
        AscendC::GlobalTensor<ElementK> gKGrad;
        AscendC::GlobalTensor<ElementG> gRab;
        AscendC::GlobalTensor<ElementG> gRabGrad;
        this->InitGlobalTensors(params, gVGrad, gKGrad, gRab, gRabGrad);
        if constexpr (IS_CAUSAL || IS_LOCAL || IS_ARBITRARY) {
            AscendC::SetCtrlSpr<60, 60>(0);
        }

        auto tensorVGrad = this->MakeTNDTensor(gVGrad, this->totalSeqLenK * this->heads, this->dimGV);
        auto tensorKGrad = this->MakeTNDTensor(gKGrad, this->totalSeqLenK * this->heads, this->dimQK);
        auto tensorRab = this->MakeBNSSTensor(gRab);
        auto tensorGrab = this->MakeBNSSTensor(gRabGrad);

        BlockEpilogueQK_ scoreGrad(this->alpha, this->scale, {CrossCoreId::QK0, CrossCoreId::QK1},
                                   {CrossCoreId::PROB0, CrossCoreId::PROB1}, this->resource);
        BlockEpilogueGV_ rabGrad({CrossCoreId::GV0, CrossCoreId::GV1}, {CrossCoreId::GRAB0, CrossCoreId::GRAB1},
                                 this->resource);
        rabGrad.SetGvScale(this->scale * this->alpha);
        BlockEpilogueTransOut_ kTrans(CrossCoreId::K_TRANS, CrossCoreId::K_UBFREE,
                                      static_cast<int64_t>(this->heads) * this->dimQK, TransTileBuffer::K_TRANS_IN,
                                      this->resource);
        BlockEpilogueTransOut_ vTrans(CrossCoreId::V_TRANS, CrossCoreId::V_UBFREE,
                                      static_cast<int64_t>(this->heads) * this->dimGV, TransTileBuffer::V_TRANS_IN,
                                      this->resource);

        QBlockScheduler_ qScheduler(this->batch, this->heads, params.ptrSeqOffsetQ);
        KBlockScheduler_ kScheduler = Gemm::Block::MakeRowScheduler<KBlockScheduler_>(
            this->batch, this->heads, params.ptrSeqOffsetK, params.ptrSeqOffsetQ, params.ptrMetadata);
        // V_UBFREE/K_UBFREE are read-complete notifications, not initial
        // buffer-ownership tokens. Cube publishes TRANS first and then waits
        // until both AIVs have consumed the corresponding FP32 TransIn. GM
        // writeback continues independently from the shared TransOut buffer.
        if constexpr (IS_TARGET) {
            kScheduler.EnableTargetWorkload(params.ptrNumContext, params.ptrNumTarget, this->targetGroupSize,
                                            IS_CONTEXT);
        }
        Predictor predictor0;
        Predictor predictor1;
        predictor0.Construct(this, params);
        predictor1.Construct(this, params);

        kScheduler.Init();
        for (; kScheduler.IsValid(); ++kScheduler) {
            auto tVg = kScheduler.GetTile(tensorVGrad);
            auto tKg = kScheduler.GetTile(tensorKGrad);
            LoopPipeEventGuard loopPipeEventGuard;
            qScheduler.Init(kScheduler);

            while (qScheduler.IsValid()) {
                auto meta0 = kScheduler.GetMeta();
                auto blockCoord0 =
                    tla::MakeCoord(static_cast<uint32_t>(tla::get<0>(meta0)), static_cast<uint32_t>(tla::get<1>(meta0)),
                                   static_cast<uint32_t>(qScheduler.GetBlockId()), qScheduler.GetRowBlockId());
                auto pred0 = predictor0.MakeBlockPredParams(blockCoord0, this, qScheduler.GetSeqLens(),
                                                            kScheduler.GetCurrentSeqLen(), qScheduler.GetSwizzleDir());
                predictor0.Classifier(pred0);
                if (predictor0.IsSkip()) {
                    ++qScheduler;
                    continue;
                }
                auto mapping0 = qScheduler.GetTileMapping(tVg.coord(), tVg.shape());
                auto coord0 = tla::get<0>(mapping0);
                auto shape0 = tla::get<1>(mapping0);
                scoreGrad(tensorRab, coord0, shape0, predictor0);
                rabGrad(tensorGrab, coord0, shape0);
                ++qScheduler;

                while (qScheduler.IsValid()) {
                    auto meta1 = kScheduler.GetMeta();
                    auto blockCoord1 = tla::MakeCoord(
                        static_cast<uint32_t>(tla::get<0>(meta1)), static_cast<uint32_t>(tla::get<1>(meta1)),
                        static_cast<uint32_t>(qScheduler.GetBlockId()), qScheduler.GetRowBlockId());
                    auto pred1 =
                        predictor1.MakeBlockPredParams(blockCoord1, this, qScheduler.GetSeqLens(),
                                                       kScheduler.GetCurrentSeqLen(), qScheduler.GetSwizzleDir());
                    predictor1.Classifier(pred1);
                    if (predictor1.IsSkip()) {
                        ++qScheduler;
                        continue;
                    }
                    auto mapping1 = qScheduler.GetTileMapping(tVg.coord(), tVg.shape());
                    auto coord1 = tla::get<0>(mapping1);
                    auto shape1 = tla::get<1>(mapping1);
                    scoreGrad(tensorRab, coord1, shape1, predictor1);
                    rabGrad(tensorGrab, coord1, shape1);
                    ++qScheduler;
                    break;
                }
            }
            // Same AB: both AIVs execute the identical output sequence. Each
            // call copies only the local M half produced by Fixpipe SPLIT_M.
            vTrans(tVg);
            kTrans(tKg);
        }

        Arch::CrossCoreFlag qTransReady{Arch::CrossCoreFlag(CrossCoreId::QTRANS)};
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(qTransReady.id);
        AscendC::SyncAll();
        this->CopyQGrad(params);
    }
};

}  // namespace Catlass::Kernel::SameAB
