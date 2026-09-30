#pragma once

#include "../backward_kernel_mmad_mainloop.hpp"
#include "backward_same_ab_kernel_resource.hpp"

namespace Catlass::Kernel::SameAB {

template <class BlockMmadQK_, class BlockMmadGV_, class BlockMmadVGrad_, class BlockMmadKGrad_, class BlockMmadQGrad_,
          class QBlockScheduler_, class KBlockScheduler_, typename ElementOffset, bool IS_LOCAL, bool IS_CAUSAL,
          bool IS_CONTEXT, bool IS_TARGET, bool IS_ARBITRARY, class Predictor>
struct BackwardSameABKernelMmadMainloop
    : public Catlass::Kernel::BackwardMmadMainloop<
          BlockMmadQK_, BlockMmadGV_, BlockMmadVGrad_, BlockMmadKGrad_, BlockMmadQGrad_, QBlockScheduler_,
          KBlockScheduler_, ElementOffset, IS_LOCAL, IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor> {
    using Base =
        Catlass::Kernel::BackwardMmadMainloop<BlockMmadQK_, BlockMmadGV_, BlockMmadVGrad_, BlockMmadKGrad_,
                                              BlockMmadQGrad_, QBlockScheduler_, KBlockScheduler_, ElementOffset,
                                              IS_LOCAL, IS_CAUSAL, IS_CONTEXT, IS_TARGET, IS_ARBITRARY, Predictor>;
    using Params = typename Base::Params;
    using ArchTag = typename Base::ArchTag;
    using ElementQ = typename Base::ElementQ;
    using ElementK = typename Base::ElementK;
    using ElementG = typename Base::ElementG;
    using ElementV = typename Base::ElementV;
    using ElementACC = typename Base::ElementACC;

    static constexpr uint32_t STAGE_COUNT = BlockMmadQK_::STAGES;
    static_assert(STAGE_COUNT == 2, "Same AB event layout assumes a two-stage L0/L0C ping-pong");

    // L0A/L0B and the QK/GV/dQ L0C ping-pong buffers use event IDs 0/1.
    // dV and dK own fixed L0C accumulators, so their M_FIX/FIX_M handoffs
    // require dedicated event IDs independent of pingPongFlag.
    static constexpr uint32_t EVENT_L0_ID0 = EVENT_ID0;
    static constexpr uint32_t EVENT_L0_ID1 = EVENT_ID1;
    static constexpr uint32_t EVENT_DV_ACC_ID = EVENT_ID2;
    static constexpr uint32_t EVENT_DK_ACC_ID = EVENT_ID3;
    static constexpr uint32_t EVENT_Q0_ID = EVENT_ID2;
    static constexpr uint32_t EVENT_Q1_ID = EVENT_ID3;
    static constexpr uint32_t EVENT_GRAD0_ID = EVENT_ID4;
    static constexpr uint32_t EVENT_GRAD1_ID = EVENT_ID5;
    static constexpr uint32_t EVENT_V_ID = EVENT_ID6;
    static constexpr uint32_t EVENT_K_ID = EVENT_ID7;

    // RAII guard：构造时释放所有会被“先 Wait”的初始空槽，析构时把这些槽回收。
    // 种子集合必须与上面事件号一一对应：L1 槽位是 K/V/Q0/Q1/Grad0/Grad1 六个，
    // M_MTE1 has two L0A/L0B slots. FIX_M also seeds the two fixed dV/dK
    // accumulator ownership tokens.
    struct PipeEventGuard {
        CATLASS_DEVICE PipeEventGuard()
        {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_V_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q0_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q1_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD0_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD1_ID);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0_ID1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_L0_ID1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_DV_ACC_ID);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_DK_ACC_ID);
        }

        CATLASS_DEVICE ~PipeEventGuard()
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_K_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_V_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q0_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_Q1_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD0_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_GRAD1_ID);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_L0_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_L0_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_DV_ACC_ID);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_DK_ACC_ID);
        }
    };

    CATLASS_DEVICE explicit BackwardSameABKernelMmadMainloop(GM_ADDR ptrTiling) : Base(ptrTiling) {}

    CATLASS_DEVICE void operator()(Params const& params)
    {
        PipeEventGuard pipeEventGuard;
        AscendC::GlobalTensor<ElementQ> gQ;
        AscendC::GlobalTensor<ElementK> gK;
        AscendC::GlobalTensor<ElementG> gGrad;
        AscendC::GlobalTensor<ElementV> gV;
        AscendC::GlobalTensor<ElementACC> gQShare;
        this->InitGlobalTensors(params, gQ, gK, gGrad, gV, gQShare);

        auto tensorQ = this->MakeTNDTensor(gQ, this->totalSeqLenQ, this->dimQK);
        auto tensorK = this->MakeTNDTensor(gK, this->totalSeqLenK, this->dimQK);
        auto tensorGrad = this->MakeTNDTensor(gGrad, this->totalSeqLenQ, this->dimGV);
        auto tensorV = this->MakeTNDTensor(gV, this->totalSeqLenK, this->dimGV);
        auto tensorQShare = this->MakeTNDTensor(gQShare, this->totalSeqLenQ, this->dimQK);

        BlockMmadQK_ qk(this->resource, this->heads, this->dimQK, {CrossCoreId::QK0, CrossCoreId::QK1}, EVENT_K_ID,
                        {EVENT_Q0_ID, EVENT_Q1_ID});
        BlockMmadGV_ gv(this->resource, this->heads, this->dimGV, {CrossCoreId::GV0, CrossCoreId::GV1}, EVENT_V_ID,
                        {EVENT_GRAD0_ID, EVENT_GRAD1_ID});
        BlockMmadVGrad_ dV(this->resource, {CrossCoreId::PROB0, CrossCoreId::PROB1}, CrossCoreId::V_TRANS,
                           CrossCoreId::V_UBFREE, EVENT_DV_ACC_ID, {EVENT_GRAD0_ID, EVENT_GRAD1_ID});
        BlockMmadKGrad_ dK(this->resource, {CrossCoreId::GRAB0, CrossCoreId::GRAB1}, CrossCoreId::K_TRANS,
                           CrossCoreId::K_UBFREE, EVENT_DK_ACC_ID, {EVENT_Q0_ID, EVENT_Q1_ID});
        BlockMmadQGrad_ dQ(this->resource);

        QBlockScheduler_ qScheduler(this->batch, this->heads, params.ptrSeqOffsetQ);
        KBlockScheduler_ kScheduler = Gemm::Block::MakeRowScheduler<KBlockScheduler_>(
            this->batch, this->heads, params.ptrSeqOffsetK, params.ptrSeqOffsetQ, params.ptrMetadata);
        if constexpr (IS_TARGET) {
            kScheduler.EnableTargetWorkload(params.ptrNumContext, params.ptrNumTarget, this->targetGroupSize,
                                            IS_CONTEXT);
        }
        Predictor predictor0;
        Predictor predictor1;
        predictor0.Construct(this, params);
        predictor1.Construct(this, params);
        using L1ReuseCache = Gemm::Block::L1BlockReuseCache<Base::MAX_L1_REUSE_BLOCKS>;
        // A block may alternate physical stages when a K row contains an odd
        // number of valid Q blocks. Keep independent residency tags per stage
        // so a swizzle hit never reuses data stored in the other L1 buffer.
        L1ReuseCache qReuseCache[STAGE_COUNT] = {L1ReuseCache(this->GetL1ReuseBlockCount(this->dimQK)),
                                                 L1ReuseCache(this->GetL1ReuseBlockCount(this->dimQK))};
        L1ReuseCache gradReuseCache[STAGE_COUNT] = {L1ReuseCache(this->GetL1ReuseBlockCount(this->dimGV)),
                                                    L1ReuseCache(this->GetL1ReuseBlockCount(this->dimGV))};
        uint32_t pingPongFlag = 0;
        uint32_t l0bFlag = 0;
        uint32_t logicalStage = 0;

        kScheduler.Init();
        for (; kScheduler.IsValid(); ++kScheduler) {
            auto kMeta = kScheduler.GetMeta();
            auto batchId = static_cast<uint32_t>(tla::get<0>(kMeta));
            auto headId = static_cast<uint32_t>(tla::get<1>(kMeta));
            for (uint32_t stage = 0; stage < STAGE_COUNT; ++stage) {
                qReuseCache[stage].UpdateContext(batchId, headId);
                gradReuseCache[stage].UpdateContext(batchId, headId);
            }
            auto tK = kScheduler.GetTile(tensorK);
            auto tV = kScheduler.GetTile(tensorV);
            qk.AcquireTensor(tK);
            gv.AcquireTensor(tV);
            qScheduler.Init(kScheduler);

            while (qScheduler.IsValid()) {
                auto meta0 = kScheduler.GetMeta();
                auto coord0 =
                    tla::MakeCoord(static_cast<uint32_t>(tla::get<0>(meta0)), static_cast<uint32_t>(tla::get<1>(meta0)),
                                   static_cast<uint32_t>(qScheduler.GetBlockId()), qScheduler.GetRowBlockId());
                auto pred0 = predictor0.MakeBlockPredParams(coord0, this, qScheduler.GetSeqLens(),
                                                            kScheduler.GetCurrentSeqLen(), qScheduler.GetSwizzleDir());
                predictor0.Classifier(pred0);
                if (predictor0.IsSkip()) {
                    ++qScheduler;
                    continue;
                }

                auto q0 = qScheduler.GetTile(tensorQ);
                auto g0 = qScheduler.GetTile(tensorGrad);
                auto qs0 = qScheduler.GetShareTile(tensorQShare, this->totalSeqLenQ);
                GemmCoord qShape0(tla::get<0>(q0.shape()), tla::get<0>(tK.shape()), this->dimQK);
                GemmCoord gShape0(tla::get<0>(g0.shape()), tla::get<0>(tV.shape()), this->dimGV);
                bool first0 = predictor0.IsInnerLoopFirstQBlock(pred0);
                bool last0 = predictor0.IsInnerLoopLastQBlock(pred0);
                auto qBlockId0 = qScheduler.GetBlockId();
                auto triggerSwizzle0 = qScheduler.GetTriggerSwizzle();
                auto swizzlePosition0 = qScheduler.GetSwizzlePosition();
                uint32_t stage0 = logicalStage % STAGE_COUNT;
                auto qReuse0 = qReuseCache[stage0].Probe(qBlockId0, triggerSwizzle0, swizzlePosition0);
                auto gradReuse0 = gradReuseCache[stage0].Probe(qBlockId0, triggerSwizzle0, swizzlePosition0);
                qk(q0, tK, pingPongFlag, l0bFlag, qReuse0.canReuse, qReuse0.slot);
                qReuseCache[stage0].Commit(qReuse0.slot, qBlockId0);
                gv(g0, tV, pingPongFlag, l0bFlag, gradReuse0.canReuse, gradReuse0.slot);
                gradReuseCache[stage0].Commit(gradReuse0.slot, qBlockId0);
                ++logicalStage;
                ++qScheduler;

                bool hasSecond = false;
                auto q1 = q0;
                auto g1 = g0;
                auto qs1 = qs0;
                GemmCoord qShape1 = qShape0;
                GemmCoord gShape1 = gShape0;
                bool first1 = false;
                bool last1 = false;
                typename L1ReuseCache::ProbeResult qReuse1;
                typename L1ReuseCache::ProbeResult gradReuse1;
                while (qScheduler.IsValid()) {
                    auto meta1 = kScheduler.GetMeta();
                    auto coord1 = tla::MakeCoord(
                        static_cast<uint32_t>(tla::get<0>(meta1)), static_cast<uint32_t>(tla::get<1>(meta1)),
                        static_cast<uint32_t>(qScheduler.GetBlockId()), qScheduler.GetRowBlockId());
                    auto pred1 =
                        predictor1.MakeBlockPredParams(coord1, this, qScheduler.GetSeqLens(),
                                                       kScheduler.GetCurrentSeqLen(), qScheduler.GetSwizzleDir());
                    predictor1.Classifier(pred1);
                    if (predictor1.IsSkip()) {
                        ++qScheduler;
                        continue;
                    }
                    q1 = qScheduler.GetTile(tensorQ);
                    g1 = qScheduler.GetTile(tensorGrad);
                    qs1 = qScheduler.GetShareTile(tensorQShare, this->totalSeqLenQ);
                    qShape1 = GemmCoord(tla::get<0>(q1.shape()), tla::get<0>(tK.shape()), this->dimQK);
                    gShape1 = GemmCoord(tla::get<0>(g1.shape()), tla::get<0>(tV.shape()), this->dimGV);
                    first1 = predictor1.IsInnerLoopFirstQBlock(pred1);
                    last1 = predictor1.IsInnerLoopLastQBlock(pred1);
                    auto qBlockId1 = qScheduler.GetBlockId();
                    auto triggerSwizzle1 = qScheduler.GetTriggerSwizzle();
                    auto swizzlePosition1 = qScheduler.GetSwizzlePosition();
                    uint32_t stage1 = logicalStage % STAGE_COUNT;
                    qReuse1 = qReuseCache[stage1].Probe(qBlockId1, triggerSwizzle1, swizzlePosition1);
                    gradReuse1 = gradReuseCache[stage1].Probe(qBlockId1, triggerSwizzle1, swizzlePosition1);
                    qk(q1, tK, pingPongFlag, l0bFlag, qReuse1.canReuse, qReuse1.slot);
                    qReuseCache[stage1].Commit(qReuse1.slot, qBlockId1);
                    gv(g1, tV, pingPongFlag, l0bFlag, gradReuse1.canReuse, gradReuse1.slot);
                    gradReuseCache[stage1].Commit(gradReuse1.slot, qBlockId1);
                    ++logicalStage;
                    hasSecond = true;
                    ++qScheduler;
                    break;
                }

                dV(gShape0, pingPongFlag, l0bFlag, first0, last0, gradReuse0.slot);
                dK(qShape0, pingPongFlag, l0bFlag, first0, last0, qReuse0.slot);
                dQ(qs0, qShape0, pingPongFlag, l0bFlag);
                if (hasSecond) {
                    dV(gShape1, pingPongFlag, l0bFlag, first1, last1, gradReuse1.slot);
                    dK(qShape1, pingPongFlag, l0bFlag, first1, last1, qReuse1.slot);
                    dQ(qs1, qShape1, pingPongFlag, l0bFlag);
                }
            }
            qk.ReleaseTensor();
            gv.ReleaseTensor();
        }
        Arch::CrossCoreFlag qTransReady{Arch::CrossCoreFlag(CrossCoreId::QTRANS)};
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(qTransReady.id);
    }
};

}  // namespace Catlass::Kernel::SameAB
