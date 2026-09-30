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

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Kernel::SameAB {

// Same AB 跨核同步的 flagId 分配表。
//
// Same AB 只用核间同步模式 0x2（AIC <-> 两个 AIV），该模式合法的 flagId 范围是
// 0-15；模式 4 才有的第二档（16-31，AIV0/AIV1 各一档）在这里不可用，而且两个
// AIV 访问的是同一个 id。因此 legacy 的 `i * SYNC_FLAG_ID_MAX + flag` 展开方式
// 不能用在 Same AB 上，必须为「每条通道 x 每个 stage」直接分配一个互不相同的
// 0-15 内的 id。11-14 被 SyncAll 占用（11=AIC, 12=AIV, 13=AIC->AIV,
// 14=AIV only），所以可用区间取 0-10。
struct CrossCoreId {
    // dV and dK are independent producer/consumer channels. K_TRANS shares
    // ID 5 with QTRANS only across disjoint phases: QTRANS starts after every
    // K tile (and therefore every kTrans) has completed.
    static constexpr uint32_t V_TRANS = 4;
    static constexpr uint32_t K_TRANS = 5;
    static constexpr uint32_t V_UBFREE = 10;
    // dV Trans and dK Trans are strictly serialized by the Cube-side
    // post-Flush wait, so they share one completion rendezvous channel.
    static constexpr uint32_t K_UBFREE = 10;
    // AIC -> AIV
    static constexpr uint32_t QK0 = 0;  // QK Fixpipe 完成  -> AIV ScoreGrad
    static constexpr uint32_t QK1 = 1;
    static constexpr uint32_t GV0 = 2;  // GV Fixpipe 完成  -> AIV RabGrad
    static constexpr uint32_t GV1 = 3;
    // QTRANS waits on PIPE_MTE2 while K_TRANS waits on PIPE_V. Keep their
    // cross-core counters independent even though their source-level phases
    // appear sequential.
    static constexpr uint32_t QTRANS = 15;

    // AIV -> AIC
    static constexpr uint32_t PROB0 = 6;  // AIV ScoreGrad 完成 -> AIC dV
    static constexpr uint32_t PROB1 = 7;
    static constexpr uint32_t GRAB0 = 8;  // AIV RabGrad 完成   -> AIC dK
    static constexpr uint32_t GRAB1 = 9;
};

static_assert(CrossCoreId::V_UBFREE < AscendC::SYNC_FLAG_ID_MAX && CrossCoreId::K_UBFREE < AscendC::SYNC_FLAG_ID_MAX,
              "Same AB V/K cross-core flagIds must stay inside the mode 0x2 range 0-15");

enum class BufferTag : uint8_t {
    QK_MMAD = 0,
    GV_MMAD,
    V_GRAD_MMAD,
    K_GRAD_MMAD,
    Q_GRAD_MMAD,
    SCORE_GRAD_EPILOGUE,
    RAB_GRAD_EPILOGUE,
    TRANS_KV_GRAD_EPILOGUE,
    TRANS_Q_GRAD_EPILOGUE
};

template <class ArchTag_, class L1TileShape_, class L0TileShape_, class Element_, class ElementAccumulator_,
          uint32_t STAGES_, bool HAS_RAB_, bool HAS_MASK_>
struct KernelResource {
    using ArchTag = ArchTag_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    static constexpr uint32_t STAGES = STAGES_;
    static constexpr uint32_t AIV_PER_AIC = 2;
    static constexpr bool HAS_RAB = HAS_RAB_;
    static constexpr bool HAS_MASK = HAS_MASK_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1_MN_ELEM = L1_TILE_M * L1_TILE_N;
    static constexpr uint32_t L1_MK_ELEM = L1_TILE_M * L1_TILE_K;
    static constexpr uint32_t L1_NK_ELEM = L1_TILE_N * L1_TILE_K;
    static constexpr uint32_t L0_MN_ELEM = L0_TILE_M * L0_TILE_N;
    static constexpr uint32_t L0_MK_ELEM = L0_TILE_M * L0_TILE_K;
    static constexpr uint32_t L0_NK_ELEM = L0_TILE_N * L0_TILE_K;

    // A complete logical QK/GV result is split across the two AIVs. With the
    // Same AB 128 x 128 QK tile, each AIV owns one 64 x 128 half tile.
    static_assert(L1_TILE_M % AIV_PER_AIC == 0, "Same AB requires an even L1 M tile");
    static constexpr uint32_t AIV_TILE_M = L1_TILE_M / AIV_PER_AIC;
    static constexpr uint32_t AIV_MN_ELEM = AIV_TILE_M * L1_TILE_N;
    static constexpr uint32_t AIV_MK_ELEM = AIV_TILE_M * L1_TILE_K;

    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "Same AB backward currently supports Ascend950 only");
    static_assert(STAGES == 2, "Same AB backward currently requires two pipeline stages");

    template <class T, uint32_t ElemCount, uint32_t UpperBound = 0, uint32_t BufferCount = 1>
    struct BufferPosition {
        static_assert(BufferCount >= 1 && BufferCount <= 2, "buffer count must be in [1, 2]");

        static constexpr uint32_t bytes = ElemCount * sizeof(T);
        static constexpr uint32_t bufferCnt = BufferCount;
        static constexpr uint32_t upperBound = UpperBound;
        static constexpr uint32_t lowerBound = upperBound + bytes * bufferCnt;

        template <uint32_t BufferIndex>
        static CATLASS_DEVICE constexpr uint32_t Get()
        {
            static_assert(BufferIndex < bufferCnt, "buffer index out of range");
            return upperBound + bytes * BufferIndex;
        }
    };

    template <bool Enabled, uint32_t ElemCount>
    struct Optional {
        static constexpr uint32_t value = Enabled ? ElemCount : 0;
    };

    template <uint32_t A, uint32_t B>
    struct Max {
        static constexpr uint32_t value = A > B ? A : B;
    };

    // L1 and all L0 memories belong to the AIC and therefore retain complete
    // logical tiles. Only per-AIV UB intermediates are split below.
    struct L1 {
        static constexpr BufferPosition<Element, L1_NK_ELEM> key;
        // Two complete logical Q tiles are resident for the two-stage
        // QK0-GV0-QK1-GV1 pipeline.
        static constexpr BufferPosition<Element, L1_MK_ELEM, key.lowerBound, STAGES> query;
        static constexpr BufferPosition<Element, L1_NK_ELEM, query.lowerBound> value;
        // GV follows the same two-complete-tile residency contract as QK.
        static constexpr BufferPosition<Element, L1_MK_ELEM, value.lowerBound, STAGES> grad;

        // Vector returns two complete logical Prob tiles to Cube.
        static constexpr BufferPosition<Element, L1_MN_ELEM, grad.lowerBound, STAGES> prob;
        // Grab has independent storage. Same AB TILE_K=128 uses a 128 x 128
        // logical tile specifically so Grad and Grab do not need aliasing.
        static constexpr BufferPosition<Element, L1_MN_ELEM, prob.lowerBound, STAGES> grab;

        static_assert(grab.lowerBound <= ArchTag::L1_SIZE, "Same AB backward exceeds L1 capacity");
    };

    struct L0A {
        static constexpr uint32_t PINGPONG_ELEM = ArchTag::L0A_SIZE / STAGES / sizeof(Element);
        static constexpr BufferPosition<Element, PINGPONG_ELEM, 0, STAGES> buffer;

        static_assert(buffer.lowerBound <= ArchTag::L0A_SIZE, "Same AB backward exceeds L0A capacity");
    };

    struct L0B {
        static constexpr uint32_t PINGPONG_ELEM = ArchTag::L0B_SIZE / STAGES / sizeof(Element);
        static constexpr BufferPosition<Element, PINGPONG_ELEM, 0, STAGES> buffer;

        static_assert(buffer.lowerBound <= ArchTag::L0B_SIZE, "Same AB backward exceeds L0B capacity");
    };

    struct L0C {
        static constexpr uint32_t PINGPONG_ELEM = ArchTag::L0C_SIZE / STAGES / sizeof(ElementAccumulator) / 2;
        static constexpr BufferPosition<ElementAccumulator, PINGPONG_ELEM, 0, STAGES> buffer;
        static constexpr BufferPosition<ElementAccumulator, L0_NK_ELEM, buffer.lowerBound> gradVAcc;
        static constexpr BufferPosition<ElementAccumulator, L0_NK_ELEM, gradVAcc.lowerBound> gradKAcc;

        static_assert(gradKAcc.lowerBound <= ArchTag::L0C_SIZE, "Same AB backward exceeds L0C capacity");
    };

    struct UB {
        // All sizes are per AIV and cover half of a complete L1 M tile.
        // The smaller Same AB tile permits independent storage without
        // cross-phase address aliasing.
        // QK0/QK1 may both be in flight. Keep one FP32 score tile per stage
        // so the second Fixpipe cannot overwrite data still owned by Vector.
        static constexpr BufferPosition<ElementAccumulator, AIV_MN_ELEM, 0, STAGES> score;
        static constexpr BufferPosition<Element, Optional<HAS_RAB, AIV_MN_ELEM>::value, score.lowerBound> rab;
        static constexpr BufferPosition<Element, Optional<HAS_MASK, AIV_MN_ELEM>::value, rab.lowerBound> mask;
        static constexpr BufferPosition<Element, AIV_MN_ELEM, mask.lowerBound> prob;
        static constexpr BufferPosition<Element, AIV_MN_ELEM, prob.lowerBound, STAGES> grabPart;
        // QK/GV stages overlap with Vector Score/RabGrad consumption, so the
        // score and gs double buffers must remain independent.
        static constexpr BufferPosition<ElementAccumulator, AIV_MN_ELEM, grabPart.lowerBound, STAGES> gs;
        static constexpr BufferPosition<Element, AIV_MN_ELEM, gs.lowerBound> grab;
        static constexpr uint32_t EPILOGUE_BYTES = grab.lowerBound;
        // TILE_K=128 must reuse the epilogue address range to fit the worst-case
        // RAB+mask configuration in UB.  Its dV destination only overlaps score,
        // whose lifetime has ended when PROB_READY is consumed.
        //
        // TILE_K=256 has a much smaller epilogue footprint while its dV/dK
        // transpose tiles remain 32 KiB each.  Place the whole KV transpose
        // workspace after the epilogue so dV Fixpipe cannot overwrite the
        // grabPart/gs data still being consumed by RabGrad.
        static constexpr uint32_t KV_TRANS_BASE = L0_TILE_K == 256 ? EPILOGUE_BYTES : 0;
        // Prob already includes scale. Keep the Fixpipe destination in FP32,
        // matching L0C; Vector casts the per-AIV half to Element before GM.
        static constexpr uint32_t AIV_NK_ELEM = CeilDiv<AIV_PER_AIC>(L0_NK_ELEM);
        // Separate FP32 destinations are required once dV and dK use
        // independent ready/free handshakes. Otherwise dK can overwrite dV
        // while vTrans is still reading it.
        static constexpr BufferPosition<ElementAccumulator, AIV_NK_ELEM, KV_TRANS_BASE> transVGradIn;
        static constexpr BufferPosition<ElementAccumulator, AIV_NK_ELEM, transVGradIn.lowerBound> transKGradIn;
        // vTrans and kTrans execute serially on each AIV, so their cast output
        // remains safely shared under the local MTE3_V/V_MTE3 event.
        static constexpr BufferPosition<Element, AIV_NK_ELEM, transKGradIn.lowerBound> transKVGradOut;

        static constexpr uint32_t KV_TRANS_BYTES = Max<EPILOGUE_BYTES, transKVGradOut.lowerBound>::value;
        static_assert(EPILOGUE_BYTES <= ArchTag::UB_SIZE, "Same AB backward exceeds UB epilogue capacity");
        static_assert(KV_TRANS_BYTES <= ArchTag::UB_SIZE, "Same AB backward exceeds UB KV transpose capacity");

        // Q-grad transpose is a separate phase. Keep the proven full-UB
        // ping-pong allocation until its Same AB ownership is implemented.
        static constexpr uint32_t ELEM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM =
            RoundDown(ArchTag::UB_SIZE / STAGES / (sizeof(ElementAccumulator) + sizeof(Element)), ELEM_PER_C0);
        static constexpr BufferPosition<ElementAccumulator, TILE_Q_GRAD_MAX_ELEM, 0, STAGES> transQGradIn;
        static constexpr BufferPosition<Element, TILE_Q_GRAD_MAX_ELEM, transQGradIn.lowerBound, STAGES> transQGradOut;

        static_assert(transQGradOut.lowerBound <= ArchTag::UB_SIZE,
                      "Same AB backward exceeds UB Q-grad transpose capacity");
    };

    // Top-level footprints make capacity contracts easy to instantiate from
    // KernelBuilder without referring through dependent nested types.
    static constexpr uint32_t L1_BYTES = L1::grab.lowerBound;
    static constexpr uint32_t L0A_BYTES = L0A::buffer.lowerBound;
    static constexpr uint32_t L0B_BYTES = L0B::buffer.lowerBound;
    static constexpr uint32_t L0C_BYTES = L0C::gradKAcc.lowerBound;
    // Include the trailing KV transpose buffer in the externally checked
    // epilogue footprint; checking only grab.lowerBound would under-report UB.
    static constexpr uint32_t UB_EPILOGUE_BYTES = UB::KV_TRANS_BYTES;
    static constexpr uint32_t UB_Q_GRAD_BYTES = UB::transQGradOut.lowerBound;
};

template <class Resource, BufferTag Tag>
struct TileBuffer {
    static_assert(DEPENDENT_FALSE<Resource>, "Same AB TileBuffer specialization failed");
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::QK_MMAD> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t AIV_PER_AIC = Resource::AIV_PER_AIC;
    static constexpr uint32_t L1B = Resource::L1::key.template Get<0>();
    static constexpr uint32_t L1A[STAGES] = {Resource::L1::query.template Get<0>(),
                                             Resource::L1::query.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {Resource::L0A::buffer.template Get<0>(),
                                             Resource::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {Resource::L0B::buffer.template Get<0>(),
                                             Resource::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[STAGES] = {Resource::L0C::buffer.template Get<0>(),
                                             Resource::L0C::buffer.template Get<1>()};
    static constexpr uint32_t DST[STAGES] = {Resource::UB::score.template Get<0>(),
                                             Resource::UB::score.template Get<1>()};
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::GV_MMAD> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t AIV_PER_AIC = Resource::AIV_PER_AIC;
    static constexpr uint32_t L1B = Resource::L1::value.template Get<0>();
    static constexpr uint32_t L1A[STAGES] = {Resource::L1::grad.template Get<0>(),
                                             Resource::L1::grad.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {Resource::L0A::buffer.template Get<0>(),
                                             Resource::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {Resource::L0B::buffer.template Get<0>(),
                                             Resource::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[STAGES] = {Resource::L0C::buffer.template Get<0>(),
                                             Resource::L0C::buffer.template Get<1>()};
    static constexpr uint32_t DST[STAGES] = {Resource::UB::gs.template Get<0>(), Resource::UB::gs.template Get<1>()};
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::V_GRAD_MMAD> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t L1A[STAGES] = {Resource::L1::prob.template Get<0>(),
                                             Resource::L1::prob.template Get<1>()};
    static constexpr uint32_t L1B[STAGES] = {Resource::L1::grad.template Get<0>(),
                                             Resource::L1::grad.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {Resource::L0A::buffer.template Get<0>(),
                                             Resource::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {Resource::L0B::buffer.template Get<0>(),
                                             Resource::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C = Resource::L0C::gradVAcc.template Get<0>();
    static constexpr uint32_t DST = Resource::UB::transVGradIn.template Get<0>();
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::K_GRAD_MMAD> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t L1A[STAGES] = {Resource::L1::grab.template Get<0>(),
                                             Resource::L1::grab.template Get<1>()};
    static constexpr uint32_t L1B[STAGES] = {Resource::L1::query.template Get<0>(),
                                             Resource::L1::query.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {Resource::L0A::buffer.template Get<0>(),
                                             Resource::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {Resource::L0B::buffer.template Get<0>(),
                                             Resource::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C = Resource::L0C::gradKAcc.template Get<0>();
    static constexpr uint32_t DST = Resource::UB::transKGradIn.template Get<0>();
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::Q_GRAD_MMAD> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t L1B = Resource::L1::key.template Get<0>();
    static constexpr uint32_t L1A[STAGES] = {Resource::L1::grab.template Get<0>(),
                                             Resource::L1::grab.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {Resource::L0A::buffer.template Get<0>(),
                                             Resource::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {Resource::L0B::buffer.template Get<0>(),
                                             Resource::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[STAGES] = {Resource::L0C::buffer.template Get<0>(),
                                             Resource::L0C::buffer.template Get<1>()};
    static constexpr uint32_t DST = 0;
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::SCORE_GRAD_EPILOGUE> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t AIV_PER_AIC = Resource::AIV_PER_AIC;
    static constexpr uint32_t LOGICAL_M = Resource::L1_TILE_M;
    static constexpr uint32_t AIV_M = Resource::AIV_TILE_M;
    static constexpr uint32_t SCORE[STAGES] = {Resource::UB::score.template Get<0>(),
                                               Resource::UB::score.template Get<1>()};
    static constexpr uint32_t RAB = Resource::UB::rab.template Get<0>();
    static constexpr uint32_t MASK = Resource::UB::mask.template Get<0>();
    static constexpr uint32_t PROB = Resource::UB::prob.template Get<0>();
    static constexpr uint32_t GRABPART[STAGES] = {Resource::UB::grabPart.template Get<0>(),
                                                  Resource::UB::grabPart.template Get<1>()};
    static constexpr uint32_t PROB_DST[STAGES] = {Resource::L1::prob.template Get<0>(),
                                                  Resource::L1::prob.template Get<1>()};
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::RAB_GRAD_EPILOGUE> {
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t AIV_PER_AIC = Resource::AIV_PER_AIC;
    static constexpr uint32_t LOGICAL_M = Resource::L1_TILE_M;
    static constexpr uint32_t AIV_M = Resource::AIV_TILE_M;
    static constexpr uint32_t GS[STAGES] = {Resource::UB::gs.template Get<0>(), Resource::UB::gs.template Get<1>()};
    static constexpr uint32_t GRABPART[STAGES] = {Resource::UB::grabPart.template Get<0>(),
                                                  Resource::UB::grabPart.template Get<1>()};
    static constexpr uint32_t GRAB = Resource::UB::grab.template Get<0>();
    static constexpr uint32_t GRAB_DST[STAGES] = {Resource::L1::grab.template Get<0>(),
                                                  Resource::L1::grab.template Get<1>()};
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::TRANS_KV_GRAD_EPILOGUE> {
    static constexpr uint32_t V_TRANS_IN = Resource::UB::transVGradIn.template Get<0>();
    static constexpr uint32_t K_TRANS_IN = Resource::UB::transKGradIn.template Get<0>();
    static constexpr uint32_t TRANS_OUT = Resource::UB::transKVGradOut.template Get<0>();
};

template <class Resource>
struct TileBuffer<Resource, BufferTag::TRANS_Q_GRAD_EPILOGUE> {
    static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM = Resource::UB::TILE_Q_GRAD_MAX_ELEM;
    static constexpr uint32_t STAGES = Resource::STAGES;
    static constexpr uint32_t TRANS_IN[STAGES] = {Resource::UB::transQGradIn.template Get<0>(),
                                                  Resource::UB::transQGradIn.template Get<1>()};
    static constexpr uint32_t TRANS_OUT[STAGES] = {Resource::UB::transQGradOut.template Get<0>(),
                                                   Resource::UB::transQGradOut.template Get<1>()};
};

}  // namespace Catlass::Kernel::SameAB
