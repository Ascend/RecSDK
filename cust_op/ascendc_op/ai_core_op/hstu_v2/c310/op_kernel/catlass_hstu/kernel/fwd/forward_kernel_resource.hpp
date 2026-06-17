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

#ifndef HSTU_CATLASS_FORWARD_UTILS_HPP
#define HSTU_CATLASS_FORWARD_UTILS_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::utils {

enum class BufferTag : uint8_t {
    QK_MMAD = 0,
    PV_MMAD,
    SCORE_EPILOGUE,
    TRANS_SV_EPILOGUE
};

template <class ArchTag_, class L1TileShape_, class L0TileShape_, class Type_, class AccType_, uint32_t STAGES_,
          bool HAS_RAB_, bool HAS_MASK_>
struct ForwardKernelResource {
public:
    using ArchTag = ArchTag_;
    using Type = Type_;
    using AccType = AccType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    template <uint32_t A, uint32_t B>
    struct Max {
        static constexpr uint32_t value = A > B ? A : B;
    };

    template <bool condition, uint32_t input>
    struct Optional {
        static constexpr uint32_t value = condition ? input : 0;
    };

    static constexpr uint32_t STAGES = STAGES_;
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

    CATLASS_DEVICE
    ForwardKernelResource() = default;

    CATLASS_DEVICE
    ~ForwardKernelResource() = default;

    template <class T, uint32_t elem_, uint32_t upperBound_ = 0, uint32_t bufferCnt_ = 1>
    struct BufferPosition {
        static_assert(bufferCnt_ <= 5 && bufferCnt_ >= 1, "bufferCnt must be range[1, 2]");

        static constexpr uint32_t bytes = elem_ * sizeof(T);
        static constexpr uint32_t bufferCnt = bufferCnt_;
        static constexpr uint32_t upperBound = upperBound_;
        static constexpr uint32_t lowerBound = upperBound + bytes * bufferCnt;

        template <uint32_t bufferIdx>
        static CATLASS_DEVICE constexpr uint32_t Get()
        {
            static_assert(bufferIdx < bufferCnt && bufferIdx >= 0, "bufferIdx invalid");
            return upperBound + bytes * bufferIdx;
        }
    };

    struct L1 {
        static constexpr BufferPosition<Type, L1_MK_ELEM, 0> query;
        static constexpr BufferPosition<Type, L0_NK_ELEM, query.lowerBound, 5> key;
        static constexpr BufferPosition<Type, L0_NK_ELEM, key.lowerBound, 5> value;
        static constexpr BufferPosition<Type, L0_MN_ELEM, value.lowerBound, 5> prob;

        static_assert(prob.lowerBound <= ArchTag::L1_SIZE, "l1 tile shape exceed the arch l1 size!.");
    };

    struct L0A {
        static constexpr uint32_t L0A_PINGPONG_ELEM = ArchTag::L0A_SIZE / 2 / sizeof(Type);
        static constexpr BufferPosition<Type, L0A_PINGPONG_ELEM, 0, 2> buffer;

        static_assert(buffer.lowerBound <= ArchTag::L0A_SIZE, "l0a tile shape exceed the arch l0a size!.");
    };

    struct L0B {
        static constexpr uint32_t L0B_PINGPONG_ELEM = ArchTag::L0B_SIZE / 2 / sizeof(Type);
        static constexpr BufferPosition<Type, L0B_PINGPONG_ELEM, 0, 2> buffer;

        static_assert(buffer.lowerBound <= ArchTag::L0B_SIZE, "l0b tile shape exceed the arch l0b size!.");
    };

    struct L0C {
        static constexpr BufferPosition<AccType, L0_MN_ELEM, 0, 2> buffer;
        static constexpr BufferPosition<AccType, L0_MK_ELEM, buffer.lowerBound> PVAcc;

        static_assert(PVAcc.lowerBound <= ArchTag::L0C_SIZE, "l0c tile shape exceed the arch l0c size!.");
    };

    struct UB {
        static constexpr BufferPosition<Type, L0_MK_ELEM / 2, 0> score;
        static constexpr BufferPosition<Type, L0_MK_ELEM / 2, score.lowerBound, 2> rab;
        static constexpr BufferPosition<AccType, L0_MK_ELEM / 2, rab.lowerBound, 5> prob;
        static constexpr BufferPosition<AccType, L0_MK_ELEM / 2, prob.lowerBound> transIn;

        static_assert(transIn.lowerBound <= ArchTag::UB_SIZE, "ub tile shape exceed the arch ub size!.");
    };
};

template <class buffer, BufferTag bufferTag>
struct TileBuffer {
    static_assert(DEPENDENT_FALSE<buffer>, "TileBuffer specialization failed!.");
};

template <class buffer>
struct TileBuffer<buffer, BufferTag::QK_MMAD> {
    static constexpr uint32_t STAGES = 5;
    static constexpr uint32_t L1Q = buffer::L1::query.template Get<0>();
    static constexpr uint32_t L1K[5] = {buffer::L1::key.template Get<0>(), buffer::L1::key.template Get<1>(),
                                        buffer::L1::key.template Get<2>(), buffer::L1::key.template Get<3>(),
                                        buffer::L1::key.template Get<4>()};
    static constexpr uint32_t L1P = buffer::L1::prob.template Get<0>();
    static constexpr uint32_t L1V[5] = {buffer::L1::value.template Get<0>(), buffer::L1::value.template Get<1>(),
                                        buffer::L1::value.template Get<2>(), buffer::L1::value.template Get<3>(),
                                        buffer::L1::value.template Get<4>()};
    static constexpr uint32_t L0A[2] = {buffer::L0A::buffer.template Get<0>(), buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[2] = {buffer::L0B::buffer.template Get<0>(), buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[2] = {buffer::L0C::buffer.template Get<0>(), buffer::L0C::buffer.template Get<1>()};
    static constexpr uint32_t VACC = buffer::L0C::PVAcc.template Get<0>();
    static constexpr uint32_t DST[5] = {buffer::UB::prob.template Get<0>(), buffer::UB::prob.template Get<1>(),
                                        buffer::UB::prob.template Get<2>(), buffer::UB::prob.template Get<3>(),
                                        buffer::UB::prob.template Get<4>()};
    static constexpr uint32_t TRANS_OUT = buffer::UB::transIn.template Get<0>();
};

template <class buffer>
struct TileBuffer<buffer, BufferTag::PV_MMAD> {
    static constexpr uint32_t STAGES = 5;
    static constexpr uint32_t L1Q = buffer::L1::query.template Get<0>();
    static constexpr uint32_t L1K[5] = {buffer::L1::key.template Get<0>(), buffer::L1::key.template Get<1>(),
                                        buffer::L1::key.template Get<2>(), buffer::L1::key.template Get<3>(),
                                        buffer::L1::key.template Get<4>()};
    static constexpr uint32_t L1P[5] = {buffer::L1::prob.template Get<0>(), buffer::L1::prob.template Get<1>(),
                                        buffer::L1::prob.template Get<2>(), buffer::L1::prob.template Get<3>(),
                                        buffer::L1::prob.template Get<4>()};
    static constexpr uint32_t L1V[5] = {buffer::L1::value.template Get<0>(), buffer::L1::value.template Get<1>(),
                                        buffer::L1::value.template Get<2>(), buffer::L1::value.template Get<3>(),
                                        buffer::L1::value.template Get<4>()};
    static constexpr uint32_t L0A[2] = {buffer::L0A::buffer.template Get<0>(), buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[2] = {buffer::L0B::buffer.template Get<0>(), buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[2] = {buffer::L0C::buffer.template Get<0>(), buffer::L0C::buffer.template Get<1>()};
    static constexpr uint32_t VACC = buffer::L0C::PVAcc.template Get<0>();
    static constexpr uint32_t DST[5] = {buffer::UB::prob.template Get<0>(), buffer::UB::prob.template Get<1>(),
                                        buffer::UB::prob.template Get<2>(), buffer::UB::prob.template Get<3>(),
                                        buffer::UB::prob.template Get<4>()};
    static constexpr uint32_t TRANS_OUT = buffer::UB::transIn.template Get<0>();
};

template <class buffer>
struct TileBuffer<buffer, BufferTag::SCORE_EPILOGUE> {
    static constexpr uint32_t STAGES = 3;
    static constexpr uint32_t SCORE = buffer::UB::score.template Get<0>();
    static constexpr uint32_t RAB[2] = {buffer::UB::rab.template Get<0>(), buffer::UB::rab.template Get<1>()};
    static constexpr uint32_t PROB[5] = {buffer::UB::prob.template Get<0>(), buffer::UB::prob.template Get<1>(),
                                         buffer::UB::prob.template Get<2>(), buffer::UB::prob.template Get<3>(),
                                         buffer::UB::prob.template Get<4>()};
    static constexpr uint32_t DST[5] = {buffer::L1::prob.template Get<0>(), buffer::L1::prob.template Get<1>(),
                                        buffer::L1::prob.template Get<2>(), buffer::L1::prob.template Get<3>(),
                                        buffer::L1::prob.template Get<4>()};
};

template <class buffer>
struct TileBuffer<buffer, BufferTag::TRANS_SV_EPILOGUE> {
    static constexpr uint32_t TRANS_IN = buffer::UB::transIn.template Get<0>();
    static constexpr uint32_t TRANS_OUT = buffer::UB::score.template Get<0>();
};

}  // namespace Catlass::utils

#endif
