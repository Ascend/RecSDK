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
 • @file backward_kernel_resource.hpp

 • @brief HSTU Backward 算子 Kernel 资源管理

 • @description 定义算子所需的缓冲区资源管理和分配，包括各阶段所需的 L1/L0/UB 内存

 */

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::utils {

/**
 • @brief 缓冲区标签枚举

 • @description 标识不同计算阶段使用的缓冲区:

 •              - QK_MMAD: Q*K^T 矩阵乘法

 •              - GV_MMAD: Score*V 矩阵乘法

 •              - V_GRAD_MMAD: dV 梯度计算

 •              - K_GRAD_MMAD: dK 梯度计算

 •              - Q_GRAD_MMAD: dQ 梯度计算

 •              - SCORE_GRAD_EPILOGUE: Score 梯度计算 (SiLU 反向)

 •              - RAB_GRAD_EPILOGUE: RAB 梯度计算

 •              - TRANS_KV_GRAD_EPILOGUE: KV 梯度转置

 •              - TRANS_Q_GRAD_EPILOGUE: Q 梯度转置

 */
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

/**
 • @brief Backward Kernel 资源结构体

 • @tparam ArchTag_ 架构标签

 • @tparam L1TileShape_ L1 Tile 形状

 • @tparam L0TileShape_ L0 Tile 形状

 • @tparam Element_ 数据元素类型

 • @tparam ElementAccumulator_ 累加器类型

 • @tparam STAGES_ 流水线阶段数

 • @tparam HAS_RAB_ 是否有相对位置偏置

 • @tparam HAS_MASK_ 是否有掩码

 • @description 管理 Backward 算子所需的所有缓冲区资源，包括 L1、L0、UB 内存的分配和引用

 */
template <class ArchTag_, class L1TileShape_, class L0TileShape_, class Element_, class ElementAccumulator_,
          uint32_t STAGES_, bool HAS_RAB_, bool HAS_MASK_>
struct BackwardKernelResource {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
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
    BackwardKernelResource() = default;

    CATLASS_DEVICE
    ~BackwardKernelResource() = default;

    template <class T, uint32_t elem_, uint32_t upperBound_ = 0, uint32_t bufferCnt_ = 1>
    struct BufferPosition {
        // buffer cnt must be range [1, 2]
        static_assert(bufferCnt_ <= 2 && bufferCnt_ >= 1, "bufferCnt must be range[1, 2]");

        static constexpr uint32_t bytes = elem_ * sizeof(T);
        static constexpr uint32_t bufferCnt = bufferCnt_;
        static constexpr uint32_t upperBound = upperBound_;
        static constexpr uint32_t lowerBound = upperBound + bytes * bufferCnt;

        template <uint32_t bufferIdx>
        static CATLASS_DEVICE constexpr uint32_t Get()
        {
            static_assert(bufferIdx <= (bufferCnt - 1) && bufferIdx >= 0, "bufferIdx invalid");
            return upperBound + bytes * bufferIdx;
        }
    };

    struct L1 {
        static constexpr BufferPosition<Element, L1_NK_ELEM, 0> key;
        static constexpr BufferPosition<Element, L0_MK_ELEM, key.lowerBound, STAGES> query;
        static constexpr BufferPosition<Element, L1_NK_ELEM, query.lowerBound> value;
        static constexpr BufferPosition<Element, L0_MK_ELEM, value.lowerBound, STAGES> grad;
        static constexpr BufferPosition<Element, L0_MN_ELEM, grad.lowerBound, STAGES> prob;
        static constexpr BufferPosition<Element, L0_MN_ELEM, prob.lowerBound, STAGES> grab;

        static_assert(grab.lowerBound <= ArchTag::L1_SIZE, "l1 tile shape exceed the arch l1 size!.");
    };

    struct L0A {
        static constexpr uint32_t L0A_PINGPONG_ELEM = ArchTag::L0A_SIZE / STAGES / sizeof(Element);
        static constexpr BufferPosition<Element, L0A_PINGPONG_ELEM, 0, STAGES> buffer;

        static_assert(buffer.lowerBound <= ArchTag::L0A_SIZE, "l0a tile shape exceed the arch l0a size!.");
    };

    struct L0B {
        static constexpr uint32_t L0B_PINGPONG_ELEM = ArchTag::L0B_SIZE / STAGES / sizeof(Element);
        static constexpr BufferPosition<Element, L0B_PINGPONG_ELEM, 0, STAGES> buffer;

        static_assert(buffer.lowerBound <= ArchTag::L0B_SIZE, "l0b tile shape exceed the arch l0b size!.");
    };

    struct L0C {
        // 除以2是因为会在L0C中常驻GradVAcc和GradKAcc在片内做累加
        static constexpr uint32_t L0C_PINGPONG_ELEM = ArchTag::L0C_SIZE / STAGES / sizeof(ElementAccumulator) / 2;
        static constexpr BufferPosition<ElementAccumulator, L0C_PINGPONG_ELEM, 0, STAGES> buffer;
        static constexpr BufferPosition<ElementAccumulator, L0_NK_ELEM, buffer.lowerBound> gradVAcc;
        static constexpr BufferPosition<ElementAccumulator, L0_NK_ELEM, gradVAcc.lowerBound> gradKAcc;

        static_assert(gradKAcc.lowerBound <= ArchTag::L0C_SIZE, "l0c tile shape exceed the arch l0c size!.");
    };

    struct UB {
        static constexpr BufferPosition<ElementAccumulator, L0_MN_ELEM, 0> score;
        static constexpr BufferPosition<Element, L0_NK_ELEM, score.lowerBound>
            transKVGradOut;  // trans过程和中的rab mask复用
        static constexpr BufferPosition<Element, Optional<HAS_RAB, L0_MN_ELEM>::value, score.lowerBound> rab;
        static constexpr BufferPosition<Element, Optional<HAS_MASK, L0_MN_ELEM>::value, score.lowerBound> mask;
        static constexpr BufferPosition<Element, L0_MN_ELEM, score.lowerBound> prob;
        static constexpr BufferPosition<Element, L0_MN_ELEM, Max<transKVGradOut.lowerBound, prob.lowerBound>::value>
            grab;  // grab和score复用 因为分为两个阶段所以可以复用
        static constexpr BufferPosition<Element, L0_MN_ELEM, grab.lowerBound> grabPart;
        static constexpr BufferPosition<ElementAccumulator, L0_MN_ELEM, grabPart.lowerBound> gs;

        static_assert(gs.lowerBound <= ArchTag::UB_SIZE, "ub tile shape exceed the arch ub size!.");

        static constexpr uint32_t ELEM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
        static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM =
            RoundDown(ArchTag::UB_SIZE / STAGES / (sizeof(ElementAccumulator) + sizeof(Element)), ELEM_PER_C0);
        static constexpr BufferPosition<ElementAccumulator, TILE_Q_GRAD_MAX_ELEM, 0, STAGES> transQGradIn;
        static constexpr BufferPosition<Element, TILE_Q_GRAD_MAX_ELEM, transQGradIn.lowerBound, STAGES> transQGradOut;

        static_assert(transQGradOut.lowerBound <= ArchTag::UB_SIZE, "ub tile shape exceed the arch ub size!.");
    };
};

/**
 • @brief Tile Buffer 模板

 • @tparam buffer 缓冲区类型

 • @tparam bufferTag 缓冲区标签

 • @description 根据不同的 BufferTag 特化，为各计算阶段分配适当的缓冲区大小

 */
template <class buffer, BufferTag bufferTag>
struct TileBuffer {
    static_assert(DEPENDENT_FALSE<buffer>, "TileBuffer specialization failed!.");
};

/**
 • @brief QK MMAD Tile Buffer 特化

 • @description 为 Q*K^T 矩阵乘法分配缓冲区 (L1A, L1B, L0A, L0B, L0C, DST)

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::QK_MMAD> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t L1B = buffer::L1::key.template Get<0>();
    static constexpr uint32_t L1A[STAGES] = {buffer::L1::query.template Get<0>(), buffer::L1::query.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {buffer::L0A::buffer.template Get<0>(),
                                             buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {buffer::L0B::buffer.template Get<0>(),
                                             buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[STAGES] = {buffer::L0C::buffer.template Get<0>(),
                                             buffer::L0C::buffer.template Get<1>()};
    static constexpr uint32_t DST = buffer::UB::score.template Get<0>();
};

/**
 • @brief GV MMAD Tile Buffer 特化 - Score * V 矩阵乘法

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::GV_MMAD> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t L1B = buffer::L1::value.template Get<0>();
    static constexpr uint32_t L1A[STAGES] = {buffer::L1::grad.template Get<0>(), buffer::L1::grad.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {buffer::L0A::buffer.template Get<0>(),
                                             buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {buffer::L0B::buffer.template Get<0>(),
                                             buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[STAGES] = {buffer::L0C::buffer.template Get<0>(),
                                             buffer::L0C::buffer.template Get<1>()};
    static constexpr uint32_t DST = buffer::UB::gs.template Get<0>();
};

/**
 • @brief V_GRAD MMAD Tile Buffer 特化 - dV 梯度计算

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::V_GRAD_MMAD> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t L1A[STAGES] = {buffer::L1::prob.template Get<0>(), buffer::L1::prob.template Get<1>()};
    static constexpr uint32_t L1B[STAGES] = {buffer::L1::grad.template Get<0>(), buffer::L1::grad.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {buffer::L0A::buffer.template Get<0>(),
                                             buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {buffer::L0B::buffer.template Get<0>(),
                                             buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C = buffer::L0C::gradVAcc.template Get<0>();
    static constexpr uint32_t DST = buffer::UB::transKVGradOut.template Get<0>();
};

/**
 • @brief K_GRAD MMAD Tile Buffer 特化 - dK 梯度计算

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::K_GRAD_MMAD> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t L1A[STAGES] = {buffer::L1::grab.template Get<0>(), buffer::L1::grab.template Get<1>()};
    static constexpr uint32_t L1B[STAGES] = {buffer::L1::query.template Get<0>(), buffer::L1::query.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {buffer::L0A::buffer.template Get<0>(),
                                             buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {buffer::L0B::buffer.template Get<0>(),
                                             buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C = buffer::L0C::gradKAcc.template Get<0>();
    static constexpr uint32_t DST = buffer::UB::transKVGradOut.template Get<0>();
};

/**
 • @brief Q_GRAD MMAD Tile Buffer 特化 - dQ 梯度计算

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::Q_GRAD_MMAD> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t L1B = buffer::L1::key.template Get<0>();
    static constexpr uint32_t L1A[STAGES] = {buffer::L1::grab.template Get<0>(), buffer::L1::grab.template Get<1>()};
    static constexpr uint32_t L0A[STAGES] = {buffer::L0A::buffer.template Get<0>(),
                                             buffer::L0A::buffer.template Get<1>()};
    static constexpr uint32_t L0B[STAGES] = {buffer::L0B::buffer.template Get<0>(),
                                             buffer::L0B::buffer.template Get<1>()};
    static constexpr uint32_t L0C[STAGES] = {buffer::L0C::buffer.template Get<0>(),
                                             buffer::L0C::buffer.template Get<1>()};
    static constexpr uint32_t DST = 0;
};

/**
 • @brief SCORE_GRAD EPILOGUE Tile Buffer 特化 - SiLU 梯度计算

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::SCORE_GRAD_EPILOGUE> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t SCORE = buffer::UB::score.template Get<0>();
    static constexpr uint32_t RAB = buffer::UB::rab.template Get<0>();
    static constexpr uint32_t MASK = buffer::UB::mask.template Get<0>();
    static constexpr uint32_t PROB = buffer::UB::prob.template Get<0>();
    static constexpr uint32_t GRABPART = buffer::UB::grabPart.template Get<0>();
    static constexpr uint32_t PROB_DST[STAGES] = {buffer::L1::prob.template Get<0>(),
                                                  buffer::L1::prob.template Get<1>()};
};

/**
 • @brief RAB_GRAD EPILOGUE Tile Buffer 特化 - RAB 梯度计算

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::RAB_GRAD_EPILOGUE> {
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t GS = buffer::UB::gs.template Get<0>();
    static constexpr uint32_t GRABPART = buffer::UB::grabPart.template Get<0>();
    static constexpr uint32_t GRAB = buffer::UB::grab.template Get<0>();
    static constexpr uint32_t GRAB_DST[STAGES] = {buffer::L1::grab.template Get<0>(),
                                                  buffer::L1::grab.template Get<1>()};
};

/**
 • @brief TRANS_KV_GRAD EPILOGUE Tile Buffer 特化 - KV 梯度转置输出

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::TRANS_KV_GRAD_EPILOGUE> {
    static constexpr uint32_t TRANS_OUT = buffer::UB::transKVGradOut.template Get<0>();
};

/**
 • @brief TRANS_Q_GRAD EPILOGUE Tile Buffer 特化 - Q 梯度转置输出

 */
template <class buffer>
struct TileBuffer<buffer, BufferTag::TRANS_Q_GRAD_EPILOGUE> {
    static constexpr uint32_t TILE_Q_GRAD_MAX_ELEM = buffer::UB::TILE_Q_GRAD_MAX_ELEM;
    static constexpr uint32_t STAGES = buffer::STAGES;
    static constexpr uint32_t TRANS_IN[STAGES] = {buffer::UB::transQGradIn.template Get<0>(),
                                                  buffer::UB::transQGradIn.template Get<1>()};
    static constexpr uint32_t TRANS_OUT[STAGES] = {buffer::UB::transQGradOut.template Get<0>(),
                                                   buffer::UB::transQGradOut.template Get<1>()};
};

}  // namespace Catlass::utils
