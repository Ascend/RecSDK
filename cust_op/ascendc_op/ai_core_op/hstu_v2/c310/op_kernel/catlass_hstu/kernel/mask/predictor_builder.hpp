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
 * @file predictor_builder.hpp
 * @brief 编译期 Predictor 类型选择: 布尔模板参数 → PredictorType 枚举 → 具体类
 *
 *   两层映射:
 *     1. GetPredictorType<IS_LOCAL,IS_CAUSAL,IS_CONTEXT,IS_TARGET,IS_ARBITRARY>::value → PredictorType
 *     2. PredictorByType<PredictorType>::Type → 具体 Predictor 类
 *
 */

#pragma once

#include "no_mask_predictor.hpp"
#include "causal_mask_predictor.hpp"

namespace Catlass::Kernel::Mask {

// =============================================================================
// PredictorType — 编译期 mask 类型标签
// =============================================================================
enum class PredictorType : uint32_t {
    NOMASK = 0,     // 无 mask
    CAUSAL = 1,     // causal / context / target
    LOCAL = 2,      // local attention 滑动窗口 (预留)
    ARBITRARY = 3,  // arbitrary mask (预留)
};

// =============================================================================
// GetPredictorType — 5 布尔值 → PredictorType 枚举
// =============================================================================
template <bool IS_LOCAL, bool IS_CAUSAL, bool IS_ARBITRARY>
struct GetPredictorType {
    static constexpr PredictorType value = (IS_CAUSAL) ? PredictorType::CAUSAL : PredictorType::NOMASK;
};

// =============================================================================
// PredictorByType — PredictorType 枚举 → 具体 Predictor 类
// =============================================================================
template <PredictorType Type, uint32_t BLOCK_M, uint32_t BLOCK_N>
struct PredictorByType {
    // 未特化: 编译期报错
    static_assert(Type != Type, "Unknown PredictorType — missing specialization");
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N>
struct PredictorByType<PredictorType::NOMASK, BLOCK_M, BLOCK_N> {
    using Predictor = NoMaskPredictor<BLOCK_M, BLOCK_N>;
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N>
struct PredictorByType<PredictorType::CAUSAL, BLOCK_M, BLOCK_N> {
    using Predictor = CausalMaskPredictor<BLOCK_M, BLOCK_N>;
};

// =============================================================================
// PredictorSelector — 一步到位: 布尔 → 具体类 (alias template)
// =============================================================================
template <bool IS_LOCAL, bool IS_CAUSAL, bool IS_ARBITRARY, uint32_t BLOCK_M, uint32_t BLOCK_N>
using PredictorSelector =
    typename PredictorByType<GetPredictorType<IS_LOCAL, IS_CAUSAL, IS_ARBITRARY>::value, BLOCK_M, BLOCK_N>::Predictor;

}  // namespace Catlass::Kernel::Mask
