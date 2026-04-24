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
 • @file dispatch_policy.hpp

 • @brief HSTU GEMM 分发策略定义

 • @description 定义 QK、PV、DQ 等矩阵乘法的分发策略，包括流水线阶段数、缓存标志等配置

 */
#pragma once

#include "catlass/gemm/dispatch_policy.hpp"

namespace Catlass::Gemm {

/**
 • @brief HSTU dQ 矩阵乘法分发策略

 • @tparam ArchTag_ 架构标签

 • @tparam PAGED_CACHE_FLAG_ 是否使用分页缓存

 • @tparam ENABLE_UNIT_FLAG_ 是否启用单元标志

 • @description 用于 dQ = dO * K^T 计算的分发策略，流水线阶段数为 2

 */
template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadHSTUDQ : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

/**
 • @brief HSTU QK 矩阵乘法分发策略

 • @tparam ArchTag_ 架构标签

 • @tparam PAGED_CACHE_FLAG_ 是否使用分页缓存

 • @tparam ENABLE_UNIT_FLAG_ 是否启用单元标志

 • @description 用于 Q * K^T 计算的分发策略，流水线阶段数为 2

 */
template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
struct MmadHSTUQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

/**
 • @brief HSTU PV 矩阵乘法分发策略

 • @tparam ArchTag_ 架构标签

 • @tparam PAGED_CACHE_FLAG_ 是否使用分页缓存

 • @tparam ENABLE_UNIT_FLAG_ 是否启用单元标志

 • @tparam BIND_SUB_CORE_ 是否绑定子核

 • @tparam SUB_CORE_ID_ 子核 ID

 • @description 用于 Score * V 计算的分发策略，支持子核绑定

 */
template <class ArchTag_, bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false, bool BIND_SUB_CORE_ = false,
          uint32_t SUB_CORE_ID_ = 0>
struct MmadHSTUPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool BIND_SUB_CORE = BIND_SUB_CORE_;
    static constexpr uint32_t SUB_CORE_ID = SUB_CORE_ID_;
};
}  // namespace Catlass::Gemm
