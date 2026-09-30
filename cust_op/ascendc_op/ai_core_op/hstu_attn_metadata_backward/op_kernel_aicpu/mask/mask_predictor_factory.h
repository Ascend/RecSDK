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

/*!
 * \file mask/mask_predictor_factory.h
 * \brief MaskMode 取值域契约 + Mask 预测器工厂（header-only inline，单 TU 约束）。
 *
 * 本文件是 mask_mode 的单一注册点：枚举定义取值域契约（wire 值与 host 共用），
 * 工厂完成「mask_mode → 解析该类型参数 → 构造预测器」的映射；新增 mask 类型时两者一起改
 * （完整依赖顺序见 DESIGN.md §7）。非法值由工厂返回 nullptr 并记录错误日志
 *  —— 拒绝而非静默退化，由调用方（Compute）统一报错。
 *
 * 工厂是 mask 感知的唯一解析点：每类 mask 的参数（causal 的 num_contexts/num_targets/
 * target_group_size、arbitrary 的 full_cnt/mask_cnt）只在对应 case 里解析并注入预测器构造
 * （约定：XxxMaskParams 数据结构 + FromCtx(ctx, in) 解析，与本类预测器同头文件内聚）。
 * 管道其余步骤（①②④）与 RowWeightCtx 均不感知 mask 参数；no-mask 链路不消费任何 ctx 入参。
 *
 * 自包含：直接包含 log.h 取 KERNEL_LOG_ERROR、cpu_context.h 取 CpuKernelContext
 * （CANN aicpu 与 host_ut 桩各有一份同名头），不依赖包含方先引入。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_MASK_PREDICTOR_FACTORY_H
#define HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_MASK_PREDICTOR_FACTORY_H

#include <cstdint>
#include <memory>
#include <new>

#include "cpu_context.h"
#include "log.h"

#include "arbitrary_mask_predictor.h"
#include "causal_mask_predictor.h"
#include "mask_predictor.h"
#include "no_mask_predictor.h"

namespace aicpu {
namespace backward {

// 前向声明：工厂只按引用读取 in.maskMode；完整定义由包含方保证
// （hstu_attn_metadata_backward_aicpu.h 在 BackwardInput 定义之后包含本文件）。
struct BackwardInput;

/*!
 * \brief mask 模式（属性 mask_mode 的取值域，wire 值即契约，host 与 kernel 共用同一套数值）。
 *
 * 取值语义三类，与本目录下的预测器一一对应，均已完整实现：
 * kMaskNoMask / kMaskCausal（history + context + target 闭式）/ kMaskArbitrary（cnt 查表）。
 */
enum MaskMode : int32_t {
    kMaskNoMask = 0,  // 无 mask（已实现）
    kMaskCausal = 1,  // 下三角 mask（history + context + target，随 num_contexts/num_targets/target_group_size 生效）
    kMaskArbitrary = 2,  // 自定义稀疏 mask（随 full_cnt / mask_cnt 输入生效）
};

/*! \brief mask_mode 是否在枚举取值域内（仅判范围，不判是否已实现）。 */
constexpr bool IsValidMaskMode(int32_t mode)
{
    return mode >= static_cast<int32_t>(kMaskNoMask) && mode <= static_cast<int32_t>(kMaskArbitrary);
}

/*!
 * \brief ③ 的策略选择入口：按 mask_mode 解析该类型所需参数并创建 MaskPredictor。
 *
 * 分层校验的最后一道闸（三层口径）：
 *   ① PrepareInput   —— IsValidMaskMode 判取值域 [0, 2]；
 *   host check.h     —— 图编译期判范围，并按 mask_mode 校验各专属段入参
 *                       （causal 的 num_contexts/num_targets/target_group_size、
 *                        arbitrary 的 full_cnt/mask_cnt）；
 *   本工厂           —— 域外值的裁决点：返回 nullptr。
 *
 * 映射规则：kMaskNoMask → NoMaskPredictor（无参数，不读 ctx）；
 * kMaskCausal → CausalMaskPredictor（CausalMaskParams::FromCtx 解析 num_contexts /
 * num_targets 并读取 target_group_size 属性，契约不符返回 nullptr）；
 * kMaskArbitrary → ArbitraryMaskPredictor（ArbitraryMaskParams::FromCtx 解析
 * full_cnt / mask_cnt，缺参或契约不符返回 nullptr）；
 * 经 cast 进入的取值域外枚举值返回 nullptr（兜底直调路径）。
 * 调用方（Compute）对 nullptr 统一翻译为参数非法。
 *
 * 扩展约定：实现新 mask 类型时，在本类预测器同头文件定义
 * XxxMaskParams（纯数据）+ XxxMaskParams::FromCtx(ctx, in)（解析 + 跨字段校验，
 * 失败返回 std::nullopt），并在对应 case 里解析后注入预测器构造 —— 本函数签名不再变化。
 * kMaskArbitrary 与 kMaskCausal 均按该约定落地。
 *
 * \param ctx 算子上下文（读属性 / 输入 tensor 用；no-mask 链路不消费）
 * \param in ① PrepareInput 的产物（只读 maskMode；BatchSize 等派生量供 FromCtx 校验用）
 * \return 对应预测器；非法值返回 nullptr
 */
inline std::unique_ptr<MaskPredictor> MakeMaskPredictor(CpuKernelContext& ctx, const BackwardInput& in)
{
    // ① 已用 IsValidMaskMode 校验过取值域，default 分支是防御性兜底：
    // 绕过 host 直调 AICPU 的路径即便漏掉 ① 的校验也不会走进未定义分支。
    switch (static_cast<MaskMode>(in.maskMode)) {
        case kMaskNoMask:
            // no-mask 无参数：ctx 不消费。后续 mask 类型的参数解析统一发生在各自 case 里，
            // 不新增管道步骤、不改 ①②④ 的签名。
            (void)ctx;
            return std::unique_ptr<MaskPredictor>(new (std::nothrow) NoMaskPredictor{});
        case kMaskCausal: {
            // 参数解析发生在各自 case 里（扩展约定）：FromCtx 已记录具体拒因，此处只翻译为 nullptr。
            const auto params = CausalMaskParams::FromCtx(ctx, in);
            if (!params.has_value()) {
                return nullptr;
            }
            return std::unique_ptr<MaskPredictor>(new (std::nothrow) CausalMaskPredictor{*params});
        }
        case kMaskArbitrary: {
            // 参数解析发生在各自 case 里（扩展约定）：FromCtx 已记录具体拒因，此处只翻译为 nullptr。
            const auto params = ArbitraryMaskParams::FromCtx(ctx, in);
            if (!params.has_value()) {
                return nullptr;
            }
            return std::unique_ptr<MaskPredictor>(new (std::nothrow) ArbitraryMaskPredictor{*params});
        }
        default:
            KERNEL_LOG_ERROR("invalid mask_mode %d (expect [%d, %d])", in.maskMode, static_cast<int32_t>(kMaskNoMask),
                             static_cast<int32_t>(kMaskArbitrary));
            return nullptr;
    }
}

}  // namespace backward
}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_MASK_PREDICTOR_FACTORY_H
