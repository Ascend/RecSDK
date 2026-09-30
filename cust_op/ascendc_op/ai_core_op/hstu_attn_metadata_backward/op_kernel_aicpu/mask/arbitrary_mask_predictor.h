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
 * \file mask/arbitrary_mask_predictor.h
 * \brief Arbitrary（自定义稀疏）mask 权重策略（mask_mode = 2）。
 *
 * 权重语义（对齐 device 侧 ArbitraryMaskPredictor，见
 * hstu_v2/c310/op_kernel/catlass_hstu/kernel/mask/arbitrary_mask_sparse_info.md）：
 *   离线生成的块稀疏信息把每个 K 行块命中的 Q 列块分成「全可见（full）」与
 *   「部分可见（mask）」两类；对负载均衡而言两者计算代价相同（full 只是跳过
 *   ApplyMask 的元素级步骤），故行块权重 = 两类列块数之和：
 *     weight(b, kBlk) = fullCnt[pos] + maskCnt[pos]
 *     pos = batchId * maxBlkCntK + kBlkId
 *   bwd 外层轴是 K，故行块上界 maxBlkCntK = max_b ceil(seqLenK[b] / BLOCK_M)，
 *   与本算子 ① 推导的同名派生量一致；pos 不含 head 维 —— 同一 batch 内所有 head
 *   共用同一份掩码模式（与 device 侧 sparse_info 生成端口径一致）。
 *
 * cnt 张量契约（arbitrary 专属输入槽位，见 InputIdx 的 kInFullCnt / kInMaskCnt）：
 *   - 逻辑布局 [B, maxBlkCntK]（生成端形如 [B, 1, maxBlk]），int32；
 *   - FromCtx 按 numel == B * maxBlkCntK 精确校验，rank 不敏感；
 *   - 其余 mask 模式这两个槽位传 nullptr（host 侧拒收非空值）。
 *
 * 依赖注入（与「mask 参数解析统一发生在工厂 case」的目录约定对齐）：
 *   ArbitraryMaskParams（纯数据）+ ArbitraryMaskParams::FromCtx(ctx, in)
 *   （存在性 / dtype / numel 校验，失败返回 std::nullopt）在本头文件内定义；
 *   工厂（mask_predictor_factory.h）的 kMaskArbitrary case 负责解析与组装。
 *
 * 权重查表：get_blk_weight 直接按 pos 读 fullCnt + maskCnt（O(1)）；
 * 某行块两类列块数均为 0 ⇒ 权重 0，退化语义（quota == 0 ⇒ 按行块数均切，
 * 覆盖性不受影响）见 mask_predictor.h 文件头。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_ARBITRARY_MASK_PREDICTOR_H
#define HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_ARBITRARY_MASK_PREDICTOR_H

#include <cstdint>
#include <optional>

#include "cpu_context.h"
#include "cpu_tensor.h"
#include "log.h"

#include "mask_predictor.h"

namespace aicpu {
namespace backward {

/*!
 * \brief arbitrary mask 的入参（纯数据）：两个 cnt 张量的数据指针与逻辑元素数。
 *
 * 只绑定指针，不拷贝数据；Tensor 生命周期由 ctx 保证（整个 Compute 期间有效）。
 */
struct ArbitraryMaskParams {
    const int32_t* fullCnt = nullptr;  // [B, maxBlkCntK] 每行块 full 类列块数
    const int32_t* maskCnt = nullptr;  // [B, maxBlkCntK] 每行块 mask 类列块数
    int64_t numel = 0;                 // == B * maxBlkCntK，weight 落地后的边界自证用

    /*!
     * \brief 从 ctx 的 arbitrary 专属槽位解析并校验 cnt 张量；任一不符返回 std::nullopt。
     *
     * 校验项（缺一则拒绝，不静默退化为均一权重）：
     *   - 槽位存在且带数据（kInFullCnt / kInMaskCnt；其余 mask 模式传空槽，本函数
     *     只会在 kMaskArbitrary case 被调用，空槽即「该传的没传」）；
     *   - dtype 为 int32（与 device 侧 sparse_info 口径一致）；
     *   - numel == batchSize * maxBlkCntK（rank 不敏感，逻辑布局 [B, maxBlkCntK]）。
     *     注：maxBlkCntK == 0（K 全空）时期望 numel 为 0；host 要求 cnt 非空，
     *     故该边角组合在全链路上会被 host 拦下（K 全空用 no-mask 即可），
     *     kernel 层按数学真值校验、不做额外防御。
     *
     * 本头文件按目录约定在 InputIdx / BackwardInput 定义之后被包含
     * （hstu_attn_metadata_backward_aicpu.h 底部统一 include），故直接引用
     * kInFullCnt / kInMaskCnt / Tensor 等完整类型。
     */
    static std::optional<ArbitraryMaskParams> FromCtx(CpuKernelContext& ctx, const BackwardInput& in)
    {
        const Tensor* fullCntT = ctx.Input(static_cast<uint32_t>(kInFullCnt));
        const Tensor* maskCntT = ctx.Input(static_cast<uint32_t>(kInMaskCnt));
        if (fullCntT == nullptr || fullCntT->GetData() == nullptr || maskCntT == nullptr ||
            maskCntT->GetData() == nullptr) {
            KERNEL_LOG_ERROR("mask_mode %d (arbitrary) requires full_cnt and mask_cnt inputs, but got null",
                             in.maskMode);
            return std::nullopt;
        }
        if (fullCntT->GetDataType() != DT_INT32 || maskCntT->GetDataType() != DT_INT32) {
            KERNEL_LOG_ERROR("full_cnt / mask_cnt must be int32, but got %d / %d",
                             static_cast<int32_t>(fullCntT->GetDataType()),
                             static_cast<int32_t>(maskCntT->GetDataType()));
            return std::nullopt;
        }
        const int64_t expected = static_cast<int64_t>(in.batchSize) * static_cast<int64_t>(in.maxBlkCntK);
        if (fullCntT->NumElements() != expected || maskCntT->NumElements() != expected) {
            KERNEL_LOG_ERROR("full_cnt / mask_cnt numel must be batchSize * maxBlkCntK = %ld, but got %ld / %ld",
                             static_cast<long>(expected), static_cast<long>(fullCntT->NumElements()),
                             static_cast<long>(maskCntT->NumElements()));
            return std::nullopt;
        }

        ArbitraryMaskParams params{};
        params.fullCnt = static_cast<const int32_t*>(fullCntT->GetData());
        params.maskCnt = static_cast<const int32_t*>(maskCntT->GetData());
        params.numel = expected;
        return params;
    }
};

class ArbitraryMaskPredictor : public MaskPredictor {
public:
    /*! \brief 构造注入：参数由工厂 case 经 ArbitraryMaskParams::FromCtx 解析组装。 */
    explicit ArbitraryMaskPredictor(const ArbitraryMaskParams& params) : params_(params) {}

    /*! \brief 行块权重 = fullCnt[pos] + maskCnt[pos]，pos = batchId * maxBlkCntK + kBlkId。 */
    uint32_t get_blk_weight(const RowWeightCtx& c) const override
    {
        // 边界自证：调用方保证 kBlkId < kBlkCnt[batchId] ≤ maxBlkCntK，且 numel
        // 已由 FromCtx 按 batchSize * maxBlkCntK 精确校验 ⇒ pos 必在界内，无需防御。
        const size_t pos = c.batchId * static_cast<size_t>(c.in.maxBlkCntK) + static_cast<size_t>(c.kBlkId);
        // cnt 语义为计数（非负）；sum 走 int64 再夹到 0，防脏数据负值回绕成巨大 uint32 污染 quota
        const int64_t w = static_cast<int64_t>(params_.fullCnt[pos]) + static_cast<int64_t>(params_.maskCnt[pos]);
        return (w > 0) ? static_cast<uint32_t>(w) : 0U;
    }

    // 聚合接口（get_head_weight / get_total_weight）不覆写：权重无闭式规律，
    // 基类默认实现逐块调用 get_blk_weight 裸值累加，与分核阶段 2 调用侧天然同口径（C4）。

private:
    ArbitraryMaskParams params_{};
};

}  // namespace backward
}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_ARBITRARY_MASK_PREDICTOR_H
