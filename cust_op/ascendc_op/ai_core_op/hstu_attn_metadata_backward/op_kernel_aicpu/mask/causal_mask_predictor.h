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
 * \file mask/causal_mask_predictor.h
 * \brief Causal（下三角）mask 权重策略 —— history + context + target 均已实现。
 *
 * 判据来源（严格对齐 device 侧 CausalMaskPredictor::MaskRegion，见
 * hstu_v2/c310/op_kernel/catlass_hstu/kernel/mask/causal_mask_predictor.hpp 与
 * detail/target_block_range.hpp）：
 *   NeedHistoryMask: belowDig = (kSeqId <= (qSeqId*qBlkSize + deltaQK + qBlkSize - 1) / kBlkSize)
 *   NeedContextMask: (numContext > 0) && (qSeqId < CeilDiv(numContext, qBlkSize)) &&
 *                    (kSeqId < CeilDiv(seqLenK - numTarget, kBlkSize))
 *   IsTargetSkip:    qSeqId >= GetTargetQBlockEnd(kSeqId, ...)   ← target 封顶 τ
 *   IsSkip = IsOutOfRange || (!trilMask && !contextMask) || IsTargetSkip
 *   （IsOutOfRange 在本算子迭代域 [0,qBlkCnt)×[0,kBlkCnt) 内恒假，不移植）
 *   其中 kBlkSize = 行块大小（BLOCK_M，K 轴）、qBlkSize = 列块大小（BLOCK_N，Q 轴）、
 *         qBlkCnt = 本 batch 的 Q 轴块数（= BackwardInput::qBlkCnt[b]，公式中省略 [b]）、
 *         deltaQK = seqLenK - seqLenQ。
 *
 * 【前置条件】seqLenQ <= seqLenK（即 deltaQK >= 0）。
 *   device 侧 NeedHistoryMask 的实现依赖 C++ 常用算术转换（BLOCK_M/BLOCK_N 形参是 uint32_t）：
 *       const int  qBase = qSeqId * BLOCK_M + deltaQK;  // 无符号乘加，再转 int
 *       const int  rcol  = qBase + BLOCK_M - 1;
 *       const int  rblk  = rcol / BLOCK_N;              // int / uint32_t → 退化为无符号除法
 *       bool belowDig    = (kSeqId <= rblk);            // uint32_t <= int → 无符号比较
 *   deltaQK >= 0 时 qBase >= 0、rcol >= BLOCK_M-1 >= 0，以上四步的隐式转换**全部保值**，
 *   device 行为等价于「有符号除法 + 有符号比较」，本文件的闭式即在此前提下与 device 逐块等价
 *   （对拍：seqLen<=520 全量 270920 例 + seqLenK<=1e6 抽样 19200 例，两档块大小，零分歧）。
 *   deltaQK < 0 时 rcol 可为负，无符号除法把它变成巨大正数 → belowDig 恒真 →
 *   device 的可算集合变成「前缀 ∪ 后缀」，本闭式会**低估**（只影响核间均衡，不影响正确性，
 *   因为覆盖性由 device qBlockScheduler 全量驱动）。该场景按设计约定**不考虑**，
 *   见 causal_mask_predictor_design.md §6 决策 1 与 verify_device_semantics.py 的对照复现。
 *
 * 权重闭式（history 项 |A|、context 项 |B|、target 项 T 均已实现；推导见
 * mask/causal_mask_predictor_design.md §5）：
 *   「需要计算的列块集合」 = (A ∪ B) ∩ T：
 *     A = { qSeqId | qSeqId*qBlkSize + deltaQK + qBlkSize - 1 >= kSeqId*kBlkSize } = [qSeqMin, qBlkCnt)
 *     B = { qSeqId | qSeqId < nCtxQ }（仅当 numContext > 0 且 kSeqId < nCtxK，
 *         nCtxK = CeilDiv(seqLenK - numTarget, kBlkSize)）
 *     T = [0, τ)，τ = GetTargetQBlockEnd(kSeqId)（逐行移植 device target_block_range.hpp；
 *         numTarget = 0 或 targetGroupSize <= 0 → τ = qBlkCnt，T 退化为整轴）
 *   权重 = |(A ∪ B) ∩ T|：cap 后迭代轴变为 [0, τ)，「前缀 ∪ 后缀」双形态公式在其上重放：
 *     bEff = min(nCtxQ, qBlkCnt)，bEffC = min(bEff, τ)
 *     |(A ∪ B) ∩ T| = { τ                   （qSeqMin <= bEffC，相接/相交 → 并集铺满 [0, τ)）
 *                     { bEffC + τ - qSeqMin （qSeqMin > bEffC，有缝；A ∩ T 为空时恰退化为 bEffC）
 *   回归不变量：numTarget = 0 时 τ = qBlkCnt、bEffC = bEff，逐行退化为 history+context 版。
 *
 * 输入口径（causal 专属槽位，见 InputIdx 的 kInNumContexts / kInNumTargets 与属性
 * target_group_size）：num_contexts/num_targets 为可选输入（未提供 = nullptr 或空张量 = 全 0，
 * FromCtx 已分别 clamp 到 [0, seqLenQ] / [0, min(seqLenQ, seqLenK)]）；
 * targetGroupSize 为标量属性（<= 0 = 无 target，等价 device GetTargetQBlockEnd 早退）。
 *
 * 依赖注入（与「mask 参数解析统一发生在工厂 case」的目录约定对齐，首个样例见
 * arbitrary_mask_predictor.h）：CausalMaskParams（纯数据）+ CausalMaskParams::FromCtx(ctx, in)
 * （存在性 / dtype / 长度 / clamp 校验，失败返回 std::nullopt）在本头文件内定义；
 * 工厂（mask_predictor_factory.h）的 kMaskCausal case 负责解析与组装。
 *
 * 当前范围：history + context + target 已生效；get_head_weight 不覆写，继承基类逐块累加，
 *           与块级权重天然恒等（聚合接口语义见 mask_predictor.h 文件头）。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_CAUSAL_MASK_PREDICTOR_H
#define HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_CAUSAL_MASK_PREDICTOR_H

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "cpu_context.h"
#include "cpu_tensor.h"
#include "log.h"

#include "hstu_attn_metadata_backward_common.h"

#include "mask_predictor.h"

namespace aicpu {
namespace backward {

/*!
 * \brief causal mask 的入参（纯数据）：per-batch context/target 行数 + target 组大小。
 *
 * numContext / numTarget 由 FromCtx 从可选输入槽位解析并 clamp（拷贝成值语义向量，
 * 与 arbitrary 的指针绑定不同——这两个张量是 per-batch 小向量，拷贝成本可忽略）；
 * targetGroupSize 来自标量属性。
 */
struct CausalMaskParams {
    std::vector<int64_t> numContext;  // [batchSize]，clamp 到 [0, seqLenQ[b]]；未提供 = 全 0
    std::vector<int64_t> numTarget;   // [batchSize]，clamp 到 [0, min(seqLenQ, seqLenK)]；未提供 = 全 0
    int32_t targetGroupSize = 0;      // <= 0 = 无 target（等价 device GetTargetQBlockEnd 早退）

    /*!
     * \brief 从 ctx 的 causal 专属槽位解析 num_contexts / num_targets 并读取 target_group_size。
     *
     * 解析语义（与 host 侧 CheckOptionalPerBatchTensor 分层同口径）：
     *   - 槽位 null 或 0 元素 = 「未提供」→ 全 0 向量（history-only / 无 target 封顶）；
     *   - 非空时必须 1D、长度 == batchSize、int32/int64；
     *   - clamp 对齐 device 语义：numContext ∈ [0, seqLenQ[b]]（<= 0 → 无 context，
     *     等价 NeedContextMask 首条判据）；numTarget ∈ [0, min(seqLenQ, seqLenK)]
     *     （target 是 Q/K 共有后缀，clamp 保证 historyLen = seqLenK - numTarget >= 0）；
     *   - target_group_size 属性必须存在且 >= 0（host CheckAttr 已校验，此处数学真值复核）。
     *
     * 本函数只会在 kMaskCausal case 被调用；其余 mask 模式该段槽位为空（host 已拒收非空值），
     * 若误入本函数也按「未提供」处理，不静默退化——误调由工厂 case 保证不会发生。
     */
    static std::optional<CausalMaskParams> FromCtx(CpuKernelContext& ctx, const BackwardInput& in)
    {
        CausalMaskParams params{};
        const size_t batch = static_cast<size_t>(in.batchSize);
        params.numContext.assign(batch, 0);
        params.numTarget.assign(batch, 0);

        // target_group_size：标量属性，causal 下必在（host 层已校验非负，缺失属 wire 断裂）。
        if (!detail::ReadAttr(ctx, kAttrTargetGroupSize, params.targetGroupSize)) {
            return std::nullopt;
        }
        if (params.targetGroupSize < 0) {
            KERNEL_LOG_ERROR("target_group_size must be non-negative, but got %d",
                             static_cast<int32_t>(params.targetGroupSize));
            return std::nullopt;
        }

        // num_contexts / num_targets：两个可选输入，解析逻辑完全平行。
        Tensor* const inputs[2] = {ctx.Input(kInNumContexts), ctx.Input(kInNumTargets)};
        std::vector<int64_t>* const outs[2] = {&params.numContext, &params.numTarget};
        const char* const names[2] = {"num_contexts", "num_targets"};
        for (size_t i = 0U; i < 2U; ++i) {
            Tensor* tensor = inputs[i];
            if (tensor == nullptr || tensor->GetData() == nullptr) {
                continue;  // 未提供：保持全 0
            }
            auto shape = tensor->GetTensorShape();
            if (shape == nullptr) {
                KERNEL_LOG_ERROR("%s tensor shape is null", names[i]);
                return std::nullopt;
            }
            if (shape->GetDims() != 1U) {
                KERNEL_LOG_ERROR("%s must be a 1D tensor, but got %u dims", names[i],
                                 static_cast<uint32_t>(shape->GetDims()));
                return std::nullopt;
            }
            if (shape->GetDimSize(0) <= 0) {
                continue;  // 0 元素 = 未提供：保持全 0
            }
            std::vector<int64_t> raw{};
            if (!detail::ReadTensorAsInt64(tensor, raw)) {
                KERNEL_LOG_ERROR("read %s failed", names[i]);
                return std::nullopt;
            }
            if (raw.size() != batch) {
                KERNEL_LOG_ERROR("%s length must equal batch size %u, but got %u", names[i],
                                 static_cast<uint32_t>(batch), static_cast<uint32_t>(raw.size()));
                return std::nullopt;
            }
            for (size_t b = 0U; b < batch; ++b) {
                const int64_t clamped = std::max(raw[b], static_cast<int64_t>(0));
                if (i == 0U) {
                    // numContext ∈ [0, seqLenQ]：> seqLenQ → 覆盖全部 Q 块（predictor 内
                    // min(nCtxQ, qBlkCnt) 已天然收住，clamp 只为口径干净）
                    outs[i]->at(b) = std::min(clamped, in.seqLenQ[b]);
                } else {
                    // numTarget ∈ [0, min(seqLenQ, seqLenK)]：保证 historyLen >= 0
                    outs[i]->at(b) = std::min(clamped, std::min(in.seqLenQ[b], in.seqLenK[b]));
                }
            }
        }
        return params;
    }
};

class CausalMaskPredictor : public MaskPredictor {
public:
    /*! \brief 构造注入：参数由工厂 case 经 CausalMaskParams::FromCtx 解析组装。 */
    explicit CausalMaskPredictor(const CausalMaskParams& params) : params_(params) {}

    /*! \brief 行块权重 = |(A ∪ B) ∩ T|（闭式见文件头；A=tril 后缀、B=context 前缀、T=target 封顶）。 */
    uint32_t get_blk_weight(const RowWeightCtx& c) const override
    {
        if (c.batchId >= c.in.qBlkCnt.size() || c.batchId >= c.in.seqLenQ.size() || c.batchId >= c.in.seqLenK.size() ||
            c.batchId >= params_.numContext.size() || c.batchId >= params_.numTarget.size() || c.in.BLOCK_M == 0U ||
            c.in.BLOCK_N == 0U) {
            return 0U;
        }
        const uint32_t qBlkCnt = c.in.qBlkCnt[c.batchId];
        const uint32_t kBlkSize = c.in.BLOCK_M;  // K 轴行块
        const uint32_t qBlkSize = c.in.BLOCK_N;  // Q 轴列块
        // 前置条件 seqLenQ <= seqLenK，故 deltaQK >= 0（见文件头）；此时 device 侧的
        // qBase / rcol 恒非负，四步隐式无符号转换全部保值，闭式与逐块判据严格等价。
        const int64_t deltaQK = c.in.seqLenK[c.batchId] - c.in.seqLenQ[c.batchId];
        // 【目的】闭式 O(1) 替代「逐 Q 块试判据」的 O(qBlkCnt) 循环：本接口在 ③ SplitCoresPersist
        //   中对每个 (batch, K行块) 调用一次，循环版总开销 = batch × kBlkCnt × qBlkCnt 次带除法判据
        //   （即整个块网格），违背 metadata 算子「廉价预处理」定位（权重闭式契约见设计文档 §4）。
        // 【device 根源】判据取自 device NeedHistoryMask（文件头四步隐式转换的核心两步）：
        //     N(qSeqId) = qSeqId*qBlkSize + deltaQK + qBlkSize - 1  // device 变量名 rcol：Q 块的
        //                 「视野右边界」= 块内最后一行（视野最好的一行）可见的最大 K 列号
        //     belowDig  = (kSeqId <= N(qSeqId) / kBlkSize)          // 整数除法（向零取整）
        //  ⟺  N(qSeqId) >= kSeqId*kBlkSize                            ……商为正时 floor(x/d)>=k ⟺ x>=k·d
        //  ⟺  qSeqId*qBlkSize >= kSeqId*kBlkSize - deltaQK - qBlkSize + 1 ≡ t   ……代入 N、移项
        //   t 的语义：把所有已知量打包成的「首行行号及格线」——qSeqId*qBlkSize 即 Q 块首行行号，
        //   首行 >= t ⟺ 块内至少一行能看到 K 块首列。注意 t 是行/列号量纲（非块号），
        //   块号量纲的阈值是下方 qSeqMin = CeilDiv(t, qBlkSize)。
        // 【单式】不区分 kBlkId==0：前置条件 deltaQK >= 0 下 k=0 代入得 t <= -qBlkSize+1 < 0，
        //   与 signed 除法特化（k=0 判据退化为 N/kBlkSize >= 0，向零取整使边界放宽到 -kBlkSize+1）
        //   同走 t<=0 分支、结果恒等；deltaQK < 0 时 device 无符号回绕使 belowDig 恒真，
        //   需整体重设计而非局部特化（见文件头【前置条件】）。
        const int64_t t = static_cast<int64_t>(c.kBlkId) * kBlkSize - deltaQK - static_cast<int64_t>(qBlkSize) + 1;
        // history 后缀 A = [qSeqMin, qBlkCnt)；t <= 0 ⟺ qSeqMin = 0（整行块在 A 内）
        const uint32_t qSeqMin =
            (t <= 0) ? 0U : static_cast<uint32_t>((t + qBlkSize - 1) / qBlkSize);  // t > 0，CeilDiv 安全

        const int64_t numCtx = (c.batchId < params_.numContext.size()) ? params_.numContext[c.batchId] : 0;
        const int64_t numTgt = (c.batchId < params_.numTarget.size()) ? params_.numTarget[c.batchId] : 0;

        // context 前缀 B = [0, bEff)（device NeedContextMask，见文件头）。
        // k 侧判据恢复：kSeqId < nCtxK = CeilDiv(seqLenK - numTarget, kBlkSize)。
        //   numTarget = 0 时 nCtxK = kBlkCnt 恒真（这正是上一版省略它的原因）；numTarget > 0 时
        //   target 段（K 尾部）的 contextMask 恒假，必须恢复，否则 B 会被高估。
        //   注意门控只看 numTarget、与 targetGroupSize 无关（device numBlkK 不引用 tgs）：
        //   tgs = 0 且 numTarget > 0 时 τ 早退但本门控依然生效，故该组合不退化到旧公式。
        //   FromCtx 已保证 numTarget <= seqLenK，故 historyLen >= 0，CeilDiv 安全。
        const int64_t historyLen = c.in.seqLenK[c.batchId] - numTgt;
        const int64_t nCtxK = (historyLen + static_cast<int64_t>(kBlkSize) - 1) / kBlkSize;
        const uint32_t bEff = (numCtx <= 0 || static_cast<int64_t>(c.kBlkId) >= nCtxK)
                                  ? 0U
                                  : std::min(static_cast<uint32_t>((numCtx + qBlkSize - 1) / qBlkSize), qBlkCnt);

        // target 封顶 T = [0, τ)（device IsTargetSkip：qSeqId >= GetTargetQBlockEnd → skip）。
        const uint32_t tau = GetTargetQBlockEnd(c, numTgt, qBlkCnt, kBlkSize, qBlkSize, deltaQK, numCtx);

        // |(A ∪ B) ∩ T|：cap 后迭代轴变为 [0, τ)，「前缀 ∪ 后缀」双形态公式在其上重放
        // （与上一版 history+context 公式的推导逐行平行，仅把 qBlkCnt/bEff 换成 τ/bEffC）。
        // numTarget = 0 时 τ = qBlkCnt、bEffC = bEff → 逐行退化为 history+context 版。
        const uint32_t bEffC = std::min(bEff, tau);
        if (qSeqMin >= tau) {
            // 防御分支（域内不可达）：域内 kBlkId < kBlkCnt ⟹ qSeqMin <= qBlkCnt-1，且可证 τ > qSeqMin
            // （τ 由 targetQEnd/kEnd-qk/seqLenQ clamp 三者驱动，均 >= qSeqMin+1，见对拍脚本 E 段
            // 百万点计数为 0）；保留它以防越域调用时 bEffC + tau - qSeqMin 的 uint32 下溢。
            return bEffC;  // A ∩ T 为空 → 只剩 context 前缀被 cap 的部分
        }
        return (qSeqMin <= bEffC) ? tau : (bEffC + tau - qSeqMin);
    }

    // get_head_weight 不覆写：基类默认逐块累加本接口结果，天然满足恒等契约；
    // O(1) 分段闭式求和（分段点：t<=0 与 qSeqMin 饱和的 k 界）为可选优化，见设计文档 §4 步骤 3。

private:
    /*!
     * \brief target 封顶 τ(kSeqId) —— 逐行移植 device Detail::GetTargetQBlockEnd
     *        （hstu_v2/c310/op_kernel/catlass_hstu/kernel/mask/detail/target_block_range.hpp）。
     *
     * 轴映射：device Q_BLOCK_SIZE(=BLOCK_M_dev，Q 轴) → qBlkSize；device K_BLOCK_SIZE(=BLOCK_N_dev，
     * K 轴) → kBlkSize；device totalQBlocks → qBlkCnt（本算子的迭代域上界，代替 device 的
     * CeilDiv(seqlenQ, Q_BLOCK_SIZE) —— 两者相等，用派生量避免重算）。
     * 分支顺序、三项 max（qElementEnd ← numContext / targetQEnd / 最后 clamp 到 [0, seqLenQ]）、
     * int64 中间量全部照搬，仅改名；device 形参为 uint32 的 == 0 早退，本侧输入非负故 <= 0 等价。
     * 唯一口径差：device 的 targetGroupSize 是 tiling 标量，本侧来自 params_（属性经
     * CausalMaskParams::FromCtx 注入，语义一致）。
     */
    uint32_t GetTargetQBlockEnd(const RowWeightCtx& c, int64_t numTarget, uint32_t qBlkCnt, uint32_t kBlkSize,
                                uint32_t qBlkSize, int64_t deltaQK, int64_t numContext) const
    {
        if (numTarget <= 0 || params_.targetGroupSize <= 0) {
            return qBlkCnt;  // device: if (numTarget == 0 || targetGroupSize == 0) return totalQBlocks;
        }
        const int64_t historyLen = c.in.seqLenK[c.batchId] - numTarget;
        const int64_t kBlockBegin = static_cast<int64_t>(c.kBlkId) * kBlkSize;
        if (kBlockBegin < historyLen) {
            return qBlkCnt;  // 本 K 行块不在 target 段：无封顶
        }
        const int64_t kBlockEnd = (static_cast<int64_t>(c.kBlkId) + 1) * kBlkSize;
        const int64_t qkOffset = deltaQK;  // = seqLenK - seqLenQ
        const int64_t groupSize = params_.targetGroupSize;
        const int64_t targetIndex = (kBlockEnd - historyLen + groupSize - 1) / groupSize;
        int64_t qElementEnd = kBlockEnd - qkOffset;
        if (numContext > qElementEnd) {
            qElementEnd = numContext;
        }
        const int64_t targetQEnd = historyLen - qkOffset + targetIndex * groupSize;
        if (targetQEnd > qElementEnd) {
            qElementEnd = targetQEnd;
        }
        const int64_t seqLenQ = c.in.seqLenQ[c.batchId];
        if (qElementEnd > seqLenQ) {
            qElementEnd = seqLenQ;
        }
        if (qElementEnd <= 0) {
            return 0U;
        }
        return static_cast<uint32_t>((qElementEnd + qBlkSize - 1) / qBlkSize);  // CeilDiv
    }

    CausalMaskParams params_{};
};

}  // namespace backward
}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_CAUSAL_MASK_PREDICTOR_H
