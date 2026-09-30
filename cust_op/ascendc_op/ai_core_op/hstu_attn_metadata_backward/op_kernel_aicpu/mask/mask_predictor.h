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
 * \file mask/mask_predictor.h
 * \brief Mask 预测器抽象基类：评估「一个 K 轴行块上有多少 Q 轴块不被 mask 跳过」的工作量权重。
 *
 * 设计动机（对应 device 侧 gemm/block/row_block_weight_design.md 的 host 镜像）：
 *   - 无 mask：行块权重恒等于该 batch 的 Q 轴块数（内层全部要算）；
 *   - causal mask：行块越靠后，被下三角跳过的 Q 块越多，权重沿 K 轴递减；
 *   - arbitrary mask：权重 = 稀疏信息给出的 maskCnt + fullCnt，逐行块任意。
 *
 * 基类只定义接口，mask 语义在各子类中表达；分核算法（SplitCoresPersist）对 mask 类型无感。
 *
 * 三层接口、同一口径（契约）：
 *   ┌────────────────────┬────────────────────────────────────────────┐
 *   │ get_blk_weight     │ 单个行块不被跳过的 Q 块数，裸值可为 0      │
 *   │ get_head_weight    │ Σ_k get_blk_weight(k)                        │
 *   │ get_total_weight   │ Σ get_head_weight(b) × numHeads              │
 *   └────────────────────┴────────────────────────────────────────────┘
 *   - 三层必须同口径：子类只需实现 get_blk_weight；覆写聚合接口做 O(1) 闭式优化时，
 *     必须与逐块裸值累加恒等（NoMaskPredictor 是参照实现）。否则 Phase1 的 totalWork
 *     与 Phase2/3 的逐块累加不一致 ⇒ quota 错位 ⇒ 刀点错位。
 *   - 权重可为 0（该 batch 无 Q 轴运算量）：totalWork == 0 ⇒ quota == 0，贪心
 *     在每块后切刀（受核数硬上界约束），分核退化为按行块数均切，覆盖性不受影响。
 *
 * 本目录约定（新增 mask 类型时遵守）：
 *   - 每种 mask 一个头文件，类名 XxxMaskPredictor，继承 MaskPredictor；
 *   - 新增 mask 类型须在 mask_predictor_factory.h 的工厂 switch 中注册（分发表即注册点）；
 *   - 全部实现 header-only inline：CANN AICPU kernel 只编一个 TU
 *     （见 CMakeLists.txt「只编一个 TU」注释），实现放 .cpp 会在真实构建链接失败；
 *   - 头文件自带命名空间 aicpu::backward，且须在 BackwardInput 定义之后包含
 *     （内联实现要访问 BackwardInput 成员）；当前由
 *     hstu_attn_metadata_backward_aicpu.h 在 BackwardInput 之后统一包含。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_MASK_PREDICTOR_H
#define HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_MASK_PREDICTOR_H

#include <cstddef>
#include <cstdint>

namespace aicpu {
namespace backward {

// 前向声明：本头只持有 BackwardInput 的引用；内联实现要求其完整定义已可见
// （由包含方保证，见文件头注释的目录约定）。
struct BackwardInput;

/*!
 * \brief 权重查询上下文：一次「行块权重」查询所需的全部信息。
 *
 * 持引用不持所有权；调用方（SplitCoresPersist）在遍历 (batch, kBlk) 时以临时量构造，
 * 与 device 侧 RowWeightCtx 的纯标量语义对齐（batchId / outerBlkId / 轴长 / 块大小）。
 */
struct RowWeightCtx {
    const BackwardInput& in;  // ① PrepareInput 的产物（seqLen/BlkCnt/块大小全在其中）
    size_t batchId = 0U;      // batch 序号，< batchSize
    uint32_t kBlkId = 0U;     // K 轴行块在本 batch 内的序号，< kBlkCnt[batchId]
};

/*!
 * \brief Mask 预测器抽象基类（行块权重策略），接口契约见文件头注释。
 */
class MaskPredictor {
public:
    virtual ~MaskPredictor() = default;

    /*!
     * \brief 单个 K 轴行块的权重（裸值，可为 0）。
     * \param c 行块上下文（batch 序号 + K 行块序号 + ① 的产物）
     * \return 该行块不被跳过的 Q 轴块数
     */
    virtual uint32_t get_blk_weight(const RowWeightCtx& c) const = 0;

    /*!
     * \brief 该 head 的总权重（= Σ_k get_blk_weight(k)，与逐块裸值累加同口径）。
     *
     * 用于 Phase1 快速累加 totalWork：totalWork = Σ get_head_weight(b)。
     * 默认实现逐行块累加 get_blk_weight 裸值，子类可优化为 O(1) 闭式
     * （覆写时必须与逐块裸值累加恒等）。
     *
     * \param in ① 的产物
     * \param batchId batch 序号，< batchSize
     * \return 该 batch 所有 K 行块的权重之和
     */
    virtual uint64_t get_head_weight(const BackwardInput& in, size_t batchId) const
    {
        uint64_t total = 0ULL;
        const uint32_t blkCnt = (batchId < in.kBlkCnt.size()) ? in.kBlkCnt[batchId] : 0U;
        for (uint32_t k = 0U; k < blkCnt; ++k) {
            total += static_cast<uint64_t>(get_blk_weight(RowWeightCtx{in, batchId, k}));
        }
        return total;
    }

    /*!
     * \brief 整个输入的总权重（= Σ get_head_weight(b) × numHeads）。
     *
     * 用于一次性统计全输入总工作量，默认实现逐 batch 累加 get_head_weight。
     *
     * \param in ① 的产物
     * \return 全部 batch 全部 head 的权重总和
     */
    virtual uint64_t get_total_weight(const BackwardInput& in) const
    {
        uint64_t total = 0ULL;
        for (size_t b = 0U; b < static_cast<size_t>(in.batchSize); ++b) {
            total += get_head_weight(in, b);
        }
        return total * static_cast<uint64_t>(in.numHeads);
    }
};

}  // namespace backward
}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_AICPU_MASK_MASK_PREDICTOR_H
