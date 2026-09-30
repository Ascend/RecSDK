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
 * \file hstu_attn_metadata_backward_aicpu.cpp
 * \brief HstuAttnMetadataBackward kernel 主入口与四个步骤的实现。
 */

#include "hstu_attn_metadata_backward_aicpu.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "log.h"
#include "status.h"

#include "hstu_attn_metadata_backward.h"
#include "hstu_attn_metadata_backward_common.h"

namespace aicpu {
namespace backward {

// metadata 内存布局（HEAD + FA + FD）的事实源（hstu_attn_metadata_backward.h）；④ 全部按它落盘。
namespace ob = optiling::backward;

namespace {

/*! \brief offsets 张量的最短合法长度：batch 个区间 + 1 个端点。 */
constexpr uint32_t kMinOffsetLen = 2U;

/*!
 * \brief 读一条 offsets（cu_seqlens）并做「只有看到数据才能判定」的校验：
 *        一维、长度 >= 2、首元素为 0、单调不减。
 *
 * host 侧 check.h 只做形状/dtype 层面的校验，这几条必须在这里补。
 */
bool ReadOffsets(Tensor* tensor, const char* name, std::vector<int64_t>& offsets)
{
    if (tensor == nullptr) {
        KERNEL_LOG_ERROR("%s is null", name);
        return false;
    }
    auto shape = tensor->GetTensorShape();
    if (shape == nullptr || shape->GetDims() != 1U) {
        KERNEL_LOG_ERROR("%s must be a 1D tensor", name);
        return false;
    }
    if (!detail::ReadTensorAsInt64(tensor, offsets)) {
        KERNEL_LOG_ERROR("read %s failed", name);
        return false;
    }
    if (offsets.size() < kMinOffsetLen) {
        KERNEL_LOG_ERROR("%s length must be >= %u, but got %u", name, kMinOffsetLen,
                         static_cast<uint32_t>(offsets.size()));
        return false;
    }
    if (offsets.front() != 0) {
        KERNEL_LOG_ERROR("%s[0] must be 0, but got %ld", name, offsets.front());
        return false;
    }
    for (size_t i = 1U; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1U]) {
            KERNEL_LOG_ERROR("%s is not non-decreasing at index %u", name, static_cast<uint32_t>(i));
            return false;
        }
    }
    return true;
}

/*!
 * \brief 单个 BN2 的 L2 驻留字节估计：seqlen_q 侧 Q/grad/output 三份 + seqlen_k 侧 K/V 两份。
 *
 * 同一 batch 内所有 head 的序列长度相同，故只与 batchIdx 有关。
 * 参照算子 SectionStreamKImpl::CalcGridInfoSection 是前向口径（Q/K/V/output 共 4 份：
 * s1Cost = Q+O、s2Cost = K+V）；反向主算子的驻留集是 Q/K/V + grad + output 共 5 份：
 *     s1Cost = seqLenQ * headDim * typeByte * 3   (Q、grad、output)
 *     s2Cost = seqLenK * headDim * typeByte * 2   (K、V)
 * 其中 grad 的 shape 为 (seqlen_q, nhead_k, headdim_v)；当前不支持 headdim_qk != headdim_v
 * 与 GQA，故 grad 与 Q/output 一样按 seqlen_q 计。
 *
 * 用 int64 累乘：大 shape 下 headDim=256、序列长度 1e6 时单项就会逼近 int32 上限。
 * 调用方保证 batchIdx < batchSize。
 */
inline int64_t HeadResidentBytes(const BackwardInput& in, size_t batchIdx)
{
    const int64_t perTokenBytes = static_cast<int64_t>(in.headDim) * static_cast<int64_t>(kTypeByte);
    return in.seqLenQ[batchIdx] * perTokenBytes * 3L + in.seqLenK[batchIdx] * perTokenBytes * 2L;
}

/*!
 * \brief 把「第 bn2 个 BN2 内第 kBlk 个行块」规范化为分核区间终点。
 *
 * kBlk 走到该 BN2 的末尾时进位成 (bn2 + 1, 0)，于是产出的终点里
 * coreKBlkEnd == 0 恒表示「切在 BN2 边界」，④ 写 FA 时不必再做判断。
 *
 * 调用方保证 bn2 < 某个 section 的 bn2End <= bn2Total，故 b < batchSize。
 */
inline void MakeBlockEndPoint(const BackwardInput& in, uint32_t bn2, uint32_t kBlk, uint32_t& outBn2, uint32_t& outKBlk)
{
    // numHeads > 0 由 host CheckAttr 与 ① PrepareInput 双重保证，无需再防御除零。
    const uint32_t heads = static_cast<uint32_t>(in.numHeads);
    const size_t b = static_cast<size_t>(bn2) / heads;
    const uint32_t blkCnt = (b < in.kBlkCnt.size()) ? in.kBlkCnt[b] : 0U;
    if (kBlk >= blkCnt) {
        outBn2 = bn2 + 1U;
        outKBlk = 0U;
    } else {
        outBn2 = bn2;
        outKBlk = kBlk;
    }
}

}  // namespace

std::optional<BackwardInput> PrepareInput(CpuKernelContext& ctx)
{
    BackwardInput in{};

    // ---- ① tensor 句柄：两个 offsets 输入 + 一个 metadata 输出 ----
    in.cuSeqlensQ = ctx.Input(static_cast<uint32_t>(kInCuSeqlensQ));
    in.cuSeqlensKv = ctx.Input(static_cast<uint32_t>(kInCuSeqlensKv));
    in.metadata = ctx.Output(static_cast<uint32_t>(kOutMetadata));
    KERNEL_CHECK_FALSE(
        (in.metadata != nullptr && in.metadata->GetData() != nullptr && in.metadata->GetTensorShape() != nullptr),
        std::nullopt, "metadata output is empty");

    // ---- ② 标量属性 ----
    if (!detail::ReadAttr(ctx, kAttrNumHeads, in.numHeads) || !detail::ReadAttr(ctx, kAttrHeadDim, in.headDim) ||
        !detail::ReadAttr(ctx, kAttrMaskMode, in.maskMode)) {
        return std::nullopt;
    }
    KERNEL_CHECK_FALSE(in.numHeads > 0, std::nullopt, "num_heads must be > 0, but got %d", in.numHeads);
    // mask_mode 只校验取值域（契约合法性）；各 mask 专属段入参（causal 的 num_contexts/
    // num_targets/target_group_size、arbitrary 的 full_cnt/mask_cnt）由 ③ 前的 MakeMaskPredictor
    // 在各自 case 的 FromCtx 中解析校验，host 侧 check.h 分层同口径把关。
    KERNEL_CHECK_FALSE(IsValidMaskMode(in.maskMode), std::nullopt, "mask_mode must be in [%d, %d], but got %d",
                       static_cast<int32_t>(kMaskNoMask), static_cast<int32_t>(kMaskArbitrary), in.maskMode);

    // ---- ③ 行块 / 列块档位：在算子内由 head_dim 推导，口径对齐 hstu_v2 host ----
    // 推导规则与依据集中在 DeriveBlockShape 的注释里（TilingKeySet 判据 + L1TileShape 对应）。
    // BLOCK_M(Rk) 是硬约束：设备侧 MetadataRowBlockScheduler 校验 HEAD[mBaseSize]（wire 字段名）。
    DeriveBlockShape(in.headDim, in.BLOCK_M, in.BLOCK_N);

    // ---- ④ offsets → 每个 batch 的 Q/K 长度与行块数 ----
    std::vector<int64_t> offsetsQ{};
    std::vector<int64_t> offsetsKv{};
    if (!ReadOffsets(in.cuSeqlensQ, "cuSeqlensQ", offsetsQ) || !ReadOffsets(in.cuSeqlensKv, "cuSeqlensKv", offsetsKv)) {
        return std::nullopt;
    }
    KERNEL_CHECK_FALSE(offsetsQ.size() == offsetsKv.size(), std::nullopt,
                       "cuSeqlensQ and cuSeqlensKv length mismatch: %u vs %u", static_cast<uint32_t>(offsetsQ.size()),
                       static_cast<uint32_t>(offsetsKv.size()));

    in.batchSize = static_cast<int32_t>(offsetsQ.size() - 1U);
    KERNEL_CHECK_FALSE(in.batchSize > 0, std::nullopt, "batch size must be > 0");

    const size_t batch = static_cast<size_t>(in.batchSize);
    in.seqLenQ.resize(batch);
    in.seqLenK.resize(batch);
    in.kBlkCnt.resize(batch);
    in.qBlkCnt.resize(batch);

    uint32_t maxBlkCntK = 0U;
    for (size_t b = 0U; b < batch; ++b) {
        in.seqLenQ[b] = offsetsQ[b + 1U] - offsetsQ[b];
        in.seqLenK[b] = offsetsKv[b + 1U] - offsetsKv[b];
        // 行块沿 K 轴（Rk）切，故一个 batch 的 K 轴块数由 K 长度决定，向上取整
        in.kBlkCnt[b] = detail::CeilDiv(static_cast<uint64_t>(in.seqLenK[b]), in.BLOCK_M);
        // 列方向（Q 轴）块数：只供 ③ 估算行块权重，不参与 section 切分
        in.qBlkCnt[b] = detail::CeilDiv(static_cast<uint64_t>(in.seqLenQ[b]), in.BLOCK_N);
        maxBlkCntK = std::max(maxBlkCntK, in.kBlkCnt[b]);
    }
    in.maxBlkCntK = maxBlkCntK;

    // ---- ⑤ 派生量 ----
    // BN2 是全局 (batch, head) 展平下标：BN2 = batchIdx * numHeads + headIdx
    in.bn2Total = static_cast<uint32_t>(batch) * static_cast<uint32_t>(in.numHeads);

    // ---- ⑥ metadata 容量自检：按「每个 BN2 独占一个 section」的最坏情况 ----
    // metadata 是调用方分配的 in-place 输出，kernel 只能读到它的长度；torch 绑定与 examples
    // 均按 ((36+72)*bn2Total + 1)*16（再 4096 对齐）分配，即 RequiredMetadataElements(bn2Total)。
    // 而 ② 的 sectionNum 上界恰好就是 bn2Total（每个 section 至少含 1 个 BN2），
    // 故用最坏情况校验，等价于给 ② 上一道「无论怎么切都不会越界写」的硬保证。
    const uint64_t worstCaseElements = static_cast<uint64_t>(optiling::backward::RequiredMetadataElements(in.bn2Total));
    const int64_t metadataLen = in.metadata->GetTensorShape()->GetDimSize(0);
    KERNEL_CHECK_FALSE(metadataLen > 0 && static_cast<uint64_t>(metadataLen) >= worstCaseElements, std::nullopt,
                       "metadata is too small: %ld < %lu (worst case, bn2Total=%u)", metadataLen,
                       static_cast<unsigned long>(worstCaseElements), in.bn2Total);

    return in;
}

std::optional<std::vector<BackwardSection>> SplitSections(const BackwardInput& in)
{
    std::vector<BackwardSection> sections{};

    KERNEL_CHECK_FALSE(in.bn2Total > 0U, std::nullopt, "bn2Total must be > 0, but got %u", in.bn2Total);
    KERNEL_CHECK_FALSE(in.numHeads > 0, std::nullopt, "numHeads must be > 0, but got %d", in.numHeads);
    KERNEL_CHECK_FALSE(in.aicCoreNum > 0U, std::nullopt, "aicCoreNum must be > 0, but got %u", in.aicCoreNum);

    const auto singleSection = [&in]() -> std::vector<BackwardSection> {
        return std::vector<BackwardSection>{BackwardSection{0U, in.bn2Total}};
    };

    // 短路 A：L2 预算未配置 —— 与参照算子 SectionStreamKImpl::CalcGridInfoSection 同款。
    if (in.l2Byte == 0U) {
        return singleSection();
    }
    const int64_t l2ByteLimit = static_cast<int64_t>(in.l2Byte);

    // 短路 B：K 轴只有一行块（即 maxBlkCntK <= 1），切分带来的 L2 复用收益为 0。
    // 参照算子的判据是 max(GroupSize * QuerySeqSize) <= 其 mBaseSize；本算子 G=1，且行块沿 K 轴
    // （设备侧 seqOffsetM 在 HSTU 反向即 seqOffsetK），故等价判据是「K 轴只有一行块」。
    if (in.maxBlkCntK <= 1U) {
        return singleSection();
    }

    // 短路 C：单个 BN2 的驻留量已小到「全部核同时铺开也装得下共享 L2」，无需按 section 串行换入换出。
    // 判据 share = l2ByteLimit / aicCoreNum 不是把 L2 切给各核，而是把共享容量换算成
    // 「每个并发消费者」的份额：maxHeadBytes <= L2 / N  ⇔  N × maxHeadBytes <= L2。
    int64_t maxHeadBytes = 0L;
    for (size_t b = 0U; b < in.seqLenQ.size(); ++b) {
        maxHeadBytes = std::max(maxHeadBytes, HeadResidentBytes(in, b));
    }
    if (maxHeadBytes <= l2ByteLimit / static_cast<int64_t>(in.aicCoreNum)) {
        return singleSection();
    }

    // ---- 贪心 first-fit：沿 BN2 累计，超预算时在「当前 BN2 之前」断开 ----
    // 断点语义与参照算子逐字对齐：residentBytes != 0 保证不产生空 section；
    // 单个 BN2 自身就超预算时不切（section 至少含 1 个 BN2）。
    uint32_t bn2 = 0U;
    uint32_t sectionBegin = 0U;
    int64_t residentBytes = 0L;
    for (size_t b = 0U; b < in.seqLenQ.size(); ++b) {
        const int64_t headBytes = HeadResidentBytes(in, b);
        for (int32_t h = 0; h < in.numHeads; ++h) {
            if (residentBytes != 0L && residentBytes + headBytes > l2ByteLimit) {
                sections.push_back(BackwardSection{sectionBegin, bn2});
                sectionBegin = bn2;
                residentBytes = 0L;
            }
            residentBytes += headBytes;
            ++bn2;
        }
    }

    KERNEL_CHECK_FALSE(bn2 == in.bn2Total, std::nullopt, "bn2 walked %u but bn2Total is %u", bn2, in.bn2Total);
    sections.push_back(BackwardSection{sectionBegin, in.bn2Total});

    return sections;
}

namespace {

/*!
 * \brief ③ 的单 section 实现：统计总工作量 → 按配额贪心切刀 → 末核无条件收尾。
 *
 * 覆盖性（唯一硬性不变量，C1~C3 见 BackwardSectionCores）由三点实现保证：
 *   a) 核 0 的起点是 section 起点（不做任何预处理，天然成立）；
 *   b) 每核终点即下一核起点（切刀只产出一个终点，起点由上一核推得）；
 *   c) 末核无条件收尾到 (sec.bn2End, 0)，兜住所有剩余行块。
 */
BackwardSectionCores SplitOneSection(const BackwardInput& in, const BackwardSection& sec, const MaskPredictor& mask)
{
    // ---- 阶段 1：该 section 的总工作量与总行块数 ----
    // 权重按 head 粒度查询（mask 场景下同一 batch 内不同行块权重可不同，
    // get_head_weight 内部负责逐行块累加或闭式优化），无 mask 时闭式 kBlkCnt × qBlkCnt。
    // 口径契约：get_head_weight == Σ_k get_blk_weight(k)，与阶段 2 的逐块裸值累加同口径，
    // 保证 totalWork == Σ coreWork（否则 quota 与逐块累加不一致，切刀点错位）。
    uint64_t totalWork = 0ULL;
    uint64_t totalBlocks = 0ULL;
    for (uint32_t bn2 = sec.bn2Begin; bn2 < sec.bn2End; ++bn2) {
        const size_t b = static_cast<size_t>(bn2) / static_cast<size_t>(in.numHeads);
        totalBlocks += in.kBlkCnt[b];
        totalWork += mask.get_head_weight(in, b);
    }

    // 配额向上取整；totalBlocks == 0 时阶段 2 循环体不执行，届时不会发生切刀（quota 可能为 0）。
    const uint64_t quota =
        (totalWork + static_cast<uint64_t>(in.aicCoreNum) - 1ULL) / static_cast<uint64_t>(in.aicCoreNum);

    // ---- 阶段 2：贪心 first-fit，按 batch → head → kBlk 的顺序推进 ----
    BackwardSectionCores cores{};
    uint64_t coreWork = 0ULL;  // 本核已累计的工作量，兼作配额判据（与 totalWork 同口径）
    uint64_t walked = 0ULL;    // 已走过的行块数，与 totalBlocks 配合判断「后面还有块」

    // 切刀：规范化产出本核终点（kBlkEnd == 0 表示切在 BN2 边界）、记录本核工作量、清零重计。
    const auto Cut = [&](uint32_t bn2, uint32_t kBlkNext) {
        uint32_t eBn2 = 0U;
        uint32_t eKBlk = 0U;
        MakeBlockEndPoint(in, bn2, kBlkNext, eBn2, eKBlk);
        cores.coreBn2End.push_back(eBn2);
        cores.coreKBlkEnd.push_back(eKBlk);
        cores.coreWork.push_back(coreWork);
        coreWork = 0ULL;
    };

    for (uint32_t bn2 = sec.bn2Begin; bn2 < sec.bn2End; ++bn2) {
        const size_t b = static_cast<size_t>(bn2) / static_cast<size_t>(in.numHeads);
        const uint32_t blkCnt = in.kBlkCnt[b];
        for (uint32_t k = 0U; k < blkCnt; ++k) {
            // 裸值直接累加：权重 = 该行块要算的 Q 块数，可为 0（该 batch 无 Q 轴运算量）。
            coreWork += mask.get_blk_weight(RowWeightCtx{in, b, k});
            ++walked;

            // 切刀三条件：
            //   coreWork >= quota    —— 本核已吃饱；
            //   刀数 + 1 < coreNum   —— 硬上界，与参照 PersistentSplitCore 的 currentCore < coreNum-1
            //       对齐。quota >= 1 时它算术上冗余：切第 coreNum 刀至少要消费
            //       coreNum * quota >= totalWork，只能在最后一块发生 —— 已被 hasMore 挡下；
            //       但 totalWork == 0 ⇒ quota == 0 时该论证失效（0 >= 0 恒真），靠它把
            //       刀数压在 coreNum-1 以内，分核退化为按行块数均切；
            //   hasMore              —— 纯优化项：在最后一块上切刀只会留下一个零行块的尾部
            //       核槽（覆盖性依然成立），加上这个条件只是让 usedCoreNum 更紧。
            const bool hasMore = (walked < totalBlocks);
            if (coreWork >= quota && cores.coreBn2End.size() + 1U < in.aicCoreNum && hasMore) {
                Cut(bn2, k + 1U);
            }
        }
    }

    // ---- 阶段 3：末核无条件收尾到 section 末尾（kBlk 归零 ⇔ 切在 BN2 边界）----
    // 这一句是覆盖性的兜底：无论前面切了几刀，剩余行块总是有主。
    cores.coreBn2End.push_back(sec.bn2End);
    cores.coreKBlkEnd.push_back(0U);
    cores.coreWork.push_back(coreWork);

    cores.usedCoreNum = static_cast<uint32_t>(cores.coreBn2End.size());
    cores.coreQBlkEnd.assign(cores.usedCoreNum, 0U);  // 反向不用 FD，Q 轴翻页起点恒 0
    return cores;
}

}  // namespace

std::optional<std::vector<BackwardSectionCores>> SplitCoresPersist(const BackwardInput& in,
                                                                   const std::vector<BackwardSection>& sections,
                                                                   const MaskPredictor& mask)
{
    KERNEL_CHECK_FALSE(in.aicCoreNum > 0U, std::nullopt, "aicCoreNum must be > 0, but got %u", in.aicCoreNum);

    // 没有 section 可切（入参 sections 为空）时保持幂等，直接返回空列表。
    if (sections.empty()) {
        return std::vector<BackwardSectionCores>{};
    }

    KERNEL_CHECK_FALSE(in.numHeads > 0, std::nullopt, "numHeads must be > 0, but got %d", in.numHeads);
    KERNEL_CHECK_FALSE(in.bn2Total > 0U, std::nullopt, "bn2Total must be > 0, but got %u", in.bn2Total);

    std::vector<BackwardSectionCores> sectionCores{};
    sectionCores.reserve(sections.size());
    for (size_t si = 0U; si < sections.size(); ++si) {
        const BackwardSection& sec = sections[si];
        KERNEL_CHECK_FALSE(sec.bn2Begin < sec.bn2End && sec.bn2End <= in.bn2Total, std::nullopt,
                           "section[%u] range [%u, %u) is invalid", static_cast<uint32_t>(si), sec.bn2Begin,
                           sec.bn2End);
        sectionCores.push_back(SplitOneSection(in, sec, mask));
    }

    return sectionCores;
}

uint32_t BuildMetadata(const BackwardInput& in, const std::vector<BackwardSection>& sections,
                       const std::vector<BackwardSectionCores>& sectionCores)
{
    // ---- 阶段 0：写之前先把「会让写越界」的量夹住 ----
    KERNEL_CHECK_FALSE(
        (in.metadata != nullptr && in.metadata->GetData() != nullptr && in.metadata->GetTensorShape() != nullptr),
        kStatusParamInvalid, "metadata output is empty");
    KERNEL_CHECK_FALSE(sectionCores.size() == sections.size(), kStatusParamInvalid,
                       "sections/sectionCores are not in sync: %u vs %u", static_cast<uint32_t>(sections.size()),
                       static_cast<uint32_t>(sectionCores.size()));

    const uint32_t sectionNum = static_cast<uint32_t>(sections.size());
    const uint64_t required = ob::RequiredMetadataElements(sectionNum);
    const int64_t metadataLen = in.metadata->GetTensorShape()->GetDimSize(0);
    KERNEL_CHECK_FALSE(metadataLen > 0 && static_cast<uint64_t>(metadataLen) >= required, kStatusParamInvalid,
                       "metadata is too small: %ld < %lu (sectionNum=%u)", metadataLen,
                       static_cast<unsigned long>(required), sectionNum);

    ob::MetadataBuffer buf = ob::MakeMetadataBuffer(in.metadata->GetData(), sectionNum);
    // 整块清零（HEAD + FA + FD 全段）。FD 区、以及各 section 里没用到的核槽位就此定稿为 0：
    // 设备侧读到区间长度为 0 即 blockCnt == 0、本核空转，正是期望行为。
    ob::ClearMetadata(buf);

    // ---- 阶段 1：HEAD —— 段数 + 块档位 ----
    // isFD 恒 0：反向不用 FD / stream-K。ClearMetadata 已置 0，这里显式写是为了让契约可见，
    // 不依赖 Clear 的实现细节。
    ob::SetHead(buf, ob::kHeadSectionNumIdx, sectionNum);
    ob::SetHead(buf, ob::kHeadIsFdIdx, 0U);
    // mBaseSize/s2BaseSize 是 wire 字段名，对应本算子的 BLOCK_M（M 轴行块 Rk）/ BLOCK_N（N 轴列块 Cq）。
    // 其中 mBaseSize 是硬约束：设备侧 Init 会拿它与编译期行块比对，不一致则该核不出任务。
    ob::SetHead(buf, ob::kHeadMBaseSizeIdx, in.BLOCK_M);
    ob::SetHead(buf, ob::kHeadS2BaseSizeIdx, in.BLOCK_N);

    // ---- 阶段 2：FA —— 逐 section、逐核写本核区间的两个端点 ----
    // 参照实现（hstu_attn_metadata 的 GenMetadata）是一张连续网格、start 从上一核终点的位置
    // 顺推；本算子每个 section 独立分核（persist），故每个 section 内部从 sec.bn2Begin 重新起算。
    for (size_t si = 0U; si < sections.size(); ++si) {
        const BackwardSection& sec = sections[si];
        const BackwardSectionCores& cores = sectionCores[si];

        KERNEL_CHECK_FALSE(
            cores.coreBn2End.size() == cores.usedCoreNum && cores.coreKBlkEnd.size() == cores.usedCoreNum,
            kStatusParamInvalid, "section[%u] endpoint arrays and usedCoreNum mismatch: [%u,%u] vs %u",
            static_cast<uint32_t>(si), static_cast<uint32_t>(cores.coreBn2End.size()),
            static_cast<uint32_t>(cores.coreKBlkEnd.size()), cores.usedCoreNum);
        // 上界守卫：FA 区每 section 固定 kAicCoreNum 个槽位，usedCoreNum 超了就会写穿到下一 section。
        KERNEL_CHECK_FALSE(cores.usedCoreNum <= ob::kAicCoreNum, kStatusParamInvalid,
                           "section[%u] usedCoreNum %u exceeds aic core num %u", static_cast<uint32_t>(si),
                           cores.usedCoreNum, ob::kAicCoreNum);
        KERNEL_CHECK_FALSE(sec.bn2Begin < sec.bn2End && sec.bn2End <= in.bn2Total, kStatusParamInvalid,
                           "section[%u] range [%u, %u) is invalid", static_cast<uint32_t>(si), sec.bn2Begin,
                           sec.bn2End);

        const uint32_t secIdx = static_cast<uint32_t>(si);
        uint32_t startBn2 = sec.bn2Begin;
        uint32_t startKBlk = 0U;

        for (uint32_t core = 0U; core < cores.usedCoreNum; ++core) {
            const uint32_t endBn2 = cores.coreBn2End[core];
            const uint32_t endKBlk = cores.coreKBlkEnd[core];

            ob::SetFa(buf, secIdx, core, ob::kFaBn2StartIdx, startBn2);
            ob::SetFa(buf, secIdx, core, ob::kFaMStartIdx, startKBlk);
            ob::SetFa(buf, secIdx, core, ob::kFaBn2EndIdx, endBn2);
            ob::SetFa(buf, secIdx, core, ob::kFaMEndIdx, endKBlk);
            // 反向不用 FD：s2（N 轴翻页位置）与首个 FD workspace 下标都是保留位，恒 0。
            ob::SetFa(buf, secIdx, core, ob::kFaS2StartIdx, 0U);
            ob::SetFa(buf, secIdx, core, ob::kFaS2EndIdx, 0U);
            ob::SetFa(buf, secIdx, core, ob::kFaFirstFdWsIdx, 0U);

            // 本核终点即下一核起点；末核终点 = (sec.bn2End, 0)（③ 的收尾保证）。
            startBn2 = endBn2;
            startKBlk = endKBlk;
        }
        // core >= usedCoreNum 的槽位不动：保持 ClearMetadata 的全 0。
        // 设备侧 LoadSection 对它们展平出 [0, 0) ⇒ 长度 0 ⇒ 本核在本 section 空转。
    }

    // ---- 阶段 3：FD 区 ----
    // 反向不产出 FD 图，整段保持全 0（ClearMetadata 已覆盖），无需逐槽写。
    return kStatusOk;
}

}  // namespace backward

uint32_t HstuAttnMetadataBackwardCpuKernel::Compute(CpuKernelContext& ctx)
{
    // ① 准备数据：tensor/属性读入与校验、块档位推导、metadata 容量自检
    auto input = backward::PrepareInput(ctx);
    KERNEL_CHECK_FALSE(input.has_value(), backward::kStatusParamInvalid,
                       "HstuAttnMetadataBackward: prepare input failed!");

    // ② 根据缓存大小划分 section
    auto sections = backward::SplitSections(*input);
    KERNEL_CHECK_FALSE(sections.has_value(), backward::kStatusParamInvalid,
                       "HstuAttnMetadataBackward: split sections failed!");

    // ③ 遍历 section，为每个 section 做 persist 分核。
    // mask 策略由 ① 读入的 mask_mode 属性决定，经工厂创建对应预测器；
    // 工厂是 mask 感知的唯一解析点：按 mode 解析该类型参数并注入预测器构造
    // （no-mask 无参数；causal 解析 num_contexts/num_targets/target_group_size；
    // arbitrary 解析 full_cnt/mask_cnt）。
    // ① 已用 IsValidMaskMode 校验取值域，工厂内的 cast 是安全的。
    auto mask = backward::MakeMaskPredictor(ctx, *input);
    KERNEL_CHECK_FALSE(mask != nullptr, backward::kStatusParamInvalid,
                       "HstuAttnMetadataBackward: unsupported mask_mode %d", input->maskMode);
    auto sectionCores = backward::SplitCoresPersist(*input, *sections, *mask);
    KERNEL_CHECK_FALSE(sectionCores.has_value(), backward::kStatusParamInvalid,
                       "HstuAttnMetadataBackward: split cores failed!");

    // ④ 落盘 metadata：HEAD + 逐 section/逐核 FA（FD 恒 0，ClearMetadata 已置）
    uint32_t ret = backward::BuildMetadata(*input, *sections, *sectionCores);
    KERNEL_CHECK_FALSE(ret == backward::kStatusOk, ret, "HstuAttnMetadataBackward: build metadata failed!");

    return backward::kStatusOk;
}

namespace {
static const char* kernelType = backward::kOpType;
REGISTER_CPU_KERNEL(kernelType, HstuAttnMetadataBackwardCpuKernel);
}  // namespace

}  // namespace aicpu
