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
 * \file hstu_attn_metadata_backward_aicpu.h
 * \brief HstuAttnMetadataBackward（AI CPU 算子）入参契约 + 主体逻辑框架声明。
 *
 * 采用面向过程写法：四个自由函数按固定顺序串成线性管道，
 * 每步函数的产出作为返回值传给下一步，不用类成员或工作区结构体承载中间状态。
 *
 *   Compute(ctx)
 *     ├─ ① PrepareInput      准备数据，完成 tensor 传入的初始化
 *     ├─ ② SplitSections     按缓存大小划分 section
 *     ├─ ③ SplitCoresPersist 遍历 section，为每个 section 做 persist 分核
 *     └─ ④ BuildMetadata     组织 metadata：头信息 + FA/FD 位置关系
 *
 * 本文件是入参契约的唯一事实源：算子类型名、属性名、槽位下标都在这里。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_AICPU_H
#define HSTU_ATTN_METADATA_BACKWARD_AICPU_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {
namespace backward {

// =============================================================================
// 契约常量
// =============================================================================
constexpr const char* kOpType = "HstuAttnMetadataBackward";

constexpr const char* kAttrNumHeads = "num_heads";
constexpr const char* kAttrHeadDim = "head_dim";
constexpr const char* kAttrMaskMode = "mask_mode";  // 取值域契约 MaskMode 见 mask/mask_predictor_factory.h
// causal mask 的 target 组大小（标量，对齐 device 侧 tiling 量 targetGroupSize）；<= 0 = 无 target
constexpr const char* kAttrTargetGroupSize = "target_group_size";

/*! \brief 输入槽位（下标与 host 侧 l0 的 OP_INPUT 顺序严格一致）。 */
enum InputIdx : uint32_t {
    kInCuSeqlensQ = 0U,   // [batch+1] int32/int64，q 的 offsets
    kInCuSeqlensKv = 1U,  // [batch+1] int32/int64，k 的 offsets
    // causal（mask_mode=1）专属段：[batch] int32/int64 可选，未提供 = 空（nullptr 或 0 元素）
    // = history-only / 无 target 封顶；其余 mask 模式该段必须缺席，契约见 mask/causal_mask_predictor.h
    kInNumContexts = 2U,  // 每 batch 的 context 行数（causal mask 的 context 项）
    kInNumTargets = 3U,   // 每 batch 的 target 行数（causal mask 的 target 项）
    // arbitrary（mask_mode=2）专属段：其余 mask 模式传空槽（nullptr）；
    // 逻辑布局 [batch, maxBlkCntK] int32，契约见 mask/arbitrary_mask_predictor.h
    kInFullCnt = 4U,  // 每个 K 行块的 full 类 Q 列块数
    kInMaskCnt = 5U,  // 每个 K 行块的 mask 类 Q 列块数
};

/*! \brief 输出槽位。 */
enum OutputIdx : uint32_t {
    kOutMetadata = 0U,  // [N] int32，HEAD+FA+FD 布局
    kOutputNum = 1U,
};

/*! \brief kernel 返回值。 */
constexpr uint32_t kStatusOk = 0U;
constexpr uint32_t kStatusParamInvalid = 1U;

/*!
 * \brief 块档位真值表：与 hstu_v2 反向的编译期 L1TileShape 一一对应（硬约束，不是可调优参数）。
 *
 * 命名沿用设备侧 kernel 口径（见 MetadataRowBlockScheduler 模板参注释）：HSTU 反向的行轴是
 * seqK，故 BLOCK_M 作用在 K 轴上（Rk）、BLOCK_N 作用在 Q 轴上（Cq）；写 metadata 时二者
 * 分别对应 wire 字段 mBaseSize / s2BaseSize。
 *
 * 真值来自 hstu_v2 的 TileSelector（backward_kernel_builder.hpp:66-81），由同文件的
 * BlockSchedulerBuilder 取 get<1>/get<0>（:370-374）：
 *
 *   TILE_K=128 → L1TileShape<256,128,128> → BLOCK_M=get<1>=128, BLOCK_N=get<0>=256
 *   TILE_K=256 → L1TileShape<128, 64,256> → BLOCK_M=get<1>= 64, BLOCK_N=get<0>=128
 */
constexpr int32_t kHeadDimTileThreshold = 128;  // 128: HSTU TILE_K 档位阈值
constexpr uint32_t kBlockMLarge = 128U;         // TILE_K=128 档：BLOCK_M（行块 Rk）
constexpr uint32_t kBlockNLarge = 256U;         // TILE_K=128 档：BLOCK_N（列块 Cq）
constexpr uint32_t kBlockMSmall = 64U;          // TILE_K=256 档：BLOCK_M（行块 Rk）
constexpr uint32_t kBlockNSmall = 128U;         // TILE_K=256 档：BLOCK_N（列块 Cq）

/*!
 * \brief 由 head_dim 推导 (BLOCK_M, BLOCK_N)。本算子不接收块大小，一律在这里算。
 *
 * 口径逐条对齐 hstu_v2 host（唯一事实源），两步：
 *
 *   ① 选 TILE_K 档位 —— hstu_backward_v2.cpp::TilingKeySet（:170-173，forward 同款 :104-107）
 *          tilingDim = (dimQK > 128 || dimGV > 128) ? 256 : 128
 *      其中 dimQK = Q/K 特征维、dimGV = V/G 特征维（同文件 ParseShape :50-51 取自 q/v 最后一维）。
 *
 *   ② TILE_K → L1TileShape → 网格块大小 —— 即上方真值表。
 *
 * 关于「两个 dim 为什么收敛成一个 head_dim」：
 *   调用方约定 head_dim = max(dimQK, dimGV)（test/hstu_v2/backend/ascend_fuse_backend.py
 *   的 create_forward_metadata / create_backward_metadata，:76 与 :105 都是这么传的），
 *   而 max(dimQK, dimGV) > 128  ⟺  dimQK > 128 || dimGV > 128，
 *   故 ① 的判据等价于本函数用的 headDim > kHeadDimTileThreshold。
 *
 * \note BLOCK_M 是硬约束：设备侧 MetadataRowBlockScheduler::Init 会校验
 *       HEAD[mBaseSize] == 其编译期行块，不一致则该核 blockCnt=0 静默空转、不报错
 *       （metadata_row_block_scheduler.hpp:104-108）。head_dim 传错就是这个后果。
 *       BLOCK_N 是软约束，仅影响 no-FD 路径的负载代价估计。
 *
 * \param headDim 必须等于 max(dimQK, dimGV)，否则与 hstu_v2 编译期块不一致。
 */
inline void DeriveBlockShape(int32_t headDim, uint32_t& blockM, uint32_t& blockN)
{
    // ① 选 TILE_K 档位：超过阈值即退到 TILE_K=256 档，行块减半
    const bool smallTile = (headDim > kHeadDimTileThreshold);
    // ② 由档位查真值表
    blockM = smallTile ? kBlockMSmall : kBlockMLarge;  // Rk，沿 K 轴
    blockN = smallTile ? kBlockNSmall : kBlockNLarge;  // Cq，沿 Q 轴
}

/*!
 * \brief 数据元素字节数，② 估算 L2 驻留量时使用。
 *
 * 入参只有 offsets + num_heads + head_dim，拿不到 dtype，故暂按 FP16(=2) 取值。
 * 后续若补上 dtype 属性（或从伴随张量取），只需改这一处。
 */
constexpr uint32_t kTypeByte = 2U;

// =============================================================================
// 数据结构：各步骤的入参与产物（无聚合工作区，每步产出独立成类型）
// =============================================================================

/*! \brief ②的产物：一段连续的 BN2 区间（section）。 */
struct BackwardSection {
    uint32_t bn2Begin = 0U;  // 含
    uint32_t bn2End = 0U;    // 不含
};

/*!
 * \brief ③的产物：单个 section 内的 persist 分核边界。
 *
 * 语义：第 i 核负责的行块区间是半开的 [起点, 终点)，其中
 *   - 起点 = 上一核的终点；核 0 的起点 = (section.bn2Begin, 0)
 *   - 终点 = (coreBn2End[i], coreKBlkEnd[i])，已规范化为「kBlkEnd == 0 表示切在 BN2 边界」
 *
 * 三个数组长度均等于 usedCoreNum（不是一个 section 固定 36 项 —— 那是 ④ 写 metadata 时的
 * 展平形式，FA 区按 AIC_CORE_NUM 固定 stride，未用到的槽位留空区间由 ④ 负责）。
 *
 * 硬性验收不变量只有「覆盖」这一条：
 *   C1 覆盖   —— 各核区间的并集恰为该 section 的全部行块，一个都不能漏；
 *   C2 不重叠 —— 区间两两不相交（由「本核起点 = 上一核终点」保证）；
 *   C3 有界   —— usedCoreNum <= aicCoreNum。
 *
 * 「每核都有活干」不是要求：行块数少于核数时切出空核（零行块区间）完全正常，
 * 设备侧 MetadataRowBlockScheduler 读到空区间即 blockCnt == 0，本核空转、不报错，
 * 并不需要每个核都参与计算。本实现额外做到了「非空 section 下不产出空核」，
 * 那是配额口径顺带的结果，不是契约 —— ④ 展平成 36 槽时尾部槽位本来就会是空区间。
 */
struct BackwardSectionCores {
    uint32_t usedCoreNum = 0U;
    std::vector<uint32_t> coreBn2End;   // 每核区间终点的 bn2 序号
    std::vector<uint32_t> coreKBlkEnd;  // 同上，终点在该 BN2 内的 K 轴块序号（0 = 切在 BN2 边界）
    std::vector<uint32_t> coreQBlkEnd;  // 该核在 Q 轴上的翻页起点（反向不用 FD，恒 0）
    std::vector<uint64_t> coreWork;     // 每核工作量估计，仅用于自检
};

/*!
 * \brief ①的产物：tensor 句柄 + 属性 + 派生 shape 信息 + 运行配置。
 *
 * 它是后续三个步骤的统一入参上下文：
 *   - tensor 句柄/属性/派生量由 ① PrepareInput 填充；
 *   - 运行配置（l2Byte / aicCoreNum）当前是默认常量，后续可由平台信息覆盖。
 *     它们语义上不属于「输入数据」，但放进同一结构体让 ②③ 的签名保持单参数即可，
 *     避免每步都带一长串配置入参。
 */
struct BackwardInput {
    // 输入张量
    Tensor* cuSeqlensQ = nullptr;
    Tensor* cuSeqlensKv = nullptr;
    // 输出张量
    Tensor* metadata = nullptr;

    // 属性
    int32_t numHeads = 0;
    int32_t headDim = 0;
    int32_t maskMode = 0;  // kMaskNoMask（MaskMode 契约见 mask/mask_predictor_factory.h），① 读入并校验范围

    // 派生量
    int32_t batchSize = 0;
    // 块档位：BLOCK_M 沿 K 轴（行块 Rk）、BLOCK_N 沿 Q 轴（列块 Cq）。
    // 与设备侧 kernel 模板参同名；写 metadata 时对应 wire 字段 mBaseSize / s2BaseSize。
    uint32_t BLOCK_M = 0U;  // 行块 Rk
    uint32_t BLOCK_N = 0U;  // 列块 Cq

    uint32_t bn2Total = 0U;         // batchSize * numHeads
    uint32_t maxBlkCntK = 0U;       // 单个 BN2 的 K 轴块数上界
    std::vector<int64_t> seqLenQ;   // [batchSize] 每个 batch 的 Q 长度
    std::vector<int64_t> seqLenK;   // [batchSize] 每个 batch 的 K 长度
    std::vector<uint32_t> kBlkCnt;  // [batchSize] 每 batch 的 K 轴块数 = ceil(seqLenK / BLOCK_M)
    std::vector<uint32_t> qBlkCnt;  // [batchSize] 每 batch 的 Q 轴块数 = ceil(seqLenQ / BLOCK_N)

    // ---- 运行配置（②③ 消费，非任何步骤的产物）----
    // 缓存预算（②的输入），当前为默认常量，后续可由平台信息覆盖
    uint64_t l2Byte = 96ULL * 1024ULL * 1024ULL;  // 96MB
    // 分核使用的核数（③的输入）
    uint32_t aicCoreNum = 36U;  // 36: 默认 AIC 核数
};

}  // namespace backward
}  // namespace aicpu

// =============================================================================
// Mask 预测器（行块权重查询）：每类 mask 一个头文件，header-only inline
// （CANN AICPU kernel 只编一个 TU，见 CMakeLists.txt）。实现依赖 BackwardInput
// 完整定义，故在上方 namespace 块关闭后包含；头文件自带命名空间 aicpu::backward。
//   - no_mask_predictor.h       无 mask，已实现；
//   - causal_mask_predictor.h   下三角 mask（history + context + target），已实现；
//   - arbitrary_mask_predictor.h 自定义稀疏 mask（cnt 查表），已实现；
//   - mask_predictor_factory.h  MaskMode 取值域契约 + 按 mask_mode 构造预测器的工厂。
// =============================================================================
#include "mask/mask_predictor.h"
#include "mask/no_mask_predictor.h"
#include "mask/causal_mask_predictor.h"
#include "mask/arbitrary_mask_predictor.h"
#include "mask/mask_predictor_factory.h"

namespace aicpu {
namespace backward {

// =============================================================================
// 四个步骤的签名与失败语义（职责概览见文件头）
//
// 失败语义：①②③ 在出错时返回 std::nullopt（错误信息已 KERNEL_LOG_ERROR）；
//          ④ 直接写 tensor 无中间产物，保留 uint32_t 返回码。
// =============================================================================

/*! \brief ① 准备数据：取 tensor、读属性、推导 batchSize 与块大小，返回 BackwardInput。 */
std::optional<BackwardInput> PrepareInput(CpuKernelContext& ctx);

/*!
 * \brief ② 按缓存大小划分 section：沿 BN2 累加驻留字节，超预算即断开。
 *
 * 产出 sections，保证：
 *   - 区间连续且不重叠，并集恰好是 [0, in.bn2Total)；
 *   - 每个 section 至少含 1 个 BN2，故 section 数 <= bn2Total
 *     （这正是 ① 按「最坏情况」校验 metadata 容量的依据）；
 *   - 有多个退化分支（L2 未配置 / 只有 1 个行块 / 单头即可独占每核 L2）直接产出单 section。
 */
std::optional<std::vector<BackwardSection>> SplitSections(const BackwardInput& in);

/*!
 * \brief ③ 遍历 section，为每个 section 做 persist 分核。
 *
 * 每个 section 独立分核（设备侧 MetadataRowBlockScheduler 就是逐 section、逐核读 FA 槽位）：
 *   ① 统计该 section 的总工作量 totalWork = Σ get_head_weight(b)（mask 场景下逐行块不同）；
 *   ② 配额 quota = ceil(totalWork / aicCoreNum)；
 *   ③ 按 batch → head → kBlk 的顺序贪心累加，达到配额且「后面还有块」即切一刀；
 *   ④ 最后一个核无条件收尾到 section 末尾。
 *
 * 正确性只依赖覆盖性（C1~C3 见 BackwardSectionCores）：核 0 起点 = section 起点、
 * 本核终点即下一核起点、末核终点 = section 终点，故每个行块恰好被一个核领走。
 * 附加条件「后面还有块」只是让 usedCoreNum 更紧（省掉一个尾部空核），并非正确性要求。
 *
 * mask 参数默认构造 NoMaskPredictor（无 mask 场景），后续可传入 CausalMask / ArbitraryMask
 * 预测器以实现不同 mask 下的精确负载均衡。
 *
 * 产出 sectionCores，与入参 sections 一一对应。
 */
std::optional<std::vector<BackwardSectionCores>> SplitCoresPersist(const BackwardInput& in,
                                                                   const std::vector<BackwardSection>& sections,
                                                                   const MaskPredictor& mask = NoMaskPredictor{});

/*!
 * \brief ④ 组织 metadata：写 HEAD 头信息与 FA/FD 区，按位置关系落盘。
 *
 * 落盘口径（布局见 hstu_attn_metadata_backward.h，设备侧读法见
 * MetadataRowBlockScheduler::Init/LoadSection）：
 *   - 先整块清零。FD 区、以及各 section 未用到的核槽位就靠这一步定稿为 0 ——
 *     设备侧把全 0 记录展平成 [0, 0) ⇒ 长度 0 ⇒ 本核空转，是期望行为而非错误。
 *   - HEAD：sectionNum / isFD(=0) / mBaseSize(=BLOCK_M) / s2BaseSize(=BLOCK_N)。
 *   - FA：逐 section、逐核写本核区间的两端点 (bn2Start,mStart) 与 (bn2End,mEnd)。
 *     核 0 起点 = 本 section 起点 (sec.bn2Begin, 0)，其后每核起点 = 上一核终点，
 *     末核终点 = (sec.bn2End, 0)。起点逐 section 独立起算，不跨 section 累加。
 *     s2 与 firstFdWs 是 FD 专属字段，反向恒 0。
 *
 * 前置条件：sections 与 sectionCores 一一对应（③ 的产出），且 in.metadata 缓冲
 * 至少 RequiredMetadataElements(sections.size()) 个 int32（① 已按最坏情况校验过）。
 * 不满足时返回 kStatusParamInvalid，绝不会越界写。
 *
 * 本步骤直接落盘到 in.metadata 指向的整块内存，不产出新数据。
 */
uint32_t BuildMetadata(const BackwardInput& in, const std::vector<BackwardSection>& sections,
                       const std::vector<BackwardSectionCores>& sectionCores);

}  // namespace backward

/*!
 * \brief HSTU 反向 metadata 生成算子（AI CPU）。
 *
 * 只有一个虚函数 Compute 会被框架调用，其余全部是自由函数。
 */
class HstuAttnMetadataBackwardCpuKernel : public CpuKernel {
public:
    HstuAttnMetadataBackwardCpuKernel() = default;
    ~HstuAttnMetadataBackwardCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext& ctx) override;
};

}  // namespace aicpu

#endif  // HSTU_ATTN_METADATA_BACKWARD_AICPU_H
