# HstuAttnMetadataBackward 设计文档

| 版本 | 日期 | 修订说明 |
| --- | --- | --- |
| V1.0 | 2026-09-17 | 初版：线性管道重构 + Mask 预测器体系 + mask_mode 属性定稿后的全景梳理 |
| V1.1 | 2026-09-21 | mask 参数传递方案定稿：参数随 mask 类型内聚（XxxMaskParams + FromCtx），工厂升级为唯一解析点 `MakeMaskPredictor(ctx, in)`；mask_mode 三层校验钉死（仅放行 no-mask 链路） |
| V1.2 | 2026-09-22 | arbitrary 入参链路解锁：新增 full_cnt / mask_cnt 输入（槽位 2/3，af 专属段），`ArbitraryMaskParams::FromCtx` 落地存在性/dtype/numel 校验，host 与绑定放行 mask_mode=2 |
| V1.3 | 2026-09-22 | `ArbitraryMaskPredictor::get_blk_weight` 落地：按 pos = batchId × maxBlkCntK + kBlkId 查表 fullCnt + maskCnt（O(1)），聚合接口走基类逐块累加默认实现 |
| V1.4 | 2026-09-24 | causal 端到端解锁（吸收 PR 3202）：新增 num_contexts / num_targets 输入（槽位 2/3，causal 专属段，full_cnt/mask_cnt 顺移至 4/5）与 `target_group_size` 属性；`CausalMaskParams::FromCtx` + 权重闭式 `\|(A ∪ B) ∩ T\|`（A=tril 后缀、B=context 前缀、T=target 封顶，推导见 causal_mask_predictor_design.md）落地，host 与绑定放行 mask_mode=1 |

---

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | hstu_attn_metadata_backward（AICPU 侧类名 HstuAttnMetadataBackward） |
| 算子类别 | 元数据构建类（纯 AI CPU 算子，非融合计算类；只产出调度元数据，不做数值计算） |
| 输入 dtype | int32 / int64（两个 offsets 输入可各自选择，kernel 侧按 dtype 分派读取） |
| 输出 dtype | int32（HEAD + FA + FD 布局的 metadata） |
| 目标平台 | A2 / A3（为 hstu_v2 c310 / DAV_3510 反向 kernel 供数） |
| Kernel 入口 | `hstu_attn_metadata_backward`（`hstu_attn_metadata_backward_aicpu.cpp` 中 `CPU_KERNEL_REGISTER` 注册） |

### 1.2 算子功能

HSTU 反向（hstu_v2）的 K 轴行调度需要一个 host 侧元数据：把 BN2（= batch × num_heads）条
序列的 K 行块切成若干 **section**，再在 section 内做 **persist 分核**，把每个 AI Core 负责的
半开区间写进 metadata，供设备侧 `MetadataRowBlockScheduler` 逐 section 加载任务。

本算子就是这段逻辑的线上实现，内部分四步线性管道：

```text
① PrepareInput      读 offsets/属性 → 校验 → 派生 (BLOCK_M, BLOCK_N)、kBlkCnt/qBlkCnt、bn2Total
② SplitSections     按 L2 驻留预算贪心切 section（驻留超预算时按 BN2 断开，串行换入换出）
③ SplitCoresPersist 每个 section 内按「权重配额」把行块均分给 aicCoreNum 个核（persist）
④ BuildMetadata     把 section 数、块档位、各核区间终点写入 metadata 的 HEAD / FA 槽位
```

- 本算子**不接收块大小**，(BLOCK_M, BLOCK_N) 一律由 head_dim 推导（§2.3）。
- 本算子**不接收 aic_core_num / l2 预算**，暂用结构体内默认值（36 核 / 96MB，见 §4.1 注）。
- 反向不用 FD / stream-K：FA 的 s2 字段、HEAD[isFd]、FD 区整段恒 0，仅为与前向
  `hstu_attn_metadata` 布局同构而占位。

### 1.3 上下游关系

```text
PyTorch 调用方（test/hstu_v2/backend/ascend_fuse_backend.py 等）
  └─ torch.ops.mxrec.hstu_attn_metadata_backward   （cust_op/framework/torch_plugin/...）
       └─ EXEC_NPU_CMD → dlopen libcust_opapi.so
            └─ aclnnHstuAttnMetadataBackwardGetWorkspaceSize + aclnnHstuAttnMetadataBackward
                 └─ op_host/op_api/l0_hstu_attn_metadata_backward.cpp（AICPU launcher）
                      └─ op_kernel_aicpu/ AICPU kernel（本文档主体）
                           └─ 写出 metadata
                                └─ hstu_v2 反向 kernel 设备侧
                                   MetadataRowBlockScheduler::flash_meta_layout 消费
```

> torch 绑定层 README（`cust_op/framework/torch_plugin/torch_library/hstu_attn_metadata/README_backward.md`）
> 记录了调用方视角的接口签名与示例；mask 扩展需要同步改动的点位见该文档 §扩展 与本文 §7。

---

## 2. 契约设计

### 2.1 接口定义

aclnn 两段式接口（`op_host/op_api/aclnn_hstu_attn_metadata_backward.{h,cpp}`）：

```cpp
aclnnStatus aclnnHstuAttnMetadataBackwardGetWorkspaceSize(
    const aclTensor* cuSeqlensQ, const aclTensor* cuSeqlensKv,
    int64_t numHeads, int64_t headDim, int64_t maskMode,
    const aclTensor* fullCnt, const aclTensor* maskCnt,
    const aclTensor* metadata,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnHstuAttnMetadataBackward(void* workspace, uint64_t workspaceSize,
                                          aclOpExecutor* executor, aclrtStream stream);
```

| 参数 | 位置 | 约束 | 校验处 |
| --- | --- | --- | --- |
| cuSeqlensQ | 输入 0 | 1D、长度 ≥ 2、dtype ∈ {int32, int64} | host check + kernel ① |
| cuSeqlensKv | 输入 1 | 同上，且长度须与 cuSeqlensQ 一致 | kernel ①（host 侧形状在 UT 由调用方保证） |
| numHeads | attr | > 0 | host check + kernel ① |
| headDim | attr | ∈ {32, 64, 128, 256}；调用方约定 = max(dimQK, dimGV) | host check + kernel ① |
| maskMode | attr | 取值域 [0, 2]，放行 0（no mask）与 2（arbitrary，随 cnt 生效），1 预留，见 §2.4 | host check + kernel ① |
| fullCnt | 输入 2 | arbitrary 专属段：int32、numel = batch × maxBlkCntK；其余 mask 模式必须空槽 | host check + kernel 工厂 |
| maskCnt | 输入 3 | 同上；与 fullCnt 成对出现/成对缺席 | host check + kernel 工厂 |
| metadata | 输出 0 | 1D、int32、长度 ≥ 最坏情况容量（§2.2） | host check + kernel ① |

kernel 侧返回码：`0` 成功；`1` 参数非法（各步骤 `std::nullopt` 汇总到 `Compute` 统一翻译）。

### 2.2 metadata 内存布局

定义于 `op_kernel_aicpu/hstu_attn_metadata_backward.h`（无 CANN 依赖，host 与 kernel 共用）：

```text
offset 0                     HEAD  [kHeadStride=16]                        int32
offset 16                    FA    [sectionNum][kAicCoreNum=36][kFaStride=16]
offset 16 + FA 区大小         FD    [sectionNum][kAivCoreNum=72][kFdStride=16]
```

- HEAD 字段：`[0]=sectionNum`、`[1]=isFd（恒 0）`、`[2]=mBaseSize（=BLOCK_M）`、`[3]=s2BaseSize（=BLOCK_N）`。
- FA 单核记录（16 个 int32 中只用前 5 个）：`bn2Start / mStart / s2Start(恒0) / bn2End / mEnd`。
  **起点、终点都必须显式写**：核 0 起点 = (sec.bn2Begin, 0)，其余核起点 = 上一核终点；
  设备侧按 `[Flatten(start), Flatten(end))` 展平，漏写 start 会让区间从全局 0 起算。
- 未用核槽位（coreId ≥ usedCoreNum）整条记录全 0 ⇒ 展平后区间长度为 0 ⇒ 设备侧该核空转。
- FD 区整段恒 0（反向无 stream-K）。

容量公式（host 侧申请、kernel 侧自检共用）：

```text
RequiredMetadataElements(n) = 16 + n * (36*16 + 72*16)        // n = sectionNum
最坏情况容量               = RequiredMetadataElements(bn2Total) // 每个 BN2 独占一个 section
```

`torch` 绑定与 examples 均按最坏情况（再 4096 对齐）预分配；① 中按最坏情况自检，
等价于给 ② 上一道「无论怎么切都不会越界写」的硬保证。

### 2.3 块档位（BLOCK_M / BLOCK_N）

真值表（`DeriveBlockShape`，与 hstu_v2 反向编译期 `L1TileShape` 一一对应，**硬约束**）：

| head_dim | TILE_K 档 | BLOCK_M（行块，沿 K 轴 = wire mBaseSize） | BLOCK_N（列块，沿 Q 轴 = wire s2BaseSize） |
| --- | --- | --- | --- |
| ≤ 128（32/64/128） | 128 | 128 | 256 |
| > 128（256） | 256 | 64 | 128 |

- BLOCK_M 是**硬约束**：设备侧 `MetadataRowBlockScheduler::Init` 校验
  `HEAD[mBaseSize] == 编译期行块`，不一致则该核 blockCnt=0 **静默空转、不报错**。
- BLOCK_N 是**软约束**：仅用于 ③ 的行块权重估计（`qBlkCnt = ceil(seqLenQ / BLOCK_N)`）。
- head_dim 判据等价于 hstu_v2 host 的 `(dimQK > 128 || dimGV > 128)`（调用方传 max 值，
  `max(a,b) > 128 ⟺ a > 128 || b > 128`）。

### 2.4 mask_mode 取值域

| 值 | 枚举 | 语义 | 状态 |
| --- | --- | --- | --- |
| 0 | kMaskNoMask | 无 mask | **已实现**（NoMaskPredictor） |
| 1 | kMaskCausal | 下三角 causal mask | **已实现**：num_contexts / num_targets / target_group_size 随输入生效（causal 专属段，槽位 2/3），权重闭式 `\|(A ∪ B) ∩ T\|`（A=tril 后缀、B=context 前缀、T=target 封顶，推导见 mask/causal_mask_predictor_design.md） |
| 2 | kMaskArbitrary | 自定义稀疏 mask | **已实现**：full_cnt/mask_cnt 随输入生效（arbitrary 专属段，槽位 4/5），权重 = fullCnt + maskCnt 按行块查表（零权重退化语义见 §5.2） |

分层校验（同一 wire 值，三层各管一段）：

1. **host 侧**（`op_host/hstu_attn_metadata_backward_check.h`）：查范围 `[0, 2]`；
   按槽位分段固定做专属段校验 —— causal 的 num_contexts/num_targets 非 null 时必须
   1D、长度 = batch、int32/int64 且 mask_mode = 1；arbitrary 的 cnt 必须成对提供
   （int32、非空），其余模式必须空槽（numel = batch × maxBlkCntK 的精确值 host 拿不到，
   留给 kernel 裁决）；
2. **kernel ①**（`PrepareInput`）：用 `IsValidMaskMode` 复核范围（绕过 host 直调 AICPU 的路径也有兜底）；
3. **kernel 工厂**（`MakeMaskPredictor(ctx, in)`）：「是否已实现」的唯一裁决点，按值创建预测器；
   causal 经 `CausalMaskParams::FromCtx`、arbitrary 经 `ArbitraryMaskParams::FromCtx` 做
   存在性/dtype/形状/clamp 校验，任一不符返回 `nullptr`，`Compute` 翻译为参数非法
   （域外值在工厂 default 分支二次兜底）。

---

## 3. 架构设计

### 3.1 模块视图

```text
hstu_attn_metadata_backward/
├── DESIGN.md                        本文档
├── CMakeLists.txt / run.sh          构建：AICPU kernel 交叉编译 + aclnn host 编译 + vendor 打包
├── op_host/
│   ├── hstu_attn_metadata_backward_check.h   aclnn 公共参数校验（含 mask_mode 分层校验）
│   └── op_api/                                aclnn 两段式 + l0 AICPU launcher
├── op_kernel_aicpu/
│   ├── hstu_attn_metadata_backward.h          metadata 布局契约（HEAD/FA/FD，无 CANN 依赖）
│   ├── hstu_attn_metadata_backward_aicpu.h    数据结构与函数签名（四步管道、块档位）
│   ├── hstu_attn_metadata_backward_aicpu.cpp  实现：①~④ + Compute/注册
│   ├── hstu_attn_metadata_backward_common.h   offsets 读取等公共辅助
│   ├── hstu_attn_metadata_backward_aicpu.json AICPU opdef（SoC 全量）
│   └── mask/                                  权重预测器体系（header-only，见 §4.5）
│       ├── mask_predictor.h                   抽象基类 MaskPredictor + RowWeightCtx
│       ├── no_mask_predictor.h                NoMaskPredictor（已实现）
│       ├── causal_mask_predictor.h            CausalMaskPredictor + CausalMaskParams（已实现）
│       ├── causal_mask_predictor_design.md    causal 权重闭式推导与 device 对拍记录
│       ├── arbitrary_mask_predictor.h         ArbitraryMaskPredictor + ArbitraryMaskParams（已实现）
│       └── mask_predictor_factory.h           MaskMode 取值域契约 + 预测器工厂
└── examples/test_aclnn_hstu_attn_metadata_backward.cpp   上板冒烟 + 回读校验（no-mask/af/causal 三用例）
```

### 3.2 线性管道（关键设计决策）

四步之间**不共享 workspace**，每步的产出显式作为下一步入参：

```text
std::optional<BackwardInput>                    PrepareInput(ctx)
std::optional<std::vector<BackwardSection>>     SplitSections(in)
std::optional<std::vector<BackwardSectionCores>>SplitCoresPersist(in, sections, mask)
uint32_t                                        BuildMetadata(in, sections, sectionCores)
```

- **错误传递**：前 ③ 步用 `std::optional`（`std::nullopt` = 失败），④ 已前置校验全部失败条件，
  返回 `uint32_t` 恒成功；`Compute` 串行短路，任一步失败即返回参数非法。
- **为什么去掉 BackwardWorkspace**：旧版所有中间产物塞进一个聚合结构体，各步骤隐式读写，
  数据依赖不可见、UT 难以隔离。线性化后每步可独立测试（输入不可变、输出独立、幂等），
  错误沿返回值传播，调用图即数据流图。
- **mask 的注入点**：`Compute` 在 ② 之后、③ 之前经工厂创建预测器（② 的 L2 驻留模型与
  mask 无关，不需要预测器），③ 以 `const MaskPredictor&` 消费；①②④ 均不感知 mask。
- **mask 参数不进 BackwardInput**（V1.1 定稿）：每类 mask 的专属参数（causal 的
  num_context/num_target/target_group_size、arbitrary 的 sparse 前缀和索引）是该类型
  预测器的构造材料，不是全管道共享的几何上下文。约定「XxxMaskParams 纯数据 +
  XxxMaskParams::FromCtx(ctx, in) 解析」与预测器同头文件内聚（header-only），
  由工厂在对应 case 里解析并注入构造 —— 工厂 `MakeMaskPredictor(ctx, in)` 是 mask
  感知的唯一解析点，`RowWeightCtx` 保持纯几何（in/batchId/kBlkId），mask 配置从
  预测器自身取。此约定下新增 mask 类型不改工厂签名、不动 ①②④、不加管道步骤。

### 3.3 运行视图

```text
aclnn...GetWorkspaceSize        aclnn...
  │ 校验 + 创建 executor           │ 两段式第二段：无 workspace（AICPU 自带栈上缓冲）
  ▼                               ▼
l0 AICPU launcher（OP_INPUT/OFFSETS 传 tensor 指针，ATTR 传 numHeads/headDim/maskMode）
  │
  ▼
AICPU kernel Compute(ctx)
  ├─ ① PrepareInput        nullopt → return kStatusParamInvalid
  ├─ ② SplitSections       nullopt → return kStatusParamInvalid（② 不感知 mask）
  ├─ MakeMaskPredictor     nullptr → return kStatusParamInvalid
  ├─ ③ SplitCoresPersist   nullopt → return kStatusParamInvalid
  └─ ④ BuildMetadata       返回 kStatusOk
```

---

## 4. 详细设计

### 4.1 ① PrepareInput：读取、校验、派生

按序完成五件事（任一失败即 `std::nullopt`）：

1. **槽位与形状**：输入 4 个（槽位 2/3 为 arbitrary 专属段，其余 mask 模式传空槽）、输出 1 个；
   offsets 为 1D 且长度 ≥ 2、两者等长；metadata 为 1D int32。
2. **offsets 内容**：按各自 dtype（int32/int64）读取；首元素必须为 0、单调不减（`ReadOffsets`）。
3. **属性**：`numHeads > 0`；`headDim ∈ {32,64,128,256}`；`maskMode` 范围校验（`IsValidMaskMode`）。
4. **派生量**：
   - `DeriveBlockShape(headDim)` → (BLOCK_M, BLOCK_N)；
   - 每 batch：`seqLenQ/seqLenK = offsets[b+1] - offsets[b]`，
     `kBlkCnt = ceil(seqLenK / BLOCK_M)`（行块沿 K 轴）、`qBlkCnt = ceil(seqLenQ / BLOCK_N)`，
     `maxBlkCntK = max(kBlkCnt)`；
   - `bn2Total = batchSize × numHeads`（BN2 = batchIdx × numHeads + headIdx 的展平域）。
5. **容量自检**：`metadataLen ≥ RequiredMetadataElements(bn2Total)`（最坏情况，§2.2）。

> 注：`aicCoreNum`（默认 36）与 `l2Byte`（默认 96MB）目前是 `BackwardInput` 的默认值，
> 未接平台查询；后续接 `PlatformAscendC` 时只改 ① 一处。

### 4.2 ② SplitSections：按 L2 驻留预算切 section

目标：同一 section 内所有核并行消费其 BN2 的数据，section 之间串行换入换出；
切分收益是让 section 内驻留量 ≤ L2 预算。

**三条短路（任一命中即返回单 section `[0, bn2Total)`）**：

| 短路 | 判据 | 含义 |
| --- | --- | --- |
| A | `l2Byte == 0` | 未配置预算，与参照算子 `SectionStreamKImpl::CalcGridInfoSection` 同款 |
| B | `maxBlkCntK ≤ 1` | K 轴只有一行块，切分无 L2 复用收益 |
| C | `maxHeadBytes ≤ l2Byte / aicCoreNum` | 单核份驻留量已小于「每并发消费者份额」，全铺开也装得下 |

**贪心 first-fit**：沿 BN2 顺序（batch → head）累计 `HeadResidentBytes(in, b)`，
当 `residentBytes != 0 && residentBytes + headBytes > l2ByteLimit` 时在**当前 BN2 之前**断开。

- `residentBytes != 0` 保证不产生空 section；单个 BN2 自身超预算时不切（section 至少含 1 个 BN2）。
- `HeadResidentBytes` 估算单个 (batch, head) 的驻留字节（反向驻留集 Q/K/V + grad + output 共 5 份：
  Q/grad/output 按 seqlen_q 计、K/V 按 seqlen_k 计；grad shape (seqlen_q, nhead_k, headdim_v)，
  暂不支持 headdim_qk != headdim_v 与 GQA 故可按 seqlen_q 计。按 `kTypeByte=2` 即 FP16 口径，
  该常量集中在一处，后续接 dtype 属性只改它）。
- 结尾断言 `bn2 == bn2Total`，并收尾最后一段。

### 4.3 ③ SplitCoresPersist：section 内按权重配额 persist 分核

对每个 section 独立执行三阶段（section 之间无续接，起点逐 section 独立推）：

**阶段 1 — 总量**：`totalWork = Σ_{bn2 ∈ section} mask.get_head_weight(in, b)`，
`totalBlocks = Σ kBlkCnt[b]`（行块总数，用于 hasMore 判断）。

**阶段 2 — 贪心切刀**：按 batch → head → kBlk 顺序逐行块推进，块权重
`weight = mask.get_blk_weight({in, b, k})`（裸值，可为 0，§4.5/§5.2）。
配额 `quota = ceil(totalWork / aicCoreNum)`；当 `accumulated ≥ quota` 且
`coreIdx + 1 < aicCoreNum` 且 `hasMore`（后面还有行块）时切刀：
`MakeBlockEndPoint` 把切点规范化为 (bn2, kBlk) 终点（`kBlkEnd == 0` 表示切在 BN2 边界），
记录终点与本核工作量，累计清零。

**阶段 3 — 末核收尾**：最后一核**无条件**收尾到 `(sec.bn2End, 0)`，兜住所有剩余行块。

产物 `BackwardSectionCores`：`usedCoreNum` + 各核 `coreBn2End / coreKBlkEnd /
coreQBlkEnd(恒0) / coreWork`。

### 4.4 ④ BuildMetadata：落盘

1. **写前夹紧**：sections 与 sectionCores 等长；按**实际** sectionNum 复核 metadata 容量
   （① 已按最坏情况自检，这里是按实际值的精确复核）。
2. **整块清零**（HEAD + FA + FD 全段）：FD 区与未用核槽位就此定稿为 0。
3. **写 HEAD**：sectionNum、isFd=0、mBaseSize=BLOCK_M、s2BaseSize=BLOCK_N。
4. **逐 section 写 FA**：核 0 起点 = (sec.bn2Begin, 0)，核 i 起点 = 核 i-1 终点；
   终点取 `coreBn2End[i] / coreKBlkEnd[i]`；s2 起终点恒 0。

### 4.5 Mask 预测器体系（mask/ 目录）

**接口**（`MaskPredictor`，三个虚函数，单一口径）：

| 接口 | 语义 |
| --- | --- |
| `get_blk_weight(ctx)` | 单个 K 轴行块的实际运算量（以 Q 列块数为单位），裸值可为 0 |
| `get_head_weight(in, b)` | 一个 (batch, head) 的总权重 = Σ get_blk_weight；默认逐行块累加，子类可闭式优化 |
| `get_total_weight(in)` | 全输入总权重 = Σ get_head_weight × numHeads；默认逐 batch 累加 |

**单一口径**：三层接口同为裸值（无 max(1,·) 兜底），`get_head_weight` /
`get_total_weight` 的默认实现与 SplitCoresPersist 阶段 2 的逐块累加恒等，
保证 `totalWork == Σ coreWork`（§5.1 C4）。权重为 0 时的退化行为见 §5.2。

**已实现**：

- `NoMaskPredictor` —— 每行块权重 = `qBlkCnt[b]`，`get_head_weight` 闭式 `kBlkCnt[b] × qBlkCnt[b]`；
- `CausalMaskPredictor` —— 每行块权重 = `|(A ∪ B) ∩ T|` O(1) 闭式（A=tril 后缀 [qSeqMin, qBlkCnt)、
  B=context 前缀 [0, bEff)、T=target 封顶 [0, τ)，τ 逐行移植 device GetTargetQBlockEnd；
  推导与对拍见 causal_mask_predictor_design.md），`get_head_weight` 不覆写走基类逐块累加；
- `ArbitraryMaskPredictor` —— 每行块权重 = fullCnt + maskCnt 按行块查表（O(1)），
  `get_head_weight` 不覆写走基类逐块累加。

**工厂**：`MakeMaskPredictor(CpuKernelContext&, const BackwardInput&)`
（mask/mask_predictor_factory.h），switch 分发 + 参数解析；
causal 经 `CausalMaskParams::FromCtx` 校验并 clamp（存在性/dtype/长度 →
numContext ∈ [0, seqLenQ]、numTarget ∈ [0, min(seqLenQ, seqLenK)]），
arbitrary 经 `ArbitraryMaskParams::FromCtx` 校验 cnt（存在性/dtype/numel），
非法值（cast 进入）返回 `nullptr`。

**单 TU 约束**：CANN AICPU 编译只编一个翻译单元，mask/ 目录全部 header-only inline，
禁止 .cpp 落地，否则链接报多重定义/未定义。

---

## 5. 关键不变式与风险

### 5.1 不变式

| 编号 | 不变式 | 保证机制 |
| --- | --- | --- |
| C1 覆盖 | 各核区间并集恰为 section 全部行块 | 核 0 起点 = section 起点；末核无条件收尾到 (bn2End, 0) |
| C2 不重叠 | 核区间两两不相交 | 本核起点 = 上一核终点（切刀只产出终点） |
| C3 有界 | usedCoreNum ≤ aicCoreNum | 切刀守卫 `coreIdx + 1 < aicCoreNum` + quota 算术下界 |
| C4 口径一致 | totalWork == Σ coreWork | `get_head_weight` 默认实现与阶段 2 调用侧同为裸值累加 |
| C5 容量 | 写 metadata 不越界 | ① 按最坏情况（每 BN2 一个 section）自检；④ 按实际复核 |

### 5.2 零权重的退化行为（C3 的算术论证）

权重允许为 0（如 seqLenQ = 0 的 batch 无 Q 轴运算量）。极端情形 `totalWork = 0`
⇒ `quota = 0` ⇒ `coreWork ≥ quota` 恒真 ⇒ 每块后都尝试切刀；此时切刀守卫
`coreIdx + 1 < aicCoreNum` 成为实际约束（quota ≥ 1 时它被 quota 算术下界覆盖），
分核退化为按行块数均切（欠均衡，但 C1~C3 覆盖性不受影响）。
quota ≥ 1 时，「切第 aicCoreNum 刀至少要消费 aicCoreNum × quota ≥ totalWork」
只能在最后一块发生，已被 hasMore 挡下。

### 5.3 BLOCK_M 硬约束（静默风险）

设备侧 `MetadataRowBlockScheduler::Init` 校验 `HEAD[mBaseSize] == 编译期行块`：
不一致**不报错**，该核 blockCnt=0 静默空转 —— 表现为结果缺块而非崩溃。
head_dim 必须传 `max(dimQK, dimGV)`；这是本算子对调用方唯一的隐式约定。

### 5.4 已知限制

- `kTypeByte` 固定按 FP16(=2) 估算 L2 驻留量；入参无 dtype，接 dtype 属性后改一处即可。
- `aicCoreNum` / `l2Byte` 未接平台查询，用结构体默认值（36 / 96MB）。
- mask_mode = 1（causal）与 2（arbitrary）均已端到端落地；causal 的权重闭式有
  前置条件 `seqLenQ <= seqLenK`（deltaQK >= 0，见 causal_mask_predictor.h 文件头
  与设计文档 §6），违反时闭式低估、仅影响核间均衡不影响覆盖性。两者的聚合接口
  均走基类逐块累加默认实现（不做覆写）。
- FD / stream-K 不用：s2 字段与 FD 区恒 0，仅占位保持与前向布局同构。

---

## 6. 验证

### 6.1 host UT（`cust_op/test/hstu_attn_metadata/host_ut_backward/`）

```bash
cd cust_op/test/hstu_attn_metadata/host_ut_backward
bash run_host_ut.sh                        # 编译 + 跑全部套件
bash run_host_ut.sh -v                     # 逐条打印
bash run_host_ut.sh --suite=split_cores    # 指定套件
```

7 个套件 / 102 条用例（单 TU 编译真实 kernel 源码 + `stubs/` 桩头，**无需 CANN / NPU**）：

| 套件 | 例数 | 覆盖点 |
| --- | --- | --- |
| prepare_input | 30 | 槽位/形状/属性校验（含 mask_mode 缺失、越界、预留值拒绝；arbitrary cnt 契约端到端）、派生量、容量自检 |
| split_cores | 21 | C1~C3、quota 边界、空核、hasMore、多 section 独立；NoMask 闭式 vs 逐块差分、工厂分发、Arbitrary 按 pos 查表权重 |
| split_sections | 15 | 三条短路、贪心断点、单 BN2 超预算 |
| build_metadata | 13 | HEAD/FA 字段、起点续接、清零定稿、容量复核 |
| pipeline_contract | 10 | 线性管道契约：输入不可变、输出独立、错误传播、幂等 |
| block_shape | 7 | 档位边界（32/64/128/256）、非法 head_dim |
| coverage | 6 | 端到端覆盖性（每个行块恰好归属一个核） |

> mask 相关断言不单独成套件，按主题分在 `prepare_input`（属性读取/校验/拒绝）
> 与 `split_cores`（预测器权重口径差分 + 工厂分发）两处。

### 6.2 构建与上板冒烟

```bash
cd cust_op/ascendc_op/ai_core_op/hstu_attn_metadata_backward
bash run.sh --stage=build     # 编译 kernel + opapi + 打包 vendor
bash run.sh --stage=install   # 安装到 opp/vendors（ASCEND_CUSTOM_OPP_PATH 指向 vendor 层）
bash run.sh --stage=run       # C++ 冒烟 example（含回读校验）
```

torch 插件侧的编译与调用示例见其 README（§1.3 路径）。

---

## 7. 扩展路线（mask 落地清单）

按依赖顺序，新增一种 mask 需要动的点位（**no-mask / causal / arbitrary 三类均已全量落地**，
以下同时标注三类的完成形态，供后续新 mask（如 sliding window 等）照抄）：

1. **mask/ 预测器**：在对应 `*_mask_predictor.h` 中定义 `XxxMaskParams`（纯数据 +
   `FromCtx(ctx, in)` 解析与跨字段校验），给预测器加 Params 构造注入，
   实现 `get_blk_weight`（裸值，可为 0）；必要时覆写 `get_head_weight` / `get_total_weight`
   做闭式优化；保持 header-only。**参数不进 BackwardInput、不改工厂签名**。
   （causal 已完成：权重闭式 `|(A ∪ B) ∩ T|` O(1)；arbitrary 已完成：按 pos 查表
   fullCnt + maskCnt；两者聚合接口均走基类逐块累加默认实现，不覆写。）
2. **工厂**：`MakeMaskPredictor` 的对应 case 内解析 Params 并构造预测器（分发表即注册点）。
   （causal / arbitrary 均已完成。）
3. **host 校验**：`check.h` 的 `CheckAttr` 放行新值；若带专属输入，同步加配对/dtype 校验
   （causal 已完成：`CheckCausalTensors`；arbitrary 已完成：`CheckCntTensors`）。
4. **算子签名**：若 mask 需要额外输入，按「槽位分段固定」扩 wire（causal：输入槽位 2/3
   num_contexts/num_targets + 属性 target_group_size；arbitrary：输入槽位 4/5
   full_cnt/mask_cnt；非当前 mode 的段传 nullptr 且 host 拒收），需同步改
   opdef json / l0 launcher / `InputIdx` 枚举 / torch 绑定（其 README §扩展列了 3 处）。
   两类均已按此流程接入。
5. **UT**：预测器单测（直接构造 Params，含与逐块参考的差分），
   `test_prepare_input` / check 用例放行新值，端到端场景（causal 已完成：入参链路
   用例 + pytest 端到端（吸收自 PR 3202）；arbitrary 已完成：入参链路用例 +
   `arbitrary_blk_weight_reads_cnt_by_pos` 权重断言）。
