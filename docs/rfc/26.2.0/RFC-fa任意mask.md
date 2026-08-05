# 需求背景

## 业务背景

在生成式推荐领域，不同厂家的 Attention Mask 灵活多变，主要原因包括：

1. **底层模型不同**：Decoder-only、Encoder-Decoder、混合分层模型等多种架构并存
2. **业务场景差异**：用户行为数据结构不同，业务驱动 Mask 持续改造
3. **训练目标不同**：预训练范式不同，Mask 约束目标随之变化
4. **工程优化需求**：算力约束与推理优化方案决定 Mask 实现细节

## 当前痛点

NPU 上应对灵活 Mask 主要有两种实现策略，均存在明显缺陷：

| 方案 | 做法 | 缺陷 |
|------|------|------|
| 方案1：外部传入 Mask | Python 构造 Mask → 传入 Attention 算子 | 无法利用 Mask 稀疏特性 → 性能差；外部构造掩码耗时 |
| 方案2：定制化开发 | 针对每种 Mask 定制块稀疏 Attention 算子 | 开发周期长（月级），无法应对互联网客户快速迭代 |

```text
方案1:                              方案2:
Python 构造 Mask (耗时)              Mask Type1 ──→ Attention 1 (绑定 Mask1)
         ↓                          Mask Type2 ──→ Attention 2 (绑定 Mask2)
Attention 算子 (无块稀疏特性)         Mask Type3 ──→ ...
```

# 提议方案

引入 **Arbitrary_Func** 作为 Mask 到 Kernel 的中间表达，结合 **Agent Skills** 实现从用户 Mask Python 代码到 Kernel 的自动生成流水线。

## 核心概念

### Arbitrary_Func

Arbitrary_Func 是 **Token 级别的掩码描述张量**，通过分段式描述一个 Query Token 可以看见的 KV Token 区间。区间个数和区间范围由用户自定义。

```text
Arbitrary_Func.shape = [q_token, head, func_num]

示例:
  Arbitrary_Func[0, 0, :] = [0, 1, 3, 4]
  Arbitrary_Func[1, 0, :] = [0, 2, 5, 6]

含义:
  q_token=0 可看到 kv [0,1) 和 [3,4) 两个区间
  q_token=1 可看到 kv [0,2) 和 [5,6) 两个区间
```

### Block_Sparse_Meta

Block_Sparse_Meta 是由 Arbitrary_Func 计算出的**块稀疏信息**。根据 Arbitrary_Func 的定义，将 Attention 矩阵划分为 Block，并标记每个 Block 的掩码状态：

| 标记 | 含义 |
|------|------|
| **FULL_BLOCK (F)** | 整个 Block 都有掩码，可完全跳过计算 |
| **PARTIAL_BLOCK (P)** | 整个 Block 部分有掩码，需逐元素处理 |
| **EMPTY_BLOCK (E)** | 整个 Block 无掩码，正常计算 |

### 整体数据流

```text
User Mask.py (Python)           ← 用户提供
        ↓  Agent Skills (Step 1)
Arbitrary_Func.py (Python)      ← 中间表达
        ↓  Agent Skills (Step 2)
Arbitrary_Func Kernel (AscendC/Triton)  ← SIMT Kernel
        ↓
Block_Sparse_Meta Kernel         ← 块稀疏信息
        ↓
Attention Kernel                 ← 带块稀疏跳过的 Attention
```

## Agent Skills 生成流水线

整个流程分三步，由 Agent Skills 编排。每一步均需通过精度验证，失败则自动回溯重试。

### Step 1: Mask.py → Arbitrary_Func.py

- **输入**：用户给定的 Mask Python 代码
- **输出**：等价的 Arbitrary_Func Python 实现
- **质量保障**：生成 100+ 样例进行精度对比，若对比不过则 Agent 重新生成

### Step 2: Arbitrary_Func.py → AscendC/Triton SIMT Kernel

- **输入**：已验证的 Arbitrary_Func Python 代码
- **输出**：AscendC SIMT Kernel 代码（或 Triton Kernel）
- **依赖 Skills**：
  - [AscendC SIMT 最佳实践](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-simt-best-practices)
  - [AscendC SIMT Tiling 设计](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-simt-tiling-design)
  - [AscendC UT 开发](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-ut-develop)
  - [AscendC ST 设计](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-st-design)

### Step 3: Kernel UT/ST 测试

- **输入**：生成的 SIMT Kernel
- **验证方式**：UT 测试 + ST 测试，精度与 Arbitrary_Func Python 代码对比
- **质量保障**：测试用例 100+，反复生成直到所有精度通过

### 流水线总览

```text
Mask.py ──→ Arbitrary_Func.py ──→ AscendC/Triton Kernel
   │                                 │
   │  Step 1: Agent 翻译             │  Step 3: UT/ST 测试
   │  100+ 样例精度对比               │  100+ 用例精度对比
   │  失败则重试                      │  失败则重试
   │                                 │
   └──────── 在线运行 ────────────────┘
            Arbitrary_Func Kernel
                 ↓
            Block_Sparse_Meta
                 ↓
            Attention Kernel
```

## Skills 编写要点

1. **流程编排**：指导 Agent 一步一步从 Python 代码翻译成 Arbitrary_Func，再翻译成 SIMT Kernel。中间生成失败或精度对比不过需重新生成
2. **Arbitrary_Func 定义**：清晰定义 Arbitrary_Func 包含的要素和规则，使 Agent 生成更加精确
3. **AscendC SIMT 参考**：Agent 需从 CANNBOT-SKILLS 获取 AscendC 语法和 SIMT 语法信息指导生成
4. **用例覆盖**：生成过程中每一步都必须做测试对比，避免精度误差累积

# 方案目标

① **快速适配**：将 Mask 算子开发周期从 **月级降低至天级**，应对互联网客户快速迭代需求

② **性能保障**：通过 Block_Sparse_Meta 保留块稀疏跳过能力，避免外部传 Mask 方案的性能损失

③ **通用覆盖**：Arbitrary_Func 具有通用性，解决一类 Mask 问题，而非"来一个 shape 写一个"

④ **自动化闭环**：Agent Skills 实现 Mask → Kernel 全自动生成 + 精度验证 + 回溯重试，最小化人工介入

# 总体设计

## 离线生成阶段

- 用户在离线环境提供 Mask Python 代码
- Agent Skills 驱动流水线：Mask.py → Arbitrary_Func.py → AscendC/Triton Kernel
- 全流程自动精度验证，通过后输出 Kernel 二进制 / 源码
- Arbitrary_Func 可选择 AscendC 或 Ascend-Triton 编写

## 上线运行阶段

```text
User Input → Arbitrary_Func Kernel → Block_Sparse_Meta Kernel → Attention Kernel
```

- Arbitrary_Func Kernel 在线计算 Token 级别可见区间
- Block_Sparse_Meta Kernel 将区间信息转换为块稀疏元数据 (F/P/E)
- Attention Kernel 利用块稀疏元数据跳过全掩码 Block，仅在 PARTIAL_BLOCK 逐元素处理

## 适用场景

当前规划的 Mask 覆盖范围：

| 场景 | 说明 |
|------|------|
| SLA Mask | 生成式推荐 SLA 场景 [参考实现](https://gitcode.com/lihao_huawei/design/ultra_hstu/sla_mask_verify.py) |
| DeepSeekV4-PRO | 大模型推理场景 |
| FLASH | 大模型训练/推理场景 |
| GLM 5.2 | 大模型场景 |

# 验收标准

## 功能验收

- 用户提供的任意 Mask Python 代码，Agent 能在 **天级** 内生成等价 Kernel
- 生成的 Kernel 支持 AscendC 和 Triton 两种后端

## 精度验收

- Arbitrary_Func.py 与原始 Mask.py 对比：100+ 用例全通过
- Kernel 与 Arbitrary_Func.py 对比：100+ 用例全通过
- 端到端 Attention 精度与 PyTorch 原生实现误差 ≤ 1e-3 (fp16)

## 性能验收

- 带 Block_Sparse 的 Attention 性能不低于对应手工定制算子的 90%
- 相比外部传 Mask 方案有显著性能提升（利用稀疏跳过）

## 自动化验收

- 精度对比失败时 Agent 能自动回溯重新生成
- 全流程无需人工介入即可完成 Mask → Kernel 转换

# 参考资料

- [AscendC SIMT 最佳实践](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-simt-best-practices)
- [AscendC SIMT Tiling 设计](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-simt-tiling-design)
- [AscendC UT 开发](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-ut-develop)
- [AscendC ST 设计](https://gitcode.com/CANN/cannbot-skills/ops/ascendc-st-design)

# 意见征集周期

截止 2026-07-30

# 抄送名单

yukunQin

# 其他补充说明

欢迎加入社区，感谢您对社区的贡献 🎉!
