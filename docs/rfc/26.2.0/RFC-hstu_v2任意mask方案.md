# 需求背景

## 业务背景

HSTU_V2 是推荐领域 Attention 的核心算子（基于 Catlass，支持 Atlas 950 及后续版本）。在生成式推荐场景下需要支持**任意 Mask**，即不同客户、不同模型架构使用完全不同的 Attention Mask 模式。

任意 Mask 给 Attention 算子带来两个核心挑战：

1. **分核负载均衡复杂化**：无 Mask 场景下各 Q_Block 对应的 KV_Block 数量固定，负载均衡简单；带 Mask 后每个 Q_Block 可见的 KV_Block 数量随 Mask 形状动态变化
2. **长序列 L2 Cache 颠簸**：NPU 统一 L2 Cache 架构下，长序列整段分配多核会导致 Cache 逐出和性能急剧下降

## 当前痛点

| 问题 | 说明 |
|------|------|
| **分核性能瓶颈** | AICORE 的 Scalar 运算能力较弱，分核计算占据算子头开销（~300us） |
| **L2 Cache 颠簸** | 长序列（如 SEQ=81920, DIM=256 → 单 Batch 单 Head 40MB HBM）多核整段分配会打满 L2 Cache (128MB) 导致颠簸 |
| **带 Mask 负载不均** | 不同 Q_Block 对应的有效 KV_Block 数量差异大，简单轮询导致多核任务量严重不均衡 |
| **无块稀疏跳过** | 无法识别全掩码 Block 并跳过计算，浪费算力 |

# 提议方案

采用 **AICPU + AICORE 协同架构**，通过 AICPU 承担分核算力和负载均衡，AICORE 专注矩阵运算主逻辑。

## 整体架构

```text
                        ┌──────────────────────┐
                        │   ArbitraryFuncKernel │  ← AICORE: 算子化掩码表达 (省显存, 高性能)
                        └──────────┬───────────┘
                                   │ AF
                        ┌──────────▼───────────┐
                        │   BlockSparseMeta     │  ← AICORE: 计算块稀疏信息并输出压缩编码
                        └──────────┬───────────┘
                                   │ BS
┌──────────────────────┐  ┌────────▼───────────┐
│   AttentionMeta       │  │                     │
│   (AICPU)             │  │   Attention         │
│   · 分核计算           │  │   (AICORE)          │
│   · CHUNK 级负载均衡   │  │   · BlockScheduler  │
│   · 压缩编码输出        │  │   · ArbitraryMask   │
└──────────────────────┘  └─────────────────────┘
```

四个组件的职责：

| 组件 | 运行设备 | 职责 |
|------|---------|------|
| **ArbitraryFuncKernel** | AICORE | 算子化的 Mask 表达，省显存、高性能，对 Partial Block 用 SIMT 处理 |
| **BlockSparseMeta** | AICORE | 计算每个 Block 的状态 (FULL / PARTIAL / EMPTY)，输出压缩编码格式 |
| **AttentionMeta** | AICPU | 分核计算、CHUNK 级负载均衡、输出压缩编码供 Attention 使用 |
| **Attention** | AICORE | 注意力机制主逻辑，接收分核信息跳过 Full Block、仅 Partial Block 逐元素处理 |

## AICPU/AICORE 协同优势

### 为什么用 AICPU 做分核

```text
当前 AICORE 分核:                          AICPU + AICORE 协同:
┌─────────────────────────┐               ┌─────────────┐  ┌──────────────────────┐
│ AICORE:                  │               │ AICPU:       │  │ AICORE:               │
│ 1. 分核计算 (scalar, 慢)  │               │ 分核计算      │  │ 核函数主逻辑           │
│ 2. 输出初始化 (ZeroLike) │      →        │ (scalar, 快)  │  │ (aiv/aic/scalar)       │
│ 3. 核函数主逻辑           │               │ ≈100us       │  │                       │
│ 约 300us 头开销          │               └─────────────┘  └──────────────────────┘
└─────────────────────────┘                      ↓ 流水并行
                                            AICPU 分核 与 AICORE ZeroLike
                                            可并行执行，掩盖时延
```

- AICORE 的 Scalar 运算能力较弱 → 分核计算耗时约 **300us**
- AICPU 具备高性能标量运算能力 → 分核计算仅 **100us**
- AICPU 分核可与 AICORE 的 ZeroLike 做**流水并行**，进一步掩盖延迟
- AICPU 还可做更细粒度的负载均衡

## AICPU 详细设计

### CHUNK 分核策略

**问题**：长序列（如 4 条 SEQ=81920, DIM=256）单 Batch 单 Head 占用 40MB HBM，4 条同时分给 4 个 Core 则 L2 Cache (128MB) 无法容纳全部，产生淘汰颠簸。

**方案**：以 CHUNK 为粒度分核。

```text
假设 CHUNK_SEQ = 30MB → CHUNK_SEQ = 30×1024×1024 / DIM / 2 = 61440

总序列长度 = 4 × 81920 = 327680
CHUNK 个数 = CeilDiv(327680, 61440) = 6

(b=0,n=0)  (b=0,n=1)  (b=1,n=0)  (b=1,n=1)
├── CHUNK0 ──┤├── CHUNK1 ──┤├── CHUNK2 ──┤├── CHUNK3 ──┤├── CHUNK4 ──┤├── CHUNK5 ──┤
```

在一个 CHUNK 内多核进行分配，分配完一个 CHUNK 再继续多核分配下一个 CHUNK。确保同一时刻 L2 Cache 内的数据总量可控，避免颠簸。

### 无 Mask 场景的 CHUNK 负载均衡

多核按 CHUNK 轮询分核，维护 `task_num[core_num]` 数组记录每个核的当前任务量，每次分配时选择任务量最小的 Core。

```text
示例: KV_BLOCK=10, 3个CHUNK

CHUNK0 (Q_BLOCK=3):         CHUNK1 (Q_BLOCK=2):         CHUNK2 (Q_BLOCK=3, KV_BLOCK=3):
Core0: 10                    Core0: +10 = 20              Core2: +3 = 13
Core1: 10                    Core1: +10 = 20              Core2: +3 = 16
Core2: 10                                                 Core2: +3 = 19

最终: CORE0=20, CORE1=20, CORE2=19  ← 均衡
```

### 带 Mask 场景的 CHUNK 负载均衡

带 Mask 场景下，每个 Q_Block 对应的**有效 KV_Block 数量**由 Mask 形状决定，需从 BlockSparseMeta 的输出获取。

```text
示例: 2个CHUNK, 3个Core

CHUNK0 (Q_BLOCK=4):                 CHUNK1 (Q_BLOCK=3):
Q0 → 1个有效KV_BLOCK   Core0: +1=1   Q0 → 2个有效KV_BLOCK   Core1: +2=4
Q1 → 2个有效KV_BLOCK   Core1: +2=2   Q1 → 4个有效KV_BLOCK   Core0: +4=7
Q2 → 2个有效KV_BLOCK   Core2: +2=2   Q2 → 3个有效KV_BLOCK   Core2: +4=6
Q3 → 3个有效KV_BLOCK   Core0: +3=4

最终: CORE0=7, CORE1=6, CORE2=4
```

### 压缩编码输出

AICPU 将分核结果编码为一维 `uint32_t` 张量输出，供 Attention Kernel 使用：

```text
压缩编码格式 (一维 uint32_t):
┌──────────┬──────────┬──────────────────────────────────────────────────┐
│ core_num │ chunk_num│ core0[2] │ core1[2] │ ... │ chunk0               │
│  用核数   │ CHUNK数  │ 起始/结束 │ 起始/结束 │     │ 每个核在CHUNK的范围    │
└──────────┴──────────┴──────────────────────────────────────────────────┘
```

每个 Core 在每个 CHUNK 中记录两个 `uint32_t`：全局起始 BLOCK_ID 和结束 BLOCK_ID。

### 输出内存计算

```cpp
constexpr uint32_t CHUNK_SEQ_BYTE_SIZE = 30 * 1024 * 1024; // 30MB

Tensor GetAttentionOutput(Tensor Q) {
    uint32_t core_num = GetCurrentPlatformAICCoreNum();
    total_seq_q_byte_size = Q.size(0) * Q.size(1) * sizeof(float16);
    chunk_num = CeilDiv(total_seq_q_byte_size, CHUNK_SEQ_BYTE_SIZE);
    int64_t outputUint32Cnt = 2 + chunk_num * core_num * 2;
    return torch::zeros(outputUint32Cnt);
}
```

## AICORE 详细设计

### BlockScheduler 修改

| 调度器 | 原功能 | 修改后 |
|--------|--------|--------|
| **RowBlockScheduler** (外循环) | Q_Block 块调度 | 增加 AICPU CHUNK 信息访问：获取每个 CHUNK 的起止范围，`operator++` 遍历 |
| **ColumnBlockScheduler** (内循环) | KV_Block 块调度 | 增加 BlockSparseMeta 和 ArbitraryFunc 信息：判断 FULL (跳过) / EMPTY (正常) / PARTIAL (SIMT 掩码) |

### ArbitraryMask Predictor

新增文件 `arbitrary_mask_predictor.hpp`：

- 新增 `IS_ARBITRARY` 特化类
- `ApplyMask` 方法增加 **SIMT 运算逻辑**，根据 ArbitraryFunc 对 Partial Block 的每个 Token 进行置位

## 端到端调用流程

```cpp
// attention_fusion_op c++ torch_plugin
Tensor call_attention_fusion_npu_impl(
    Tensor Q, Tensor K, Tensor V,
    Tensor ArbitraryFunc, Tensor BlockSparseMeta
) {
    // Step 1: 调用 AICPU AttentionMeta，与 ZeroLike 流水并行
    auto attention_meta = GetAttentionOutput(Q);
    InvokeAicpuKernel(AttentionMeta, ArbitraryFunc, BlockSparseMeta,
                      CHUNK_SEQ_BYTE_SIZE, attention_meta);

    auto output = zeros_like(Q); // 与 AICPU 流水并行，掩盖 AICORE 时延

    // Step 2: 调用 AICORE Attention 算子
    InvokeAicoreKernel(Attention, Q, K, V, ArbitraryFunc,
                       BlockSparseMeta, attention_meta, output);
    return output;
}
```

# 方案目标

① **性能提升**：利用 AICPU 标量加速分核计算，将头开销从 ~300us 降低到 ~100us

② **Cache 亲和**：CHUNK 粒度分核避免长序列场景下 L2 Cache 颠簸

③ **负载均衡**：带 Mask 场景下基于真实 KV_Block 数量的动态负载均衡

④ **块稀疏跳过**：通过 BlockSparseMeta 识别 FULL Block 并跳过计算，Partial Block 用 SIMT 高效处理

# 总体设计

## 组件层次

```text
                        ┌─────────────────────────┐
                        │ ArbitraryFuncKernel      │  AICORE
                        │ (算子化 Mask 表达)        │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │ BlockSparseMeta          │  AICORE
                        │ (F/P/E 块状态 + 压缩编码) │
                        └────────────┬────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
┌─────────▼──────────┐    ┌──────────▼──────────┐    ┌──────────▼──────────┐
│ AttentionMeta       │    │ BlockScheduler      │    │ ArbitraryMaskPred    │
│ · CHUNK 分核         │    │ · 外循环: CHUNK 遍历 │    │ · IS_ARBITRARY 特化  │
│ · 无Mask 负载均衡    │    │ · 内循环: F/P/E 判断 │    │ · SIMT 逐元素置位    │
│ · 带Mask 负载均衡    │    │                      │    │                      │
│ · 压缩编码输出       │    └─────────────────────┘    └─────────────────────┘
│ AICPU               │
└────────────────────┘
```

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `arbitrary_mask_predictor.hpp` | 新增 | IS_ARBITRARY 特化类，SIMT ApplyMask |
| `block_scheduler.hpp` | 修改 | RowScheduler 接入 CHUNK 信息；ColumnScheduler 接入 F/P/E 判断 |
| `attention_meta_kernel` (AICPU) | 新增 | CHUNK 分核 + 负载均衡 + 压缩编码输出 |
| `attention_fusion_op` (torch_plugin) | 修改 | 增加 AICPU 调用 + ZeroLike 流水并行 |

## 开放问题

1. AICPU 分核计算流程是否可用**多线程并行**加速（AICPU 支持多线程）
2. 分核计算流程是否可用 **SIMT 计算**替代 AICPU Scalar
3. AICPU 分核是否可以与 AICORE 计算**完全并行**，实现在线分核、在线计算

# 验收标准

## 功能验收

- 支持任意 Mask 场景下 HSTU_V2 Attention 的正确计算
- 覆盖无 Mask / 简单 Casual Mask / SLA Mask / 自定义复杂 Mask 四类场景

## 精度验收

- 与 PyTorch 原生 Attention + Mask 实现精度对比，误差 ≤ 1e-3 (fp16)

## 性能验收

- 无 Mask 场景：分核开销从 ~300us 降至 ~100us
- 长序列场景（SEQ ≥ 32768）：L2 Cache 颠簸消除，性能不劣于分 CHUNK 前短序列场景
- 带 Mask 场景：各 Core 任务量差异 ≤ 10%
- Full Block 跳过率达到理论最大值（100% FULL Block 被跳过）

## 代码验收

- `arbitrary_mask_predictor.hpp` 完成 IS_ARBITRARY 特化
- `BlockScheduler` 完成 CHUNK + F/P/E 改造
- AICPU AttentionMeta Kernel 完成 CHUNK 分核 + 负载均衡 + 压缩编码

# 意见征集周期

截止 2026-07-30

# 抄送名单

yukunQin

# 其他补充说明

欢迎加入社区，感谢您对社区的贡献 🎉!
