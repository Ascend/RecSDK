# CausalMaskPredictor 设计步骤（讨论记录）

> 本文记录 `causal_mask_predictor.h` 从空骨架到可用的设计结论与实施步骤。
> 讨论日期：2026-09-20 ~ 09-21。状态：history-only 已实现（get_blk_weight 闭式见 §5，
> 2026-09-21 已化简为**单式**并删除 `kSeqId==0` 特例，验证见 §5.0/§5）；get_head_weight 继承基类；
> 决策点 1 已拍板（含前置条件 `seqLenQ <= seqLenK`，见 §5.1/§6），余 2 个待拍板（见文末）。

## 0. 记号约定

- 块下标沿用 device 源码命名：`qSeqId`（Q 轴块号）、`kSeqId`（K 轴块号），仅用于文档与头注释公式；
  host 代码字段仍是 `RowWeightCtx::kBlkId`（同义，不改名）。
- 块大小用轴字母命名：`kBlkSize` = K 轴行块大小（metadata BLOCK_M，128/64）、
  `qBlkSize` = Q 轴列块大小（metadata BLOCK_N，256/128）。不用 device 的 BLOCK_M/BLOCK_N，原因见 §1。
- `qBlkCnt` = 本 batch 的 Q 轴块数（= `BackwardInput::qBlkCnt[b]`，公式中省略 [b]）。
- `qSeqMin(kSeqId)` = 固定 kSeqId 时满足判据的最小 qSeqId（闭式核心量）。
- `nCtxQ / nCtxK` = context 区 Q/K 轴块数（沿用头注释原记号）。
- **前置条件 `seqLenQ <= seqLenK`（`deltaQK >= 0`）**：闭式与 device 逐块等价的前提，
  成因（device 侧无符号隐式转换）与失效边界见 §5.1。当前按设计约定不考虑 `seqLenQ > seqLenK`。

## 1. 命名映射（最易踩坑点，已闭环）

device 侧 `BLOCK_M/BLOCK_N` 沿用 GEMM 惯例（M = score 矩阵行 = Q 轴）；
metadata 侧沿用调度惯例（M = 外层行 = K 轴）。**两套命名，数值完全吻合**：

| TILE_K 档 | metadata BLOCK_M（K 行块，wire mBaseSize）= device BLOCK_N | metadata BLOCK_N（Q 列块，wire s2BaseSize）= device BLOCK_M |
| --- | --- | --- |
| 128（head_dim ≤ 128） | 128 | 256 |
| 256（head_dim = 256） | 64 | 128 |

验证：device `belowDig = kSeqId ≤ (qSeqId·BLOCK_M_dev + deltaQK + BLOCK_M_dev − 1) / BLOCK_N_dev`，
代入 `BLOCK_M_dev → qBlkSize`、`BLOCK_N_dev → kBlkSize` 后即头注释公式。**头注释公式本身是对的**。

下文一律使用 §0 记号。

## 2. Causal 场景的 5 类块形态（device Classifier/IsSkip/ApplyMask）

| 形态 | 位标志 | 判据要点 | 对权重的影响 |
| --- | --- | --- | --- |
| History（对角块） | HAS_HISTORY=0x2 | `lblk ≤ kSeqId ≤ rblk`（belowDig 且 diagonal） | 计入权重 |
| Context | HAS_CONTEXT=0x1 | `qSeqId < CeilDiv(numContext, qBlkSize)` 且 `kSeqId < CeilDiv(seqlenK−numTarget, kBlkSize)` | 计入权重 |
| Target | HAS_TARGET=0x4 | target 行块（`kSeqId ≥ tbaseK`）且 tril | 计入权重，但 Q 轴被 targetQEnd 封顶 |
| Full | 无 | 严格下三角内部（tril 且非对角非 target） | 计入权重 |
| Skip | 无 | ① 越界 ② 上三角且无 context ③ `IsTargetSkip`（qSeqId ≥ targetQEnd） | 不计入 |

关键洞察：**mask 类型不影响权重值**（块要么算要么跳，mask 类型只改 mask buffer 内容），
权重公式只关心算/跳边界；但三种类型的几何参数各自贡献边界的一部分。

## 3. 完整权重公式（含 target 补全）

头注释的 `weight = |A ∪ B|` **未含 target 封顶**（`numTarget > 0` 时会多算）。完整式：

```text
A = [qSeqMin(kSeqId), qBlkCnt)       // tril 区，qSeqMin 由 belowDig 不等式解出
B = [0, nCtxQ)                       // context 区，仅当 kSeqId < nCtxK 且 numContext > 0
T = [0, targetQEnd(kSeqId))          // target 封顶，仅当 kSeqId 在 target 区（numTarget > 0）
weight(kSeqId) = |(A ∪ B) ∩ T|       // clamp 到 [0, qBlkCnt]
```

其中 `targetQEnd` 移植自 device 侧 `GetTargetQBlockEnd`（含 targetGroupSize 对齐，int64 运算）。

依赖形态（device 证据）：`numContext/numTarget` 是 **per-batch 张量**（`gNumContext.GetValue(b)`），
`targetGroupSize` 是标量 tiling。三者由 ① 解析进 `BackwardInput::mask`（`MaskParams` 结构体）。

## 4. 实施步骤

### 步骤 0：语义对齐（已完成，即本文 §1~§3）

### 步骤 1：补依赖缺口（算子接口扩展，工作量大头）

1. ① `PrepareInput`：解析新输入（per-batch `numContext/numTarget` + 标量 `targetGroupSize`）进 `BackwardInput::mask`；
2. 连带改动（DESIGN.md §7 清单）：opdef json → l0 launcher → host check（mask_mode=1 放行）→ torch 绑定 schema → examples/pytest 辅助函数；
3. metadata 布局契约不动（与 mask 类型无关）。

### 步骤 2：实现 `get_blk_weight`（必须，语义核心）

按 §3 三个区间分量求交。易错点：

- 中间量必须用 int64（`kSeqId * kBlkSize` 先抬位再运算）；`deltaQK` 在 §5.1 前置条件下非负，
  故**不涉及**「负 CeilDiv 截断到 0」，`seqLenQ > seqLenK` 已按 §6 决策 1 移出范围；
- `qSeqMin = CeilDiv(t, qBlkSize)`：门槛已**化简为单式**（不再按 `kSeqId` 分段，见 §5.0 末段）；
- `targetQEnd` 的 int64 边界照搬 device 写法。

### 步骤 3：`get_head_weight` 分两版

- 第一版：**不覆写**，基类默认逐块累加（天然满足恒等契约）；
- 第二版（可选优化）：分段闭式求和（分段点：nCtxK、target 区起点、qSeqMin 的 ceil 边界），
  须配「闭式 == 逐块累加」参数网格等价验证后才合入。

### 步骤 4：`get_total_weight` 不动

无调用点；默认实现自动吃到步骤 3 的好处；覆写零收益。

### 步骤 5：验证

- host_ut（目录待补）：① 块级权重对拍（get_blk_weight vs 按 5 类形态逐块暴力判定）；
  ② 闭式恒等验证（若做）；③ 退化/边界场景（numContext=0、numTarget=0、**deltaQK=0 与 deltaQK≫0**、空序列）。
  注：`deltaQK < 0` 已按 §6 决策 1 移出范围，不作为用例；`deltaQK == 0`（等长）是前提出入口，
  必须覆盖，`verify_device_semantics.py` 的扫描域已含该边界；
- 上板：metadata 驱动 causal 反向端到端精度。

## 5. History-only 场景的 `get_blk_weight` 闭式（本轮重点结论）

History-only = `numContext == 0 && numTarget == 0`，权重 = `|A|` 一项。
**无需任何接口扩展**（mask_mode 属性已存在，seqLen/块大小全在 BackwardInput）。

device 判据（C++ 整数除法，截断向零；在前置条件 `deltaQK >= 0` 下截断方向不产生分歧，见 §5.1）：

```text
belowDig(qSeqId, kSeqId) = (kSeqId <= truncDiv(qSeqId*qBlkSize + deltaQK + qBlkSize - 1, kBlkSize))
```

### 5.0 `N(qSeqId)` 的三项来源 与 `t` 的由来

**`N(qSeqId)` 是"第 qSeqId 个 Q 块在 K 轴坐标系上的右端点"（一个 token 下标）**，三项依次为：

| 项 | 含义 |
| --- | --- |
| `qSeqId * qBlkSize` | 块左端在 **Q 自身坐标系**中的位置 |
| `+ deltaQK` | 坐标平移：两条序列**右对齐**，Q 坐标 `i` ↔ K 坐标 `i + deltaQK` |
| `+ qBlkSize - 1` | 从块左端走到块右端（块横跨 qBlkSize 个 token，闭区间故 -1） |

**为什么取右端点**：块级判定问"两个块在 K 轴上是否相交"，等价于「一方最左 ≤ 另一方最右」。
device 取的正是两极——`rcol`（Q 块最右）对 `kSeqId * kBlkSize`（K 块最左）。
取端点而非区间本身，是块级判定能塌成一次整数比较的原因。

**`t` 是反解产物，不是自由设计量。** 反解四步：

```text
1. device 原式      kSeqId <= floor( N(qSeqId) / kBlkSize )
2. 消 floor         kSeqId * kBlkSize <= N(qSeqId)          // 需 rcol >= 0，即 deltaQK >= 0（§5.1）
3. 展开 N           kSeqId * kBlkSize <= qSeqId*qBlkSize + deltaQK + qBlkSize - 1
4. 移项孤立 qSeqId  qSeqId * qBlkSize >= kSeqId*kBlkSize - deltaQK - qBlkSize + 1
                                        └────────────────── t ──────────────────┘
```

第 1 步不可跳：floor 是舍入运算、不可逆，不降为线性不等式就无法解出 qSeqId。
第 2 步能否做，正是 §5.1 前置条件在代数上的落点。

`t` 的等价形态：`t = kSeqId*kBlkSize - N(0)`（因 `N(0) = deltaQK + qBlkSize - 1`）。
语义即 **"qSeqId=0 差多少"**——因 `N` 关于 qSeqId 单调，`q=0` 一够则整行可算、一不够才需右移，
故以 `N(0)` 为参照是自然的。孤立后 qSeqId 系数为 `qBlkSize`，最小整数解即 `CeilDiv(t, qBlkSize)`；
几何上 `qSeqId` 每加 1，`N` 沿 K 轴恰右移 `qBlkSize`（纵横 1:1），故"横向差 ÷ 块长"直接给出要右移几格。

注意 `t` 中的 `+1` **不是取整补偿**，而是 `N` 里 `+(qBlkSize - 1)` 移项变号的产物（闭区间端点约定）。

`kBlkId == 0` 的特例写法（门槛 `-kBlkSize + 1`）是全式里**唯一**为复刻 device 截断语义而加的修正
（门槛放宽 `kBlkSize - 1` 格）；在 `deltaQK >= 0` 下恒有 `t <= 0`、不起作用。
**该特例已于 2026-09-21 从实现中删除**（改为上方单式），依据两条：

1. 前提内它与单式**逐点等价**——307968 个行块样本 0 分歧（`verify_special_branch_value.py`）；
2. 「保留为 `deltaQK < 0` 兜底」的理由**不成立**——该域闭式的前提（可算集合 = 单调后缀）已崩，
   保留特例仍有 194106 个 (用例,行块) 与 device 不对齐（删掉为 242376，两者都不可用），
   真要支持必须整体重推而非局部特化（`verify_causal_v2.py` B 段）。

**为什么特例只可能落在 `kBlkId == 0`（分支数 = 1，不是 2+）。**
device 判据可化为「门槛 = `kBlkId * kBlkSize`，比较 `trunc(rcol / kBlkSize) >= kBlkId`」。
由截断函数性质：

| 门槛 n | 等价条件 | 依据 |
| --- | --- | --- |
| `n >= 1` | `trunc(x) >= n` ⟺ `x >= n` | `trunc(x) <= x < trunc(x)+1` |
| `n = 0` | `trunc(x) >= 0` ⟺ `x > -1`（**不是** `x >= 0`） | 负数截断成 0 |

`kBlkId` 取值域为 `{0,1,2,...}`，故唯一贴着截断不连续点（`n = 0`）的就是 `kBlkId == 0`：
其门槛为 `rcol >= -kBlkSize + 1`；而 `kBlkId >= 1` 的门槛是 `kBlkId*kBlkSize >= kBlkSize > 0`，
深在正区（`trunc == floor`），无需特例。**分段的依据必然是 `kBlkId`**：它既是 device 判据左侧的
被比较量（`p.kSeqId`），也是唯一决定「门槛落在哪个数值区间」的变量；而 `rcol` 是 `qSeqId` 的函数
（内层变量），不能在它上面分支。

等价写法（**仅用于说明特例式的来历，实现已不采用**，删除依据见上）：
`threshold = (kBlkId == 0) ? (-kBlkSize + 1) : (kBlkId * kBlkSize)`；`t = threshold - deltaQK - qBlkSize + 1`。
即「门槛的正常值是 `kBlkId*kBlkSize`，`kBlkId == 0` 时被截断向零拉低到 `-kBlkSize + 1`」——
特例式里的 `+2` 与 `-kBlkSize` 由此可自明，并非笔误。

左端关于 qSeqId 单调不减 ⇒ 可算集合是后缀 `[qSeqMin(kSeqId), qBlkCnt)`，O(1) 闭式
（**已化简为单式**，与 `causal_mask_predictor.h::get_blk_weight` 逐行对应）：

```text
t = kSeqId*kBlkSize - deltaQK - qBlkSize + 1   // 门槛统一为 kSeqId*kBlkSize，无 kSeqId==0 特例
if (t <= 0) weight = qBlkCnt                   // qSeqMin = 0：整行块可算
else        weight = qBlkCnt - CeilDiv(t, qBlkSize)
```

已用样例验证：等长 512（kBlkSize=128, qBlkSize=256）得权重序列 2,2,1,1（总 6 块，几何核对一致）。

**单式化验证（2026-09-21）**：`deltaQK >= 0` 全域 270920 例与 device 逐块**零分歧**，
且与化简前的三目版**逐点相同**（纯行为保持重构，见 `verify_causal_v2.py`）。

### 5.1 前置条件：`seqLenQ <= seqLenK`（`deltaQK >= 0`）

device 的 `NeedHistoryMask`（`causal_mask_predictor.hpp:134-140`）用 **uint32_t** 形参 `BLOCK_M/BLOCK_N`
参与运算，触发 C++ 常用算术转换：

```cpp
const int qBase = qSeqId * BLOCK_M + deltaQK;   // 无符号乘加 → 转 int
const int rcol  = qBase + BLOCK_M - 1;
const int rblk  = rcol / BLOCK_N;               // int / uint32_t → 退化为无符号除法
belowDig        = (kSeqId <= rblk);             // uint32_t <= int → 无符号比较
```

- **`deltaQK >= 0`（本闭式的前提）**：`qBase >= 0`、`rcol >= BLOCK_M−1 >= 0`，上述四步的隐式转换
  **全部保值** ⇒ device 等价于「有符号除法 + 有符号比较」，与本文闭式**严格等价**。
  此时 `truncDiv == floorDiv`，且 `kSeqId == 0` 的两种 `t` 写法都落在 `t <= 0`（首个行块整行可算）——
  与「非负 delta 下第一行块必在三角内」的几何直觉一致，故**无需**再区分「截断向零」的边界特例。
- **`deltaQK < 0`（本设计不考虑）**：`rcol` 可为负 → 无符号除法把它变成巨大正数 →
  `belowDig` 对**所有** `kSeqId` 恒真 → device 的可算集合变成 `[0, qLow) ∪ [qSeqMin, qBlkCnt)`，
  即**前缀 ∪ 后缀**，纯后缀闭式会**低估**。等价性在该域**不成立**。

**后果可控**：权重是软约束，低估只让核间均衡变差（覆盖性由 device qBlockScheduler 全量驱动，
与 metadata 权重无关），**不会算错数**；表现为这些 case 变慢。

**验证记录**（脚本 `verify_device_semantics.py`；基准 = 严格照抄 device 类型语义的逐块暴力计数）：

| 扫描域（两档块大小各跑一遍） | 用例 / 采样点 | 分歧 |
| --- | --- | --- |
| `seqLen <= 260` 全量 | 67860 例 | **0** |
| `seqLen <= 520` 全量 | 270920 例 | **0** |
| 随机 `seqLenQ<=1e5, seqLenK<=1e6`，每例 8 个关键 `kSeqId` | 19200 点 | **0** |
| 对照：`seqLenQ > seqLenK` 全域（`seqLen <= 260`） | 67340 例 | 4771 个 (例,行块) 分歧点 |

前两行是 `deltaQK >= 0` 的严格等价证据；最后一行说明分歧**恰好**由 `deltaQK < 0` 触发，
边界落在 `deltaQK == 0`，与 §5.1 的机理分析一致。

## 6. 待拍板决策点

1. ~~**target 范围**：第一版是否只支持 history-only（numContext=numTarget=0）？~~ **已拍板：是**
   （2026-09-20 落地：get_blk_weight 按 §5 闭式实现，步骤 1 接口扩展整体跳过）。
   依据：权重是软约束——host/device 口径不一只影响核间均衡，不影响正确性
   （Q 内层遍历由 device qBlockScheduler 全量驱动，覆盖性与 metadata 权重无关）。

   **补充前提（2026-09-21 拍板）：`seqLenQ <= seqLenK`。** 闭式与 device 逐块等价的必要前提
   （device 侧 `rcol` 必须非负，见 §5.1）。验证已闭环：`deltaQK >= 0` 全域 **35.8 万例/采样点零分歧**，
   分歧恰好由 `deltaQK < 0` 触发。实现侧已把该前提写进 `causal_mask_predictor.h` 文件头；
   `seqLenQ/seqLenK` 由 AICPU 侧 `PrepareInput` 从 offsets 相减得到
   （`hstu_attn_metadata_backward_aicpu.cpp:180-181`），host 侧看不到数据、无法校验，
   如需 fail-loud 只能在 AICPU 加诊断。

   注意：预测器仍有**三道闸门**未放行——① 工厂 `kMaskCausal → CausalMaskPredictor` 接线
   （且未 include 该头文件）；② host check 放行 `mask_mode=1`；③ torch 绑定侧的
   `TORCH_CHECK(mask_mode == MASK_MODE_NO_MASK)`。三者当前都拒绝 causal，实现不可达。
2. **传入形式**（若支持 context/target）：numContext/numTarget 做 tensor 输入
   （对齐 device per-batch 语义，与 forward 侧一致）还是 intArray 属性？
3. **闭式优化**：步骤 3 第二版做不做，或等 host_ut 实测阶段 1 占比后再定？
