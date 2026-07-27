# hstu_attn_metadata (AI CPU 自定义算子) PyTorch 绑定

将 HSTU 的 `hstu_attn_metadata`（纯 AI CPU 算子）以「独立编译」方式接入
RecSDK，并通过 `torch.ops.mxrec.hstu_attn_metadata` 暴露给 PyTorch。

算子在 AI CPU 上执行 **SectionStreamK 负载均衡**，产出一段 INT32 metadata（用户预分配、算子内
in-place 写入），供后续 FlashAttention 主算子做**分核调度**使用。

---

## 一、依赖关系（与普通 AscendC 算子不同）

本算子的 aclnn 接口位于**独立 vendor** 的 `libcust_opapi.so`，运行时由 `EXEC_NPU_CMD` 通过
`dlopen` 动态加载，因此使用前必须先编译并安装 AICPU kernel + aclnn vendor 包：

```bash
cd ../../../../ascendc_op/ai_core_op/hstu_attn_metadata
bash run.sh --stage=install        # 交叉编译 AICPU kernel + host 编译 aclnn，安装到 opp/vendors
```

## 二、编译 torch 插件

```bash
source /opt/buildtools/torch_v2_pt2.7.1/bin/activate   # torch/torch_npu 环境
source /usr/local/set_cann_env.sh a2
bash build_ops.sh                  # 产出 build/libhstu_attn_metadata.so
```

集成构建（`RecOps.cmake`）已登记本算子适配源，随聚合 adapter .so 一起编译，二者可并存。

## 三、运行前置环境变量

```bash
export ASCEND_CUSTOM_OPP_PATH=<path-to-repo>/cust_op/ascendc_op/ai_core_op/hstu_attn_metadata/build/vendor
```

---

## 四、接口签名与输出布局

```python
metadata = torch.ops.mxrec.hstu_attn_metadata(
    cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,   # 4 个可选 int32 张量，可传 None
    batch_size, max_seqlen_q, max_seqlen_kv,              # int 属性
    num_heads_q, num_heads_kv, head_dim,
    mask_mode, win_left, win_right,
    layout_q, layout_kv, layout_out,                     # str，如 "BSND"
)  # -> int32 Tensor (metadata)
```

### 4.1 入参详解

| 参数 | 类型 | 含义 | 说明 |
| --- | --- | --- | --- |
| `cu_seqlens_q` | `Tensor?` int32 | Q 的**累积**序列长度（前缀和），长度 `batch+1` | 变长/TND 场景用；不用传 `None` |
| `cu_seqlens_kv` | `Tensor?` int32 | KV 的累积序列长度，长度 `batch+1` | 同上 |
| `seqused_q` | `Tensor?` int32 | 每个 batch **实际** Q 序列长度，长度 `batch` | 提供后覆盖 `max_seqlen_q`，并据此推导 `batch` |
| `seqused_kv` | `Tensor?` int32 | 每个 batch 实际 KV 序列长度，长度 `batch` | 提供后覆盖 `max_seqlen_kv` |
| `batch_size` | `int` | batch 数 B | 传 `-1` 时从 `seqused_q`/`cu_seqlens_q` 推导 |
| `max_seqlen_q` | `int` | Q 最大序列长度 S1 | 无 `seqused_q` 时所有 batch 用此值 |
| `max_seqlen_kv` | `int` | KV 最大序列长度 S2 | 无 `seqused_kv` 时所有 batch 用此值 |
| `num_heads_q` | `int` | Q 头数 N1 | 必须能被 `num_heads_kv` 整除 |
| `num_heads_kv` | `int` | KV 头数 N2（GQA 分组数） | `G = num_heads_q / num_heads_kv` 为组大小 |
| `head_dim` | `int` | 每个头维度 D | 影响基本块 `s2BaseSize`（D=256 时切更小） |
| `mask_mode` | `int` | 稀疏/mask 模式 | 0=DEFAULT,1=ALL,2=LEFT_UP_CAUSAL,3=RIGHT_DOWN_CAUSAL,4=BAND；0 表示无 mask |
| `win_left` | `int` | 左窗口（preToken） | `-1` 表示无限（全部可见） |
| `win_right` | `int` | 右窗口（nextToken） | `-1` 表示无限 |
| `layout_q/kv/out` | `str` | 数据排布 | `BSND`/`BNSD`/`BSH`/`TND` 等 |

### 4.2 输出张量：预分配大小

- **dtype**：`int32`，一维。
- **预分配公式**（torch 插件，最坏情况上界）：

```text
elems   = ((AIC_CORE_NUM + AIV_CORE_NUM) * batch * num_heads_kv + 1) * 16
        = ((36 + 72) * B * Nkv + 1) * 16
aligned = ceil(elems / 4096) * 4096
```

- 原因：真实布局长度依赖运行时才知道的 `sectionNum`，而 `sectionNum ≤ B×Nkv`（每个 BN2 最多切成一段），
  故用 `B×Nkv` 当 `sectionNum` **上界**保守预分配；实际有效内容往往更短，多余部分为 0 / 对齐 padding。
- 常量：`AIC=36`（Cube）、`AIV=72`（Vector）、每条记录 **stride=16** 个 int32。

### 4.3 内存布局总览

逻辑布局（见 `op_kernel_aicpu/hstu_attn_metadata.h`）：

```text
metadata[int32]:
┌────────────────── HEAD ──────────────────┐
│  16 × int32                               │  固定 1 条
├────────────────── FA ────────────────────┤
│  [sectionNum][36][16]                     │  每个 section × 每个 AIC 各 1 条
├────────────────── FD ────────────────────┤
│  [sectionNum][72][16]                     │  每个 section × 每个 AIV 各 1 条
└───────────────────────────────────────────┘
（其后可能还有 4096 对齐产生的 padding）
sectionNum为BatchSize*HeadNum提前计算最大
36为AIC最大数量
72为AIV的最大数量
这三个数值都是为了申请metadata的内存大小时，按照最大来申请

```

**字节/元素偏移**（单位：int32 下标）：

| 段 | 起始下标 | 长度（int32 个数） |
| --- | --- | --- |
| HEAD | `0` | `16` |
| FA | `16` | `sectionNum × 36 × 16` |
| FD | `16 + sectionNum×36×16` | `sectionNum × 72 × 16` |

单条记录寻址：

```text
FA[sec][aic][k] = metadata[16 + sec*36*16 + aic*16 + k]
FD[sec][aiv][k] = metadata[16 + sectionNum*36*16 + sec*72*16 + aiv*16 + k]
```

> **多 section 时**：同一物理核 `AIC0` 在 `FA[0][0]`、`FA[1][0]`… 各有一条独立记录；
> 段内区间仍连续，段间串行消费。未使用的 `aic/aiv` 槽位保持全 0。

### 4.4 HEAD 段（16 个 int32，只用前 4 个）

| 下标 | 字段 | 含义 |
| --- | --- | --- |
| 0 | `sectionNum` | section 个数（沿 BN2 切了几段） |
| 1 | `isFd` | 是否启用 flash-decoding 归约（0/1） |
| 2 | `mBaseSize` | M(=G×S1) 方向基本块大小（token 数） |
| 3 | `s2BaseSize` | S2(KV) 方向基本块大小（token 数） |
| 4..15 | （保留） | 填 0 |

### 4.5 FA 段（每个 AIC 一条，16 个 int32，前 7 个有效）

格式：`[bn2Start, mStart, s2Start, bn2End, mEnd, s2End, fdWsIdx, 0…]`

迭代空间是三层嵌套 **`bn2`(外) → `m`(中) → `s2`(内)**。每个核负责展平序列上从 start 到 end 的**连续开区间**。

| 下标 | 字段 | 含义 |
| --- | --- | --- |
| 0 | `bn2Start` | 起始 BN2 = `b*Nkv + kvHead` |
| 1 | `mStart` | 起始 M 基本块索引 |
| 2 | `s2Start` | 起始 S2 基本块索引 |
| 3 | `bn2End` | 结束 BN2（开区间上界） |
| 4 | `mEnd` | 结束 M 基本块索引（开区间） |
| 5 | `s2End` | 结束 S2 基本块索引（开区间） |
| 6 | `fdWsIdx` | 本核第一份 partial 在 FD workspace 的落点；参与跨核归约时用 |
| 7..15 | （保留） | 0 |

起止采用**接力**：当前核 start = 上一核 end（section 内首核 start = 上一 section 末核 end，section0 首核从 0 起）。
有效核判定：`(bn2Start,mStart,s2Start) < (bn2End,mEnd,s2End)`；否则该槽未使用（全 0）。

### 4.6 FD 段（每个 AIV 一条，16 个 int32，前 6 个有效）

格式：`[bn2Idx, mIdx, wsIdx, wsNum, mStart, mNum, 0…]`

仅当 `isFd=1` 且该 AIV 被分到归约任务时非零。描述：**把 FA 阶段沿 S2 拆出的多份 partial 合并成最终 O**。

| 下标 | 字段 | 含义 |
| --- | --- | --- |
| 0 | `bn2Idx` | 归约对应的 BN2 |
| 1 | `mIdx` | 归约对应的 M 基本块 |
| 2 | `wsIdx` | partial 在 workspace 的起始下标（对齐 FA 的 `fdWsIdx`） |
| 3 | `wsNum` | 该行被拆成几份 = 要合并的 partial 个数 |
| 4 | `mStart` | 本 AIV 负责的 M 轴相对起点（一行可再拆给多个 AIV） |
| 5 | `mNum` | 本 AIV 负责的 M 轴行数 |
| 6..15 | （保留） | 0 |

> FD 必须由下游 FA 主 kernel 的 **Vector 归约路径**执行（Cube 写 partial，AIV 做 online-softmax combine）。
> metadata 只排班，不算注意力。

---

## 五、分核逻辑

### 5.1 总体流水线

把三维空间 **`BN2(=B×Nkv) × M(=G×S1) × S2`** 切成基本块，再分给 AIC；若一行被跨核切开，再排 AIV 归约。

| 步 | 做什么 |
| --- | --- |
| 1. 定 block | `AdjustSinnerAndSouter` → `mBaseSize` / `s2BaseSize` |
| 2. 建网格 | 每 batch：`mBaseNum=Ceil(G·S1/mBaseSize)`，`s2BaseNum=Ceil(S2/s2BaseSize)` |
| 3. 切 section | 沿 BN2 按 L2(96MB) 切段；段内并行、段间串行 |
| 4. 算 cost | 结合 mask 裁掉无效 S2 块，统计有效块与代价 |
| 5. FA 分核 | 在 `[minCore,maxCore]` 枚举核数；no-FD / with-FD 两套方案择优 |
| 6. FD 排班 | 若选 with-FD：把归约任务分给 AIV，写 FD 段 |

**不是**「每个 batch 固定 36 核均分」。调度单元是 **section 内的 BN2 序列**；核数动态；是否拆行取决于是否走 FD。

### 5.2 怎么确定 block 大小

`AdjustSinnerAndSouter`（与 layout 无关）：

```text
默认:     sOuter=64,  sInner=128
decode 典型（Q短 KV长、窗口大、mask≠左上因果、D≤128）:
          sOuter=32,  sInner=256
D==256:   sOuter=32,  sInner=256

mBaseSize  = sOuter × (aiv/aic) = sOuter × 2     # Cube:Vector=1:2，放大 M 块喂饱 Vector
s2BaseSize = sInner
```

### 5.3 怎么做 section

- **BN2** = `b*Nkv+n2`，长度 `B×Nkv`（要调度的「单 head」个数）。
- **section** = 沿 BN2 切出的**连续区间**；一段可含多个 head。**默认常是 `sectionNum=1` 覆盖全部 BN2**，不是「每个 head 一个 section」。
- 切法：按序累加 `singleHeadCost≈S1·D·2·sizeof(Q)+S2·D·2·sizeof(KV)`，快超过 L2(96MB) 就切一刀。
- 短路成 1 段：`maxGS1Size≤mBaseSize`（decode 常见）、或单 head 已很轻、或未设 L2。
- 每个 section **独立** `ScheduleSection`；FA/FD 按 `[section][core]` 分槽存储。

### 5.4 怎么做 FD

两套方案每个 section 都算一遍再择优：

| 方案 | 能否沿 S2 拆行 | 分配粒度 |
| --- | --- | --- |
| no-FD | 否（一行一核） | `AssignByBatch` → `AssignByRow` |
| with-FD | 是 | 再加 `AssignByBlock`；记录 `fdWsIdx` / `wsNum` |

择优：`sectionNum>1` 倾向 FD；no-FD 已足够轻则不 FD；否则比 `maxCost`。

有 FD 时下游必须：**AIC 写 partial → AIV `FlashDecode` 归约**。

### 5.5 具体实例：分核怎么算出来

输入（decode，多 batch；对应 C++ example 量级）：

```python
# B=4, Sq=1, Skv=8192, Nq=Nkv=32, D=128, mask=3, layout=BSND
# 设备 aic=36, aiv=72
```

| 步骤 | 结果 |
| --- | --- |
| block | decode 分支 → `sOuter=32,sInner=256` → `mBaseSize=64`, `s2BaseSize=256` |
| 网格 | 每 batch：`mBaseNum=Ceil(1/64)=1`，`s2BaseNum=8192/256=32` |
| section | `maxGS1Size=1≤64` → **强制 `sectionNum=1`**（整条 BN2=`4×32=128` 一段） |
| 有效块 | 每 head 可见全部 32 个 S2 块 → `128×32=4096` 块 |
| 核数 | `maxCore=36`，枚举后开满；按 cost 贪心切 |
| FD | M 只有 1「行」、S2 极长 → 倾向 **with-FD**，`isFd=1` |

HEAD 预期：`sectionNum=1, isFd=1, mBaseSize=64, s2BaseSize=256`。

更小的单 head 实测（便于看清「一行拆多核」）见下一章。

---

## 六、完整输入输出实例（实测，带 section 槽位）

真机（CANN 9.0.0 / A2，`aic=36,aiv=72`）。复现：`cust_op/test/hstu_attn_metadata/dump_example.py`。

### 6.1 输入

```python
metadata = torch.ops.mxrec.hstu_attn_metadata(
    None, None, None, None,
    1,                               # B=1
    1,                               # Sq=1 (decode)
    8192,                            # Skv=8192
    1, 1,                            # Nq=Nkv=1 → BN2 长度=1
    128, 3, -1, -1,
    "BSND", "BSND", "BSND",
)
```

### 6.2 预分配与有效区

```text
预分配 elems = ((36+72)*1*1 + 1)*16 = 1744
对齐后 numel = 4096（后段 padding）

实际 sectionNum=1，有效布局：
  HEAD:          下标 [0, 16)
  FA section0:   下标 [16, 16+36*16) = [16, 592)     ← 36 个 AIC 槽，只用前 20 个
  FD section0:   下标 [592, 592+72*16) = [592, 1744) ← 72 个 AIV 槽，只用 AIV0
  padding:       [1744, 4096)
```

本例只有 **1 个 section**，故没有 `FA[1][*]`。但 FA/FD **永远按 section 维排布**；
下面显式标成 `section0`，并把**未使用槽**写成全 0，方便对照「多 section 时每个 section 各有一套 36+72 槽」。

若存在 section1，会紧接在 section0 的 FA/FD 之后：

```text
FA[1][AIC0] 起点 = 16 + 1*36*16
FD[1][AIV0] 起点 = 16 + sectionNum*36*16 + 1*72*16
```

### 6.3 HEAD

| 字段 | 值 | 含义 |
| --- | --- | --- |
| sectionNum | 1 | 仅 section0（M 太小短路） |
| isFd | 1 | 启用 FD |
| mBaseSize | 64 | M 基本块 |
| s2BaseSize | 256 | S2 基本块 → 8192/256=**32** 个 S2 块 |

### 6.4 FA：按 section × AIC 列出（含空槽）

格式 `[bn2Start, mStart, s2Start, bn2End, mEnd, s2End, fdWsIdx]`。

```text
======== section 0 / FA（AIC 分核）========
# 有效：AIC0~AIC19 沿 S2 切开同一行 (bn2=0,m=0)；AIC20~35 未使用 → 全 0

section0 AIC 0:  [0, 0,  0, 0, 0,  2,  0]   # S2 块 [0,2)， partial → ws 0
section0 AIC 1:  [0, 0,  2, 0, 0,  4,  1]   # S2 块 [2,4)， partial → ws 1
section0 AIC 2:  [0, 0,  4, 0, 0,  6,  2]   # S2 块 [4,6)
section0 AIC 3:  [0, 0,  6, 0, 0,  8,  3]
section0 AIC 4:  [0, 0,  8, 0, 0, 10,  4]
section0 AIC 5:  [0, 0, 10, 0, 0, 11,  5]   # 仅 1 块
section0 AIC 6:  [0, 0, 11, 0, 0, 13,  6]
...   # AIC7~AIC18 同理沿 S2 推进
section0 AIC18:  [0, 0, 29, 0, 0, 31, 18]
section0 AIC19:  [0, 0, 31, 1, 0,  0, 19]   # 收尾：最后 1 块，bn2End 推到 1

section0 AIC20:  [0, 0,  0, 0, 0,  0,  0]   # 空槽（未使用）
section0 AIC21:  [0, 0,  0, 0, 0,  0,  0]
...
section0 AIC35:  [0, 0,  0, 0, 0,  0,  0]

======== section 1 / FA（本例不存在）========
# sectionNum=1，没有 FA[1][*]。
# 若有多 section，这里会再出现一套 AIC0..AIC35，例如：
# section1 AIC 0: [bn2Start, ...]   # 新一段 BN2 上的连续区间
# section1 AIC 1: ...
# section1 AIC35: [0,0,0,0,0,0,0]   # 该段未用满的空槽同样为 0
```

解读：所有有效核 `bn2Start=0,mStart=0`，只在 `s2` 上接力 → **with-FD / stream-K**；
`fdWsIdx=0..19` 对应 20 份 partial。

### 6.5 FD：按 section × AIV 列出（含空槽）

格式 `[bn2Idx, mIdx, wsIdx, wsNum, mStart, mNum]`。

```text
======== section 0 / FD（AIV 归约）========
section0 AIV 0:  [0, 0, 0, 20, 0, 1]   # 合并 ws[0..20) 共 20 份，M 上 1 行
section0 AIV 1:  [0, 0, 0,  0, 0, 0]   # 空槽
...
section0 AIV71:  [0, 0, 0,  0, 0, 0]

======== section 1 / FD（本例不存在）========
# 若 sectionNum≥2，这里再有一套 AIV0..AIV71
```

### 6.6 对照：关掉 FD 时

`Skv=2048` 负载较小时，常变为 `isFd=0`：`section0 AIC0` 单独吃完整 head，其余 AIC/全部 AIV 槽为 0——仍是同一套 `[section][core]` 布局，只是绝大多数槽空着。
