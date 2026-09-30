# hstu_attn_metadata_backward (AI CPU 自定义算子) PyTorch 绑定

将 HSTU 反向的 `hstu_attn_metadata_backward`（纯 AI CPU 算子）以「独立编译」方式接入
RecSDK，并通过 `torch.ops.mxrec.hstu_attn_metadata_backward` 暴露给 PyTorch。

算子由 q/k 的 offsets（`cu_seqlens`）出发，为 **HSTU 反向的 K 轴行调度**产出 metadata：
把 BN2（= batch × num_heads）条序列的 K 行块切成若干 section，再在 section 内做
persist 分核，把每个核负责的半开区间写进 FA 槽位，供下游 HSTU 反向 kernel 取任务。

---

## 一、依赖关系（与普通 AscendC 算子不同）

本算子的 aclnn 接口位于**独立 vendor** 的 `libcust_opapi.so`，运行时由 `EXEC_NPU_CMD` 通过
`dlopen` 动态加载，因此使用前必须先编译并安装 AICPU kernel + aclnn vendor 包：

```bash
cd ../../../../ascendc_op/ai_core_op/hstu_attn_metadata_backward
bash run.sh --stage=install        # 交叉编译 AICPU kernel + host 编译 aclnn，安装到 opp/vendors
```

## 二、编译 torch 插件

```bash
source /opt/buildtools/torch_v2_pt2.7.1/bin/activate   # torch/torch_npu 环境
source /usr/local/set_cann_env.sh a2
bash build_ops.sh                  # 产出 build/libhstu_attn_metadata.so（前向 + 反向同一库）
```

> 反向绑定与前向 `hstu_attn_metadata.cpp` 同目录、编进同一个 .so；反向接口文档即本文件，
> 前向接口见同目录 README.md。

聚合构建（`torch_library/common/build_ops.sh`，产出 `libfbgemm_npu_api.so`）用
`file(GLOB_RECURSE "../*.cpp")` 收集源码，**会自动纳入本目录的 .cpp**，无需登记到
`RecOps.cmake` —— 该文件只登记 AscendC（`ai_core_op/<op>/<build_ver>/`）算子。
两种编译方式（独立 .so / 聚合 .so）可并存。

## 三、运行前置环境变量

```bash
export ASCEND_CUSTOM_OPP_PATH=<path-to-repo>/cust_op/ascendc_op/ai_core_op/hstu_attn_metadata_backward/build/vendor
```

> `ASCEND_CUSTOM_OPP_PATH` 必须指向**真正含 `op_impl/`** 的那一层。
> 源码树里直接跑时是 `build/vendor/hstu_attn_metadata_backward_transformer`。

---

## 四、接口签名与输出布局

```python
metadata = torch.ops.mxrec.hstu_attn_metadata_backward(
    cu_seqlens_q,        # Tensor, int32/int64, 1D, 长度 batch+1，必须首元素为 0 且单调不减
    cu_seqlens_kv,       # Tensor, int32/int64, 1D, 长度与 cu_seqlens_q 相同
    num_heads,           # int, 注意力头数（> 0）
    head_dim,            # int, 取值 32/64/128/256
    mask_mode=0,         # int, 可选，默认 0 = no mask；1=causal 已解锁；2=arbitrary 已解锁
    num_contexts=None,   # Tensor | None, int32/int64, 1D 长度 batch，仅 mask_mode=1 时使用
    num_targets=None,    # Tensor | None, 同上
    target_group_size=0, # int, causal 的 target 项组大小；<= 0 = 无 target
    full_cnt=None,       # Tensor | None, int32, numel = batch*maxBlkCntK，仅 mask_mode=2 时使用
    mask_cnt=None,       # Tensor | None, 同上；必须与 full_cnt 成对提供/成对缺席
)  # -> int32 Tensor (metadata)
```

两个 offsets 张量都是**必选**且必须是 NPU 张量（注册在 `PrivateUse1` 分发键上）；
`mask_mode` 带默认值 0。三种模式按「槽位分段固定」互斥使用：

- **no mask（0）**：直接可用，三个 mask 专属段全保持 None/0；
- **causal（1）**：`num_contexts` / `num_targets` 独立可选（None/未提供 = 全 0，即
  history-only / 无 target 封顶），提供时必须 1D、长度 = batch、int32/int64；
  `target_group_size` <= 0 = 无 target。权重闭式与语义见算子
  `op_kernel_aicpu/mask/causal_mask_predictor.h` 及其 `causal_mask_predictor_design.md`；
- **arbitrary（2）**：需成对提供 `full_cnt` / `mask_cnt`（int32、NPU 张量、
  numel = batch × maxBlkCntK，逻辑布局 [batch, maxBlkCntK]，语义见 hstu_v2 的
  `arbitrary_mask_sparse_info.md`）。

非当前模式的段必须保持 None/0（causal 段在非 causal 模式下拒收非空值，
full_cnt/mask_cnt 在非 arbitrary 模式下拒收非空值，均在绑定层先拦一道）。

> 注意：arbitrary 的分核权重 = `full_cnt + mask_cnt` 按行块查表（不含 head 维，
> 同一 batch 内所有 head 共用同一份掩码模式）；某行块两类计数均为 0 时权重为 0，
> 该批次的分核退化语义见算子 `DESIGN.md` §5.2，覆盖性不受影响。
> causal 的分核权重 = 该 K 行块需要计算的 Q 列块数 `|(A ∪ B) ∩ T|`
> （A=tril 后缀、B=context 前缀、T=target 封顶，闭式推导见设计文档 §5），
> 同样只影响切点、不影响覆盖性。

### 4.1 输出张量：预分配大小

- **dtype**：`int32`，一维；**device**：NPU。
- 由**本绑定自行分配**（算子把结果 in-place 写进这个输出张量）。预分配公式：

```text
sectionNumMax = batch * num_heads          # 每个 BN2 最多切成一个 section，故这是上界
elems         = 16 + sectionNumMax * 1728  # 1728 = 36(AIC)*16 + 72(AIV)*16
aligned       = ceil(elems / 4096) * 4096
```

- 真实 `sectionNum` 由算子按 L2 预算动态决定，通常远小于 `batch * num_heads`；
  尾部多出来的容量算子不会触碰（`ClearMetadata` 只清到实际 `sectionNum` 对应的长度）。

### 4.2 内存布局总览

逻辑布局（见 `op_kernel_aicpu/hstu_attn_metadata_backward.h`，与 `hstu_attn_metadata` 前向同构）：

```text
metadata[int32]:
┌────────────────── HEAD ──────────────────┐
│  16 × int32                               │  固定 1 条
├────────────────── FA ────────────────────┤
│  [sectionNum][36][16]                     │  每个 section × 每个 AIC 各 1 条
├────────────────── FD ────────────────────┤
│  [sectionNum][72][16]                     │  反向不使用，整段恒 0（仅占位保持同构）
└───────────────────────────────────────────┘
（其后可能还有 4096 元素对齐产生的 padding）
```

单条记录寻址（单位：int32 下标）：

```text
HEAD[k]         = metadata[k]
FA[sec][aic][k] = metadata[16 + sec*36*16 + aic*16 + k]
FD[sec][aiv][k] = metadata[16 + sectionNum*36*16 + sec*72*16 + aiv*16 + k]
```

### 4.3 HEAD 段（16 个 int32，只用前 4 个）

| 下标 | 字段 | 反向取值 |
| --- | --- | --- |
| 0 | `sectionNum` | 实际切出的 section 个数（≥ 1） |
| 1 | `isFd` | **恒 0**（反向不做 flash-decoding 归约） |
| 2 | `mBaseSize` | = `BLOCK_M`（K 轴行块大小），128 或 64 |
| 3 | `s2BaseSize` | = `BLOCK_N`（Q 轴列块大小），256 或 128 |
| 4..15 | （保留） | 0 |

> `mBaseSize` 是**硬约束**：设备侧 `MetadataRowBlockScheduler::Init` 会拿它与编译期行块比对，
> 不一致则该核 `blockCnt = 0` 静默空转、不报错。所以 `head_dim` 传错**不会报错，只会什么都不算**。

### 4.4 FA 段（每个 AIC 一条，16 个 int32，前 4 个有效）

格式：`[bn2Start, mStart, s2Start, bn2End, mEnd, s2End, fdWsIdx, 0…]`

| 下标 | 字段 | 反向取值 |
| --- | --- | --- |
| 0 | `bn2Start` | 本核区间起点 BN2 |
| 1 | `mStart` | 本核区间起点 K 行块序号（`(bn2Start, mStart)` 配成对） |
| 2 | `s2Start` | **恒 0**（保留位） |
| 3 | `bn2End` | 本核区间终点 BN2（半开区间右端） |
| 4 | `mEnd` | 本核区间终点 K 行块序号 |
| 5 | `s2End` | **恒 0**（保留位） |
| 6 | `fdWsIdx` | **恒 0**（保留位） |
| 7..15 | （保留） | 0 |

**两端点都必须写**：设备侧 `LoadSection` 读的正是这 4 个字段（`metadata_row_block_scheduler.hpp:205-216`），
再各自展平后相减得到本核块数：

```text
FlattenRowBlock(bn2, m) = Σ_{x<bn2} ceil(seqLenK(x / num_heads) / BLOCK_M) + m
rowBlockStart = FlattenRowBlock(bn2Start, mStart)
rowBlockEnd   = FlattenRowBlock(bn2End,   mEnd)
blockCnt      = rowBlockEnd > rowBlockStart ? rowBlockEnd - rowBlockStart : 0   # 0 ⇒ 本核空转
```

**接力规则**：section 内核 0 起点 = `(sec.bn2Begin, 0)`，其后每核起点 = 上一核终点，
末核终点 = `(sec.bn2End, 0)`。起点**逐 section 独立起算**，不像前向那样跨 section 连续累加
—— 反向是 persist 分核，同一个 section 由全部核重新分一遍排版。

未用到的核槽位（`coreId >= usedCoreNum`）保持整条全 0 ⇒ 展平出 `[0, 0)` ⇒ 长度 0 ⇒ 该核空转。

### 4.5 FD 段

反向不产出 FD 图，整段恒 0，设备侧也只镜像到 `kFaMEndIdx`、未声明 s2 / FD 下标。

---

## 五、调度语义

### 5.1 总体流水线

```text
Compute(ctx)
  ├─ ① PrepareInput       解析 cu_seqlens + num_heads/head_dim/mask_mode，算出 BLOCK_M/BLOCK_N 与每 batch 的块数
  │                       （mask 专属段入参不在这里解析，由 ③ 前的 MakeMaskPredictor 在工厂
  │                        case 内经 XxxMaskParams::FromCtx 解析，见 op_kernel_aicpu/mask/）
  ├─ ② SplitSections      沿 BN2 按 L2 预算切段（段内并行、段间串行）
  ├─ ③ SplitCoresPersist  在 section 内做 persist 分核，把区间端点落进工作区
  └─ ④ BuildMetadata      写 HEAD + FA（FD 保持全 0）
```

### 5.2 块档位（由 `head_dim` 唯一决定）

真值来自 hstu_v2 反向的编译期 `L1TileShape`（`TILE_K = head_dim > 128 ? 256 : 128`）：

| head_dim | BLOCK_M（K 轴行块 Rk） | BLOCK_N（Q 轴列块 Cq） | wire 字段 |
| --- | --- | --- | --- |
| 32 / 64 / 128 | 128 | 256 | `mBaseSize=128, s2BaseSize=256` |
| 256 | 64 | 128 | `mBaseSize=64, s2BaseSize=128` |

> 调用方约定 `head_dim = max(dimQK, dimGV)`（与 `test/hstu_v2/backend/ascend_fuse_backend.py` 一致）。

### 5.3 section 切分（②）

按 BN2 顺序贪心累计单头驻留量 `resident(b) = (seqLenQ[b]*3 + seqLenK[b]*2) * head_dim * 2B`
（反向驻留集 Q/K/V + grad + output 共 5 份：Q/grad/output 按 seqlen_q 计、K/V 按 seqlen_k 计），
超过 L2（96MB）就在当前 BN2 之前断开。三个短路直接产出单 section：

| 短路 | 条件 | 含义 |
| --- | --- | --- |
| A | `l2Byte == 0` | L2 预算未配置 |
| B | `maxBlkCntK <= 1` | K 轴只有一行块，切分无 L2 复用收益 |
| C | `maxHeadBytes <= L2 / aicCoreNum` | 全部核同时铺开也装得下共享 L2 |

> 短路 C 的 `/aicCoreNum` 不是「把 L2 切给各核」，而是把**共享**容量换算成每个并发消费者的份额：
> `maxHeadBytes <= L2 / N  ⇔  N × maxHeadBytes <= L2`。对照实现见 `hstu_attn_metadata` 的
> `section_stream_k_impl.h`（同样传整块 L2 再在内部除核数）。

### 5.4 persist 分核（③）

每个 section 独立分核：`quota = ceil(totalWork / aicCoreNum)`，按 `bn2 → kBlk` 顺序贪心累加，
达到配额且「后面还有块」即切一刀，最后一个核收尾到 section 末尾。

- 工作量权重（当前口径，**不考虑 mask**）：`w(b) = ceil(seqLenQ[b] / BLOCK_N)`，
  即「本 BN2 的列块数」（seqLenQ = 0 的 batch 权重为 0，无 Q 轴运算量）。例如 `seqLenQ = [256, 512]`、`BLOCK_N = 128` 时，
  落在 batch0 的 K 块权重为 2、落在 batch1 的权重为 4。
- **唯一硬性不变量是覆盖性**：各核区间并集 = 该 section 全部 K 行块、两两不重叠、
  `usedCoreNum <= aicCoreNum`。**空核合法** —— 块数少于核数时切出空核是正常结果，
  设备侧读到 `blockCnt == 0` 即空转，只要求「每个块都有核在算」。

---

## 六、校验

| 层次 | 位置 | 说明 |
| --- | --- | --- |
| host 单测（无需 NPU/CANN） | `cust_op/test/hstu_attn_metadata/host_ut_backward/` | 82 用例，`bash run_host_ut.sh` |
| torch 绑定用例 | `cust_op/test/hstu_attn_metadata/test_hstu_attn_metadata_backward.py` | 按设备侧读法独立回读 metadata，校验 HEAD / 覆盖性 / 空槽位 / FD 全 0 |
| 断言逻辑离线自检 | `cust_op/test/hstu_attn_metadata/selfcheck_metadata_assertions.py` | 不需要 NPU：用合成的良构 metadata 验证断言既有假阳性也有区分度 |
| C++ 上板冒烟 | `examples/test_aclnn_hstu_attn_metadata_backward.cpp` | `bash run.sh --stage=run`，同样校验覆盖性 |

```bash
pytest -sv cust_op/test/hstu_attn_metadata/test_hstu_attn_metadata_backward.py
# 改完断言先离线自检（无需 NPU），再上板跑正式用例
python3 cust_op/test/hstu_attn_metadata/selfcheck_metadata_assertions.py
```

> python 用例里**另起一套**布局常量与 `LoadSection` 解析器，刻意不复用算子头，
> 让写入侧与读取侧各持一份独立声明 —— 任何一侧改了 stride / 下标就会立刻对不上。
> 同样的道理，`examples/` 里的校验器也自己实现了一份 `FlattenRowBlock`。

---

## 七、与前向 `hstu_attn_metadata` 的差异

| 维度 | `hstu_attn_metadata`（前向） | `hstu_attn_metadata_backward`（反向） |
| --- | --- | --- |
| 入参 | 4 张量 + 11 个属性/字符串 | **2 张量 + 3 个 int 属性** |
| 调度轴 | Q 行（M = G×S1） | **K 行（行块沿 seqK）** |
| 块来源 | `AdjustSinnerAndSouter` 运行时决定（含 decode 分支） | **由 `head_dim` 查编译期真值表** |
| 分核 | 逐 section 枚举核数、no-FD/with-FD 两套方案择优 | **固定 persist 分核，不做 FD** |
| 起点接力 | 跨 section 连续累加（一张连续网格） | **逐 section 独立起算** |
| FD 段 / `isFd` | 可能非 0 | 恒 0 |

---

## 八、扩展点

1. **新增 mask 类型**只改 kernel 侧与 host 放行（不改算子签名的前提下）：
   `op_kernel_aicpu/mask/` 下实现对应预测器（含 XxxMaskParams 数据 + FromCtx 解析，同头文件内聚，
   参数不进 BackwardInput）+ 在 `mask_predictor_factory.h` 工厂对应 case 里解析并构造，
   再放开 `op_host/hstu_attn_metadata_backward_check.h` 的模式校验。完整依赖顺序见
   算子 `DESIGN.md` §7。（no-mask / causal / arbitrary 三类均已按此流程接入）
2. **扩展入参**才需要改签名，按「槽位分段固定」扩 wire（非当前 mask_mode 的段传
   nullptr 且 host 拒收），同步改四处：
   `op_host/op_api/aclnn_hstu_attn_metadata_backward.{h,cpp}`、`op_host/op_api/l0_*.cpp`／
   `op_host/hstu_attn_metadata_backward_check.h`、`op_kernel_aicpu/*_aicpu.h` 的 `InputIdx`/attr
   常量与 ①，最后才是本绑定的 schema。
   （arbitrary 的 full_cnt/mask_cnt：输入槽位 4/5 + 绑定两个可选参数；causal 的
   num_contexts/num_targets/target_group_size：输入槽位 2/3 + `target_group_size` 属性 +
   绑定两个可选参数与一个 int。均已按此流程接入。本绑定只暴露算子真实消费的参数 ——
   避免暴露算子并不读取的参数。）
3. **`kTypeByte` 目前硬编码为 2**（FP16）用于 ② 的 L2 估算；输入里拿不到 dtype，
   若下游主算子用 BF16/FP32，需要把 dtype 作为属性传进来。
