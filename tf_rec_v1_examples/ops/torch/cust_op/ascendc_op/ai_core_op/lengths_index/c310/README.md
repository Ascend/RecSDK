# LengthsIndex 算子说明（c310）

## 1. 功能与场景

**功能**：根据前缀和形式的 **`offsets`**，构造长度 **`output_size`** 的一维 **`output`**，使得对每个输出下标 `k`，**`output[k]`** 表示 **`k` 落在第几条序列**（序列编号 `0 .. num_seq-1`）。

**典型场景**：变长序列展平后的「属于哪一条」掩码；推荐 / NLP 中 jagged 张量辅助索引等。

**特性摘要**

| 项 | 说明 |
|----|------|
| 芯片配置 | `ascend950`（见 `op_host/lengths_index.cpp` 中 `AddConfig`） |
| 数据类型 | `int32` / `int64` |
| 输入 | 1D `offsets` |
| 输出 | 1D `output`，shape 由属性 `output_size` 推导 |
| 并行 | 多 AICore + 块内 SIMT（`asc_vf_call`） |

---

## 2. 接口

算子原型见 **`lengths_index.json`**。摘要：

| 名称 | 角色 | 说明 |
|------|------|------|
| `offsets` | 输入 | 1D；长度须为 **`num_seq`**（exclusive）或 **`num_seq + 1`**（complete），单调非降 |
| `output` | 输出 | 1D，长度 = 属性 **`output_size`** |
| `output_size` | 属性 | 输出元素个数 |
| `num_seq` | 属性 | 序列条数 |

---

## 3. 语义与示例

对序列行号 **`rowIdx`**：

- **`rowStart = offsets[rowIdx]`**
- **`rowEnd = offsets[rowIdx + 1]`**（当 `rowIdx < num_seq - 1`）；**最后一行**用 **`rowEnd = output_size`**

在 **`[rowStart, rowEnd)`** 内所有下标写入 **`rowIdx`**。

**例**：`lengths = [2, 3, 1]`，`output_size = 6`，**exclusive** 前缀和（长度 **`num_seq = 3`**）：

```text
offsets = [0, 2, 5]
```

输出下标 `0..5` 对应序列编号：`[0, 0, 1, 1, 1, 2]`。

**complete** 前缀和时长度为 **`num_seq + 1`**，例如 **`[0, 2, 5, 6]`**，算法仍按上式取区间（最后一行右端为 `output_size`，与 `offsets[num_seq]` 一致时等价）。

---

## 4. 算法要点（与 CUDA 参考对照）

仓库内 **CUDA 参考**见同目录文件 **`lengths_index_cuda_kernel`**（含 `vector_size` 分档、`threads_per_block = 512` 等逻辑）。

**AscendC SIMT 实现**（`op_kernel/lengths_index_kernel.h`）将 CUDA 的二维 `threadIdx` 思路改为一维 **`tid`**：

- **`rowOffset = tid / vectorSize`**：当前 `logicalBlock` 内第几行（相对 **`startRow`**）
- **`colOffset = tid % vectorSize`**：行内「步长车道」
- **`rowIdx = startRow + rowOffset`**；若 **`rowIdx >= num_seq`** 则该线程不写（块尾空行槽）
- 循环 **`i = rowStart + colOffset; i < rowEnd; i += vectorSize`**，将 **`output[i] = rowIdx`**

SIMT 入口带 **`LAUNCH_BOUND`**，与仓库内其它 **`asc_vf_call`** 用法一致；`output` 使用 **`volatile`** 指针以利于 GM 写语义。

---

## 5. Host Tiling（`op_host/lengths_index.cpp`）

### 5.1 计算流程

1. 读取 **`output_size`**、**`num_seq`**，校验 **`offsets`** 维度、类型及 **`offsets` 长度 ∈ {num_seq, num_seq+1}**。
2. **`avgLen = output_size / num_seq`**（`num_seq > 0`），按阈值选定 **`vectorSize ∈ {2,4,8,16,32}`**。
3. **`rowsPerBlock = 512 / vectorSize`**（常量 512 为 Host 侧 `MAX_THREADS_PER_BLOCK`）。
4. **`totalBlocks = ceil(num_seq / rowsPerBlock)`**。
5. **`actualCoreNum = min(totalBlocks, GetCoreNumAiv())`**；**`blocksPerCore`**、**`remainderBlocks`** 将 **`totalBlocks`** 均分到各核。
6. **`SetBlockDim(actualCoreNum)`**，并把各字段写入 **`LengthsIndexTilingData`** 后 **`SaveToBuffer`**。

### 5.2 Tiling 字段

| 字段 | 含义 |
|------|------|
| `numSeq` / `outputSize` | 与属性一致 |
| `totalBlocks` | 逻辑块个数（按行切块） |
| `blocksPerCore` / `remainderBlocks` | 多核分块负载均衡 |
| `vectorSize` / `rowsPerBlock` | SIMT 步长与每块最大行槽数 |
| `ubCanUsed` | UB 可用量估算（随平台查询） |

---

## 6. Kernel（`op_kernel/`）

**入口**：`lengths_index(...)` 构造 **`LengthsIndexKernel<DTYPE_OFFSETS>`** 并 **`Compute()`**。

**`Compute`**：用 **`GetBlockIdx()`** 与 tiling 中的 **`blocksPerCore` / `remainderBlocks`** 得到本核负责的 **`logicalBlock`** 区间；对每个块 **`startRow = logicalBlock * rowsPerBlock`**，再 **`asc_vf_call<IndexRowsSimt<T>>(dim3{vectorSize*rowsPerBlock,1,1}, ...)`**（恒 **512** 线程）。

**`vectorSize` 与 `rowsPerBlock` 分档表**（与 Host 一致）

| `avgLen` | `vectorSize` | `rowsPerBlock` |
|----------|----------------|----------------|
| < 2 | 2 | 256 |
| < 4 | 4 | 128 |
| < 64 | 8 | 64 |
| < 128 | 16 | 32 |
| ≥ 128 | 32 | 16 |

---

## 7. 目录与编译

```text
lengths_index/c310/
├── lengths_index.json          # 算子原型
├── lengths_index_cuda_kernel   # CUDA 参考（命名与实现以仓库为准）
├── run.sh
├── op_host/
│   ├── lengths_index.cpp       # OpDef、Tiling、InferShape/InferDataType
│   └── lengths_index_tiling.h
└── op_kernel/
    ├── lengths_index.cpp       # kernel 入口
    └── lengths_index_kernel.h  # SIMT 与 Compute
```

```bash
cd lengths_index/c310
./run.sh
```

环境需已配置 **CANN / Ascend Toolkit**（路径以本机安装为准）。

---

## 8. 测试提示

PyTorch 侧封装见 **`torch_plugin/.../lengths_index`**；NPU 上需设备与算子已注册。功能用例可对比 **`output[k]`** 与手工按 **`offsets`** 区间展开是否一致。

---

## 9. 约束

1. **`offsets`** 必须为 **1D**，且长度 **`== num_seq`（exclusive）或 `== num_seq + 1`（complete）**。
2. **`num_seq ≥ 0`**，**`output_size ≥ 0`**。
3. 数据类型：**`int32` / `int64`**，输出与输入类型一致。
4. **`offsets`** 须满足前缀和语义（非降，且与 **`output_size`** 一致），否则结果无意义或存在越界风险。

---

## 10. 参考

- [Ascend C 算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)
- [CANN 安装](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/installation_guide/Atlas200DK_A2/01/index.html)
