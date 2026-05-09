# GenPositionIdsReverseV2 算子及样例说明

本算子仅支持 NPU 调用。算子名为 `GenPositionIdsReverseV2`，与 PyTorch 适配层中 `aclnnGenPositionIdsReverseV2` 对应。

## GenPositionIdsReverseV2 算子文件结构

```shell
├── gen_position_ids_reverse_v2.json    # 算子原型配置
├── op_host                               # Host 侧实现（Tiling / InferShape / 原型注册）
├── op_kernel                             # Kernel 侧实现（AICore SIMT）
├── README.md                             # 本文档
└── run.sh                                # 编译安装脚本
```

## Ascend C 参考设计

更多说明可参考 CANN 官方 Ascend C 算子开发手册 [Ascend C 算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## GenPositionIdsReverseV2 算子使用

1. 将 `gen_position_ids_reverse_v2/c310`（或工程内等价路径）上传到目标环境，在包含 `run.sh` 的目录下编译与部署。

默认编译安装 Atlas A5 训练系列产品（`AI_CORE_PROFILE=c310` 对应 ascend950）：

```shell
bash run.sh
```

指定 AI Core 类型：

```shell
bash run.sh ai_core-<soc_version>
```

> AI 处理器型号 `<soc_version>`：在服务器执行 `npu-smi info` 查询 `Chip Name`，实际配置值为 `Ascend` + Chip Name。

注：需先配置 CANN 相关环境变量后再编译安装。示例（版本号按现场安装路径替换）：

```shell
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/<cann_version>/bin/setenv.bash
export ASCEND_OPP_KERNEL_PATH=/usr/local/Ascend/ascend-toolkit/<cann_version>
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## gen_position_ids_reverse_v2 算子介绍

### 1. 算子分析

a) **功能**：在 jagged 布局下为一维展平缓冲写入 `position_ids`：第 `i` 条样本在 `[offsets[i], offsets[i]+seqlen[i])` 内，前 `rspos[i]` 个位置为递减正整数，其余为 0；与 CUDA 参考核 `gen_position_ids_reverse_v2` 语义一致。  
b) **输入**：`seqlen`、`seqlen_offsets`、`rspos` 均为 **int32**、**ND**、一维。  
c) **属性**：`batchSize`（必填，**int32/int64**），须与 `seqlen` 第 0 维长度一致；与适配层传入的 `batch_size` 对齐。  
d) **输出**：`position_ids`，int32，一维，逻辑长度等于 `seqlen_offsets[batchSize]`（与对 `seqlen` 求和一致，数据合法时等价）。  
e) **约束简述**：

* 机型：Atlas A5（ascend950）等与本工程 `op_host` 中 `AddConfig` 一致的产品；
* CANN：建议 8.3.RC1 及之后版本（CANN 9.x 上 `InferShape` 中 `gert::Tensor::GetData` 需使用模板形式 `GetData<int32_t>()`）；
* `interleaved_action` / `with_ctx` 仅在 PyTorch 适配层侧拦截，本算子 IR 不包含这两项。

### 2. Host 侧实现（`op_host`）

源码：`gen_position_ids_reverse_v2.cpp`、`gen_position_ids_reverse_v2_tiling.h`。

a) **TilingFunc**

* 校验输入 dtype/shape；从 `GetAttrs()` 读取 `batchSize`，并与 `seqlen` dim0 对齐；
* 校验 `seqlen_offsets.length == batchSize + 1`、`rspos.length == batchSize`，`batchSize >= 0`；
* 按 batch 划分 logical block，计算 `blocksPerCore`、`remainderBlocks`，写入 `GenPositionIdsReverseV2TilingData`（含 `batchSize` 等）。

b) **InferShape**

* 根据 `seqlen` 形状与 `seqlen_offsets[batch]+1` 关系检查 batch；在具备常量数据时通过 `GetInputTensor(SEQLEN_INDEX)->GetData<int32_t>()` 对 `seqlen` 逐元素求和得到输出一维长度（与图编译阶段能力相关）。

c) **InferDataType**

* 输出固定为 `DT_INT32`。

d) **OpDef**

* 注册三个输入、`position_ids` 输出及属性 `batchSize`；绑定 `InferShape`、`InferDataType` 与 `TilingFunc`；AICore 配置 `ascend950`。

### 3. Kernel 侧实现（`op_kernel`）

入口：`gen_position_ids_reverse_v2.cpp` 中 `extern "C" __global__ __aicore__ void gen_position_ids_reverse_v2`。  
核心逻辑：`gen_position_ids_reverse_v2_kernel.h` 内按核划分 batch，对每个 batch 调用 `AscendC::Simt::VF_CALL` 的 SIMT 矢量化函数，使用 `__gm__ int32_t*` 基址访问全局内存（与 CANN 9.x 上 `GlobalTensor::GetBasePtr` 缺失的用法兼容，与同目录兄弟算子 `gen_position_ids_with_timestamp` 传入 GM 指针方式一致）。

## 单算子测试

算子编译部署及 PyTorch 调用说明见适配层 [README.md](../../../../framework/torch_plugin/torch_library/gen_position_ids_reverse_v2/README.md)。
