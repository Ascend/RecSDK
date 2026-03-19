**说明**

本算子仅支持NPU调用。

## 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A5系列产品     | 是  |
| Atlas A2训练系列产品  | 否  |
| Atlas A3训练系列产品  | 否  |


## init_address_lookup算子目录层级
```shell
-- init_address_lookup
   |-- c310
      |-- op_host                      # 算子host侧实现
      |-- op_kernel                    # 算子kernel侧实现
      |-- init_address_lookup.json     # 算子原型配置
      |-- README.md                    # 算子说明文档
      |-- run.sh                       # 算子编译部署脚本
```

## 功能

初始化地址查找表，用于训练中嵌入剪枝（In-training Embedding Pruning）。该算子为每个嵌入表建立初始的"恒等映射"地址关系：

- 在 `buffer_offsets[t]` 到 `buffer_offsets[t+1]` 范围内
- 如果行索引 `r < emb_sizes[t]`，则映射到自身（`address_lookups[idx] = r`）
- 如果行索引 `r >= emb_sizes[t]`，则映射到0（`address_lookups[idx] = 0`）

用于处理缓冲区大小可能大于实际嵌入表大小的情况。

## 算子实现原理

算子工作原理说明：

1. 输入 `buffer_offsets` 定义了每个嵌入表在连续内存中的起始位置，必须（非严格）递增
2. 输入 `emb_sizes` 定义了每个嵌入表的实际逻辑行数
3. 对于每个表内的位置，如果在有效范围内（< emb_sizes），则映射到自身行号；否则映射到0

计算公式：
对于表 `t` 中的第 `r` 行（`0 <= r < rows_t`），其在全局 `address_lookups` 数组中的索引为 `idx = buffer_offsets[t] + r`。
```
address_lookups[idx] = r      (若 r < emb_sizes[t])
address_lookups[idx] = 0      (若 r >= emb_sizes[t])
```

例如：
```python
buffer_offsets = [0, 5, 9]  # 表0 行数 = 5，表1 行数 = 4
emb_sizes = [3, 4]          # 表0 有效行 0,1,2；表1 有效行 0..3

# 处理后：
# 表0 段（长度 5）变为 [0, 1, 2, 0, 0]（行 0..2 保留自身，行 3、4 指向占位 0）
# 表1 段（长度 4）变为 [0, 1, 2, 3]（全部保留自身）
# 扁平化 address_lookups 的最终内容为：
# [0, 1, 2, 0, 0, 0, 1, 2, 3]
```

## 算子输入与输出

|  名称  |  输入/输出  |  数据类型  |  数据格式  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |
|  buffer_offsets | 输入 | int64 | [num_tables + 1] | CSR格式的行偏移，定义每个嵌入表的起始索引 |
|  emb_sizes | 输入 | int64, int32 | [num_tables] | 每个嵌入表的逻辑行数（有效数据区大小） |
|  address_lookups | 输入 | int64, int32 | [total_rows] | 调用方预分配的地址查找表缓冲区，大小等于 buffer_offsets[-1] |
|  address_lookups_out | 输出 | int64, int32 | [total_rows] | 写入后的地址查找表（与输入 address_lookups 为同一块内存，原地操作） |

## 算子编译部署

编译算子：
```shell
cd cust_op/ascendc_op/ai_core_op/init_address_lookup/c310
bash run.sh
```

算子完整编译步骤请参考[RecSDK\cust_op\README.md](../../../../README.md)。
算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/init_address_lookup/README.md)。

## 相关算子

该算子通常与以下算子配合使用：
- `remap_indices_update_utils`：重映射输入索引到当前物理地址
- `prune_embedding_tables`：执行剪枝过程

