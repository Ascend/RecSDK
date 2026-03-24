**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持 |
| -------------------- |------|
| Atlas A2训练系列产品  | 是    |
| Atlas A3训练系列产品  | 是    |
| Atlas A5训练系列产品  | 是    |
| Atlas 推理系列产品    | 否    |

# offsets_range算子目录层级

```shell
offsets_range
|-- c310
    |-- run.sh                  # 算子编译部署脚本
|-- v220
    |-- op_host                 # 算子host侧实现
    |-- op_kernel               # 算子kernel侧实现
    |-- offsets_range.json      # 算子原型配置
    |-- README.md               # 算子说明文档
    |-- run.sh                  # 算子编译部署脚本
```

# 功能

根据输入分段起始位置 `offsets` 和输出长度 `rangeSize`，生成每个分段内的局部下标。

输出 `result` 的定义如下：
- 第 `i` 段区间：`[offsets[i], offsets[i+1])`（最后一段为 `[offsets[last], rangeSize)`）
- 在每段内填充 `0, 1, 2, ...`

# 算子实现原理

输入：

```python
offsets = [0, 2, 5, 5]  # 一维，表示4个分段的起始位置
rangeSize = 7            # 输出长度
```

分段区间为：
- 第0段: `[0, 2)` -> `[0, 1]`
- 第1段: `[2, 5)` -> `[0, 1, 2]`
- 第2段: `[5, 5)` -> `[]`（空段）
- 第3段: `[5, 7)` -> `[0, 1]`

输出：

```python
result = [0, 1, 0, 1, 2, 0, 1]
```

# 算子输入与输出
| 名称 | 输入/输出 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---|---|---|---|---|
| offsets | 输入 | int32/int64 | [dim0] | 一维，dim0∈[1,2^17] | 各分段起始位置 |
| rangeSize | 输入（属性） | int | NA | rangeSize∈[1,2^32] | 输出长度 |
| result | 输出 | int32/int64 | [rangeSize] | 一维，长度为 `rangeSize` | 每个分段内的局部下标 |

约束说明：
- `offsets` 必须非递减。
- `offsets[0]` 必须为 `0`。
- `offsets[-1] <= rangeSize`。
- 允许空分段（即 `offsets[i] == offsets[i+1]`）。
- 输出 `result` 的数据类型与 `offsets` 一致。
- `v220/run.sh` 默认构建目标仅支持int32，`c310/run.sh`默认构建目标支持int32/int64。

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/offsets_range/README.md)
