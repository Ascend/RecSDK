# 使用PyTorch框架调用offsets_range算子

该样例基于Pytorch2.6.0、python3.11.0运行

## offsets_range算子

### PyTorch框架对外接口原型

```python
torch.ops.mxrec.offsets_range(offsets, range_size) -> Tensor
```

### 参数说明

| 名称 | 输入/输出 | 数据类型 | 数据格式 | 范围 | 说明 |
|------|---|---|---|---|---|
| offsets | 输入 | int32/int64 | [dim0] | 一维，dim0∈[1,2^17] | 分段起始位置，需在 NPU 上 |
| range_size | 输入（属性） | int | int | rangeSize∈[1,2^32] | 属性输入，表示输出长度 |
| result | 输出 | int32/int64 | [range_size] | 一维，长度为 `range_size` | 每个分段内的局部下标，dtype 与 `offsets` 一致 |

约束说明：
- `offsets` 必须为 1D 且非空。
- `offsets` 必须满足非递减、`offsets[0] = 0`、`offsets[-1] <= range_size`。
- 允许空分段（即 `offsets[i] == offsets[i+1]`）。

算子逻辑：
- 第 `i` 段区间定义为 `[offsets[i], offsets[i+1])`，最后一段为 `[offsets[-1], range_size)`。
- 在每段内填充 `0, 1, 2, ...`。

示例：

```python
offsets = [0, 2, 5, 5]
range_size = 7
result = [0, 1, 0, 1, 2, 0, 1]
```

### 编译与部署

算子编译与部署请参考 [RecSDK\cust_op\README.md](../../../../README.md) 中 "单算子使用说明" 章节。

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_gpu

DEVICE = "npu:0"
torch.npu.config.allow_internal_format = False

# 加载算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 准备输入
offsets = torch.tensor([0, 2, 5, 5], dtype=torch.int32, device=DEVICE)
range_size = 7

# 执行算子
result = torch.ops.mxrec.offsets_range(offsets, range_size)
print(result.cpu())  # tensor([0, 1, 0, 1, 2, 0, 1], dtype=torch.int32)
```

> **提示**
> 上述用例为通用场景执行，更详细精度、多场景测试用例请参考：
> - [`RecSDK/cust_op/test/offsets_range_test/torch/test_offsets_range.py`](../../../../test/offsets_range_test/torch/test_offsets_range.py)
> - [`RecSDK/cust_op/test/offsets_range_test/torch/special_test_offsets_range.py`](../../../../test/offsets_range_test/torch/special_test_offsets_range.py)
