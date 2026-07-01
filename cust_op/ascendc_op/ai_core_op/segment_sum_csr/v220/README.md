# `SegmentSumCsr`

本算子仅支持 NPU 调用。

## 目录结构

```text
segment_sum_csr
|-- segment_sum_csr.cpp
|-- README.md
|-- v220/
|   |-- segment_sum_csr.json
|   |-- op_host/
|   |-- op_kernel/
|   |-- run.sh
|-- c310/
|   |-- run.sh
```

## 硬件支持情况

| 实现目录    | 典型硬件               |
|---------|--------------------|
| `v220/` | Atlas A2 / A3 训练系列 |
| `c310/` | Atlas A5 训练系列      |

## 接口定义

```python
torch.ops.mxrec.segment_sum_csr(
    int batch_size,
    Tensor csr_seg,
    Tensor values,
) -> Tensor y
```

## 功能说明

根据 `batch_size` 和 `csr_seg` 对 `values` 中各个分段求和。

`csr_seg` 为 CSR 格式的分段偏移数组：`csr_seg[i] * batch_size` 到 `csr_seg[i+1] * batch_size`（不含）之间的 `values` 元素属于第 `i` 段，对该段内所有元素求和得到输出 `y[i]`。

`batch_size` 定义了每一行的长度，相当于先将 `values` reshape 为 `[-1, batch_size]` 的二维张量，再按段求和。

### 仿真/伪代码

```python
def segment_sum_csr(batch_size, csr_seg, values):
    segment_nums = len(csr_seg) - 1
    y = torch.empty(segment_nums, dtype=values.dtype)
    for i in range(segment_nums):
        start = csr_seg[i] * batch_size
        end = csr_seg[i + 1] * batch_size
        y[i] = values[start:end].sum()
    return y
```

## 参数说明

| 名称 | 输入/输出 | 类型 | 数据格式/形状 | 说明 |
| --- | --- | --- | --- | --- |
| batch_size | 属性 | int64 | NA | batch 大小，要求 `values.size(0) % batch_size == 0`；允许 `batch_size = 0`（空 tensor） |
| csr_seg | 输入 | Tensor[int32/int64] | `[segment_nums + 1]` | CSR 分段偏移数组，单调递增，首元素为 0 |
| values | 输入 | Tensor[float32/float16/bfloat16/int32/int64] | `[N]` | 非零值数组，1D；整数类型在 NPU 侧自动转 FP32 计算后回 cast |
| y | 输出 | Tensor | `[segment_nums]` | 每段求和结果，类型与 `values` 一致 |

### 参数约束

- `csr_seg.dim() == 1`，`values.dim() == 1`
- `csr_seg` 必须单调递增，且 `csr_seg[0] == 0`
- `batch_size != 0` 时，`values.size(0) % batch_size == 0`
- `values` 为空 tensor（`values.numel() == 0`）时允许，输出为空 tensor
- `values` 为 int32/int64 时，NPU 内部先 cast 到 FP32 求和，再 cast 回整数类型；FP32 可精确表示 int32 全范围及绝对值 ≤2²⁴ 的 int64

## 调用示例

```python
import torch
import fbgemm_gpu  # noqa: F401
import fbgemm_ascend  # noqa: F401

torch.npu.set_device("npu:0")

# CSR 分段偏移：3 段，分别包含 2、3、1 个元素
csr_seg = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device="npu:0")
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32, device="npu:0")

batch_size = 1
output = torch.ops.mxrec.segment_sum_csr(batch_size, csr_seg, values)

# output = [3.0, 12.0, 6.0]
expected = torch.tensor([3.0, 12.0, 6.0], dtype=torch.float32)
torch.testing.assert_close(output.cpu(), expected)
```

大 shape 示例：

```python
segment_nums = 100
csr_seg = torch.arange(0, segment_nums + 1, dtype=torch.int32, device="npu:0") * 10
values = torch.randn(segment_nums * 10, dtype=torch.float32, device="npu:0")

batch_size = 1
output = torch.ops.mxrec.segment_sum_csr(batch_size, csr_seg, values)

assert output.shape == (segment_nums,)
assert output.dtype == values.dtype
```

## 编译与测试

- 算子编译请参考[RecSDK/cust_op/README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。
- 测试示例参考：`cust_op/test/segment_sum_csr_test/torch/test_segment_sum_csr.py`
