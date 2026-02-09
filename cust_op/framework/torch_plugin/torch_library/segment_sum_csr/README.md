**使用pytorch框架调用方式调用segment_sum_csr算子**

# Pytorch框架对外接口原型

```python
torch.ops.mxrec.segment_sum_csr(Tensor csr_seg, Tensor values, int batch_size) -> Tensor
```
# 参数说明

| 名称         | 输入/输出 | 数据类型    | 数据格式   | 范围                      | 说明                                                       |
|------------|-------|---------|--------|-------------------------|----------------------------------------------------------|
| csr_seg    | 输入    | int32/int64 | Tensor | | 各分段长度的完整累积和，分段长度是指每个段所包含的行数，csr_seg张量的形状为num_segments+1，其中num_segments为段的数量 |
| values     | 输入    | float32 | Tensor | | 需要分段求和的张量，长度是batch_size的倍数                               |
| batch_size | 输入    | int32/int64 | int    | | 每行包含的元素个数                                                |
| y          | 输出    | float32 | Tensor | | 输出                                                       |

# 运行算子样例

## 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明" - "算子编译"章节。

## Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明" - "算子适配层编译"章节，
此算子可在当前目录下执行bash build_ops.sh编译好动态库。

## 算子调用示例

以下示例为通过python3方式调用NPU侧算子：

# 使用方法

```python
import sysconfig
import pytest
import torch
import torch_npu
import fbgemm_gpu

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DEVICE = "npu:0"

def get_op(csr_seg: torch.Tensor, values: torch.Tensor, batch_size: int):
    y = torch.ops.mxrec.segment_sum_csr(csr_seg, values, batch_size)
    return y

def generate_random_segment_sum_data(device, csr_type, v_type):
    batch_size_val = torch.randint(1, 33, (1,), device=device, dtype=csr_type).item()
    num_segments = torch.randint(2, 101, (1,), device=device, dtype=csr_type).item()
    segment_lengths = torch.randint(1, 101, (num_segments,), device=device, dtype=csr_type)

    csr_seg = torch.cat([
        torch.tensor([0], device=device, dtype=csr_type),
        segment_lengths.cumsum(dim=0)
    ], dim=0)
    csr_seg = csr_seg.to(torch.int32)

    total_elements_per_batch = csr_seg[-1].item()
    total_values_length = batch_size_val * total_elements_per_batch

    if v_type.is_floating_point:
        values = torch.empty(total_values_length, device=device, dtype=v_type).uniform_(-5, 5)
    else:
        values = torch.randint(-5, 6, (total_values_length,), device=device, dtype=v_type)

    batch_size = torch.tensor([batch_size_val], device=device, dtype=csr_type)

    return batch_size, csr_seg, values

if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    batch_size, csr_seg, values = generate_random_segment_sum_data("cpu", torch.int32, torch.float32)
    segment_sum_npu = get_op(csr_seg.to(DEVICE), values.to(DEVICE), batch_size.item())
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[test_segment_sum_csr.py](../../../../test/segment_sum_csr_test/torch/test_segment_sum_csr.py)。