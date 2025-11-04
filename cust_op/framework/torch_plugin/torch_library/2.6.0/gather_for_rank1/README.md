# 使用PyTorch框架调用gather_for_rank1和index_select_for_rank1_backward算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

## gather_for_rank1算子

### PyTorch框架对外接口原型

```python
torch.ops.fbgemm.gather_for_rank1(Tensor x, Tensor index) -> Tensor
torch.ops.mxrec.gather_for_rank1(Tensor x, Tensor index) -> Tensor
```

### 参数说明

| 名称      | 输入/输出 | 参数类型 | 数据类型         | 数据格式       | 范围         | 说明                                     |
|---------|--------|------|--------------|------------|------------|----------------------------------------|
| x       | 输入     | Tensor | float16/float32 | `[embed_dim]` | `[1, 20480]` | 仅支持一维张量                                |
| index   | 输入     | Tensor | int32/int64    | `[index_num]` | `[1, ]`      | 仅支持一维张量；index 中每个元素值不得超过 `x.size(0)` |
| y (返回值) | 输出     | Tensor | float16/float32 | `[index_num]` | NA         | 返回从 `x` 中按 `index` 索引选取的元素组成的张量           |

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 准备输入
weight = np.random.randn(129).astype(np.float32)
index = np.random.randint(129, size=(128 * 211 * 211,)).astype(np.int64)

weight_tensor = torch.from_numpy(weight).to("npu:0")
index_tensor = torch.from_numpy(index).to("npu:0")

# 执行算子
output = torch.ops.mxrec.gather_for_rank1(weight_tensor, index_tensor)
print(output.shape)  # torch.Size([5690624])
```

## index_select_for_rank1_backward算子

### PyTorch框架对外接口原型

```python
torch.ops.fbgemm.index_select_for_rank1_backward(Tensor grad_y, Tensor x, Tensor index) -> (Tensor, Tensor)
torch.ops.mxrec.index_select_for_rank1_backward(Tensor grad_y, Tensor x, Tensor index) -> (Tensor, Tensor)
```

### 参数说明

|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  grad_y | 输入 | Tensor | float32 | [index_num, ...] | NA | index select的反向grad |
|  x | 输入 | Tensor | float32 | [embed_dim] | NA | index select查询的tensor |
|  index | 输入 | Tensor | int32/int64 | [index_num, ...] | NA | index select查询的index，index内的值不得超过x第0维长度 |
|  grad_x | 输出 | Tensor | float32 | [embed_dim] | NA | x的梯度 |
|  grad_index | 输出 | Tensor | int32/int64 | [index_num, ...] | NA | index的梯度 |

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu

DEVICE = "npu:7"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

grad_input = torch.randn(128*211*211, dtype=torch.float32, device=DEVICE)
x = torch.empty(129, device=DEVICE)
index = torch.randint(129, size=(128*211*211,), dtype=torch.int64, device=DEVICE)

grad_x, _ = torch.ops.mxrec.index_select_for_rank1_backward(grad_input, x, index)
```

## 编译与部署

算子编译与部署请参考 [RecSDK\cust_op\README.md](../../../../../README.md) 中 "单算子使用说明" 章节：
- [算子编译](../../../../../README.md#1算子编译)
- [算子适配层编译](../../../../../README.md#2算子适配层编译)

> **提示**  
> 以上示例仅展示基本用法，如需更全面的精度测试与边界用例，请参考完整测试文件：  
> - [`RecSDK/cust_op/test/gather_for_rank1_test/torch/test_gather_for_rank1.py`](../../../../../test/gather_for_rank1_test/torch/test_gather_for_rank1.py)
> - [`RecSDK/cust_op/test/gather_for_rank1_test/torch/test_index_select_for_rank1_backward.py`](../../../../../test/gather_for_rank1_test/torch/test_index_select_for_rank1_backward.py)