# 使用PyTorch框架调用norm_multiply_dropout算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

## Pytorch框架对外接口原型

```python
norm_multiply_dropout(Tensor x, Tensor u, Tensor weight, Tensor bias, float eps, float dropout_ratio) -> Tensor[]
```

> 该算子接口支持PyTorch自动求导。

### 参数说明

| 名称            | 输入/输出  | 参数类型   | 数据类型                     | 数据格式   | 范围                            | 说明                                                    |
|---------------|--------|--------|--------------------------|--------|-------------------------------|-------------------------------------------------------|
| x             | 输入     | Tensor | float32/float16/bfloat16 | [B, C] | B∈[1,1000000]，C仅支持值为512,1024。 |                                                       |
| u             | 输入     | Tensor | float32/float16/bfloat16 | [B, C] | B∈[1,1000000]，C仅支持值为512,1024。 | 数据类型需与输入x数据类型保持一致。                                    |
| weight        | 输入     | Tensor | float32/float16/bfloat16 | [C]    | C仅支持值为512,1024。               | 数据类型需与输入x数据类型保持一致。                                    |
| bias          | 输入     | Tensor | float32/float16/bfloat16 | [C]    | C仅支持值为512,1024。               | 数据类型需与输入x数据类型保持一致。                                    |
| eps           | 输入(属性) | float  | float                    |        | 数值范围:[1e-10, 1e-4]            | 极小值，防止除0                                              |
| dropout_ratio | 输入(属性) | float  | float                    |        | 数值范围:[0.0, 1.0]               | 当dropout_ratio值<= 1e-10时，会当做dropout概率为0而不进行dropout处理。 |
| output        | 输出     | Tensor | float32/float16/bfloat16 | [B, C] | B∈[1,1000000]，C仅支持值为512,1024。 | 前向计算结果，输出数据类型与输入x数据类型一致。                              |

## 算子运行样例

```python
import sysconfig

import numpy as np
import torch
import torch_npu

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
np.random.seed(42)
low = -1.0
high = 1.0
dim0 = 262144
dim1 = 512
eps = 1e-5
dropout_ratio = 0.0
dtype = torch.bfloat16
npu_device = torch.device("npu")
x_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
u_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))
w_np = np.random.uniform(low=low, high=high, size=(dim1,))
b_np = np.random.uniform(low=low, high=high, size=(dim1,))
dy_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))

x_fused = torch.tensor(x_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
u_fused = torch.tensor(u_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
w_fused = torch.tensor(w_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
b_fused = torch.tensor(b_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
output = torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, w_fused, b_fused, eps, dropout_ratio)[0]
```

## 编译与部署

算子编译与部署请参考 [RecSDK/cust_op/README.md](../../../../README.md) 中 "单算子使用说明" 章节：

- [算子编译](../../../../README.md#算子编译)
- [算子适配层编译](../../../../README.md#算子适配层编译)

> **提示**
> 以上示例仅展示基本用法，如需更全面的测试用例，请参考完整测试文件：  
>
> - [test](../../../../test/norm_multiply_dropout/torch/test_norm_multiply_dropout.py)
