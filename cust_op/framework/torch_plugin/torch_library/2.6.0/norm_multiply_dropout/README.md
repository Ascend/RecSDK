# 使用PyTorch框架调用norm_multiply_dropout算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

### 算子调用示例

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
g_np = np.random.uniform(low=low, high=high, size=(dim1,))
b_np = np.random.uniform(low=low, high=high, size=(dim1,))
dy_np = np.random.uniform(low=low, high=high, size=(dim0, dim1))

x_fused = torch.tensor(x_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
u_fused = torch.tensor(u_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
g_fused = torch.tensor(g_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
b_fused = torch.tensor(b_np, dtype=dtype, device=npu_device).contiguous().requires_grad_(True)
y, mean, var = torch.ops.mxrec.norm_multiply_dropout(x_fused, u_fused, g_fused, b_fused, eps, dropout_ratio)
```

## 编译与部署

算子编译与部署请参考 [RecSDK/cust_op/README.md](../../../../../README.md) 中 "单算子使用说明" 章节：
- [算子编译](../../../../../README.md#算子编译)
- [算子适配层编译](../../../../../README.md#算子适配层编译)

> **提示**
> 以上示例仅展示基本用法，如需更全面的测试用例，请参考完整测试文件：  
> - [test](../../../../../test/norm_multiply_dropout/torch/test_norm_multiply_dropout.py)