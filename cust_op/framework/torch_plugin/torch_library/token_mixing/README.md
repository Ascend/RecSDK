# 使用PyTorch框架调用token_mixing算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

## token_mixing算子

### PyTorch框架对外接口原型

```python
torch.ops.mxrec.token_mixing(Tensor x, Tensor gamma, Tensor beta, float epsilon = 1e-7) -> Tensor
```

### 参数说明

| 名称      | 输入/输出 | 参数类型 | 数据类型         | 数据格式       | 范围         | 说明                                  |
|---------|------------|------|--------------|------------|------------|----------------------------------------|
| x       | 输入       | Tensor | float32 | [B, S, H] | `[]` | 三维张量的S,H维度需要相同的                |
| gamma   | 输入       | Tensor | float32 | [H] | `[]` | 仅支持一维张量              |
| beta    | 输入       | Tensor | float32 | [H] | `[]` | 仅支持一维张量              |
| epsilon | 输入  | float | float   |           |          | 极小数，防止除零，默认值为1e-7    |
| y       | 输出     | Tensor | float32 | [B, S, H] | NA         | 返回融合算子归一化结果张量           |

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu
import numpy as np
from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent /
                           "cust_op/framework/torch_plugin/torch_library/token_mixing/build"
                           "/libtoken_mixing.so"))

x = np.random.randn(10, 32, 32).astype(np.float32)
gamma = np.ones(32, dtype=np.float32)
beta = np.zeros(32, dtype=np.float32)

x_tensor = torch.from_numpy(x).to("npu:0")
gamma_tensor = torch.from_numpy(gamma).to("npu:0")
beta_tensor = torch.from_numpy(beta).to("npu:0")

y = torch.ops.mxrec.token_mixing(x_tensor, gamma_tensor, beta_tensor, 1e-7)

# y shape: torch.Size([10, 32, 32])
print("y shape:", y.shape)
```
