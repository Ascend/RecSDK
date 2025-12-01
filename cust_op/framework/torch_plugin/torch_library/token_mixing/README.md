# 使用PyTorch框架调用tokenmixing算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

## token_mixing算子

### PyTorch框架对外接口原型

```python
torch.ops.mxrec.token_mixing(Tensor x, Tensor gamma, Tensor beta, float epsilon = 1e-7) -> (Tensor, Tensor, Tensor)
```

### 参数说明

| 名称      | 输入/输出 | 参数类型 | 数据类型         | 数据格式       | 范围         | 说明                                  |
|---------|--------|------|--------------|------------|------------|----------------------------------------|
| x       | 输入     | Tensor | float16/float32/bfloat16 | [B, S, H] | `[]` | 仅支持三维张量S,H维度一致                |
| gamma   | 输入     | Tensor | float16/float32/bfloat16 | [H] | `[]` | 仅支持一维张量              |
| beta    | 输入     | Tensor | float16/float32/bfloat16 | [H] | `[]` | 仅支持一维张量              |
| epsilon | 输入(属性)  | float | float   |           |          | 小整数，防止除零，默认值为1e-7    |
| y (返回值) | 输出     | Tensor | float16/float32/bfloat16 | `[B, S, H]` | NA         | 返回融合算子归一化结果张量           |
| mean (返回值) | 输出     | Tensor | float16/float32/bfloat16 | `[B, S]` | NA         | 返回融合算子归一化均值结果张量       |
| rstd (返回值) | 输出     | Tensor | float16/float32/bfloat16 | `[B, S]` | NA         | 返回融合算子归一化标准差结果张量     |

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu
import numpy as np

x = np.random.randn(10, 32, 32).astype(np.float32)
gamma = np.ones(32, dtype=np.float32)
beta = np.zeros(32, dtype=np.float32)

x_tensor = torch.from_nump(x).to("npu:0")
gamma_tensor = torch.from_nump(gamma).to("npu:0")
beta_tensor = torch.from_nump(beta).to("npu:0")

y = torch.ops.mxrec.token_mixing(x_tensor, gamma, beta, 1e-7)
```
