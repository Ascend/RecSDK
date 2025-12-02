**使用pytorch框架调用方式调用ln_mul算子**

# Pytorch框架对外接口原型

```python
torch.ops.mxrec.ln_mul(Tensor x, Tensor u, Tensor gamma, Tensor beta) -> Tensor
```
# 参数说明

| 名称     | 输入/输出 | 参数类型    | 数据类型          | 数据格式                                  | 范围           | 说明                                                                     |
|--------|-------|---------|---------------|---------------------------------------|--------------|------------------------------------------------------------------------|
| x      | 输入    | Tensor  | float32 | [A, R]                          |              |                                                                        |
| u      | 输入    | Tensor  | float32   |    [A, R]                                 |  |
| gamma  | 输入    | Tensor | float32           |      [R]                                 |              |                                        |
| beta   | 输入    | Tensor   | float32         |       [R]                                |              |
| output | 输出    | Tensor   | float32         |       [A, R]                              |              |

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
from pathlib import Path
import pytest
import torch
import torch_npu
import numpy as np

torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(str(CURR_DIR.parent.parent.parent /
                           "framework/torch_plugin/torch_library/ln_mul/build/libln_mul.so"))

def get_op(x: torch.Tensor, u: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    y = torch.ops.mxrec.ln_mul(x, u, gamma, beta)
    return y.cpu().numpy()

if __name__ == "__main__":
    torch.npu.set_device(0)
    a = 4
    r = 4
    dtype = torch.float32
    x = torch.randn(a, r, device="npu", dtype=dtype)
    u = torch.randn(a, r, device="npu", dtype=dtype)
    gamma = torch.randn(r, device="npu", dtype=dtype)
    beta = torch.randn(r, device="npu", dtype=dtype)
    y_op = get_op(x, u, gamma, beta)
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[test_ln_mul.py](../../../../test/ln_mul_test/torch/test_ln_mul.py)。