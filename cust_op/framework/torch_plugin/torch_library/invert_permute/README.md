# 使用PyTorch框架调用invert_permute算子

该算子当前支持两种软件版本配套：PyTorch 2.6.0和PyTorch2.7.1。详细配套说明见[RecSDK\cust_op\README.md](../../../../README.md)。

## invert_permute算子

### PyTorch框架对外接口原型

```python
torch.ops.fbgemm.invert_permute(Tensor permute) -> Tensor
torch.ops.mxrec.invert_permute(Tensor permute) -> Tensor
```

### 参数说明
| 名称             | 输入/输出 | 参数类型 |  数据类型  | 数据格式                                       | 范围             | 说明                                          |
|----------------|-------|  ----  |  ----  |--------------------------------------------|----------------|---------------------------------------------|
| permute        | 输入    | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^31-1) | 仅支持一维输入, 0 <= permute[i] <= length(permute) |
| invert_permute | 输出    | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) |                |                                             |

### 编译与部署

算子编译与部署请参考 [RecSDK\cust_op\README.md](../../../../README.md) 中 "单算子使用说明" 章节：

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_gpu

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 准备输入
tensor = torch.arange(0, length).to(dtype)
permute = tensor[torch.randperm(tensor.size(0))]

# 执行算子
output = torch.ops.fbgemm.invert_permute(permute)
```

> **提示**  
> 上述用例为通用场景执行，更详细精度、多场景测试用例，请参考完整测试文件：  
> - [`RecSDK/cust_op/test/invert_permute/torch/test_invert_permute.py`](../../../../test/invert_permute/torch/test_invert_permute.py)