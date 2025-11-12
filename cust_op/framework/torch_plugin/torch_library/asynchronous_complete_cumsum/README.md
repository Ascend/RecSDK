**使用pytorch框架调用方式调用asynchronous_complete_cumsum算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.fbgemm.asynchronous_complete_cumsum(Tensor offset) -> Tensor
或
torch.ops.mxrec.asynchronous_complete_cumsum(Tensor offset) -> Tensor
```

#### 参数说明
| 名称            | 输入/输出 | 参数类型 |  数据类型  | 数据格式                                       | 范围             | 说明      |
|---------------|-------|  ----  |  ----  |--------------------------------------------|----------------|---------|
| offset        | 输入    | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^31-1) | 仅支持一维输入 |
| cumsum_offset | 输出    | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) |                |         |


### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例
```python
import sysconfig
import pytest
import torch
import torch_npu
import fbgemm_gpu

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")


def get_result(t_in):
    return torch.ops.fbgemm.asynchronous_complete_cumsum(t_in)


def get_ops_result(t_in, is_mxrec):
    if is_mxrec:
        return torch.ops.mxrec.asynchronous_complete_cumsum(t_in).cpu()
    else:
        return torch.ops.fbgemm.asynchronous_complete_cumsum(t_in).cpu()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0", "npu:5"])
@pytest.mark.parametrize("length", [1, 10, 100, 1000, 10000])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_asynchronous_complete_cumsum(dtype, device, length, is_mxrec):
    t_int = torch.randint(0, 100, (length,), dtype=dtype)
    golden = get_result(t_int)
    result = get_ops_result(t_int.to(device), is_mxrec)
    assert torch.allclose(result, golden)
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[`RecSDK/cust_op/test/asynchronous_complete_cumsum_test/torch/test_asynchronous_complete_cumsum.py`](../../../../test/asynchronous_complete_cumsum_test/torch/test_asynchronous_complete_cumsum.py)。