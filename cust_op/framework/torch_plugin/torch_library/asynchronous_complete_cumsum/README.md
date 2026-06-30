# 使用PyTorch框架调用asynchronous_inclusive_cumsum/asynchronous_exclusive_cumsum/asynchronous_complete_cumsum算子

该样例基于PyTorch2.6.0、python3.11.0运行

## PyTorch框架对外接口原型

```python
torch.ops.fbgemm.asynchronous_inclusive_cumsum(Tensor offset) -> Tensor
torch.ops.fbgemm.asynchronous_exclusive_cumsum(Tensor offset) -> Tensor
torch.ops.fbgemm.asynchronous_complete_cumsum(Tensor offset) -> Tensor
或
torch.ops.mxrec.asynchronous_inclusive_cumsum(Tensor offset) -> Tensor
torch.ops.mxrec.asynchronous_exclusive_cumsum(Tensor offset) -> Tensor
torch.ops.mxrec.asynchronous_complete_cumsum(Tensor offset) -> Tensor
```

### 参数说明

| 名称            | 输入/输出 | 参数类型 |  数据类型  | 数据格式                                       | 范围             | 说明                          |
|---------------|-------|  ----  |  ----  |--------------------------------------------|----------------|-----------------------------|
| offset        | 输入    | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) | 长度:[1, 2^63-1) | 仅支持一维输入                     |
| cumsum_offset | 输出    | Tensor | int32/int64 | torch.tensor([value1, value2, value3 ...]) |                | 用户需自行控制总和不超过int32/int64数值范围 |

**算子分析**

三个算子的功能都是实现输入offset的异步并行累加（前缀和计算），区别在于`asynchronous_complete_cumsum`包含总和与起点0，
`asynchronous_inclusive_cumsum`只包含总和，不含起点0，`asynchronous_exclusive_cumsum`只包含起点0，不含总和。

## 示例

```python
算子输入 x：输入的offset tensor, eg: [1,5,6]

算子输出 y：
1）`asynchronous_complete_cumsum`: [0, 1, 6, 12]
2）`asynchronous_inclusive_cumsum`: [1, 6, 12]
3）`asynchronous_exclusive_cumsum`: [0, 1, 6]
```

## 运行算子样例

### 算子编译与部署

算子编译部署请参考[RecSDK/cust_op/README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

### PyTorch编译

PyTorch框架适配层编译请参考[RecSDK/cust_op/README.md](../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

### 算子调用示例，以下以pytest方式调用为例

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


def get_inclusive_result(t_in):
    return torch.ops.fbgemm.asynchronous_inclusive_cumsum(t_in)


def get_inclusive_ops_result(t_in, is_mxrec):
    if is_mxrec:
        return torch.ops.mxrec.asynchronous_inclusive_cumsum(t_in).cpu()
    else:
        return torch.ops.fbgemm.asynchronous_inclusive_cumsum(t_in).cpu()


def get_exclusive_result(t_in):
    return torch.ops.fbgemm.asynchronous_exclusive_cumsum(t_in)


def get_exclusive_ops_result(t_in, is_mxrec):
    if is_mxrec:
        return torch.ops.mxrec.asynchronous_exclusive_cumsum(t_in).cpu()
    else:
        return torch.ops.fbgemm.asynchronous_exclusive_cumsum(t_in).cpu()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("length", [1, 10, 100, 1000, 1024, 10000])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_asynchronous_complete_cumsum(dtype, device, length, is_mxrec):
    t_int = torch.randint(0, 100, (length,), dtype=dtype)
    golden = get_result(t_int.to(device))
    result = get_ops_result(t_int.to(device), is_mxrec)
    assert torch.allclose(result, golden)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("length", [1, 10, 100, 1000, 1024, 10000])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_asynchronous_inclusive_cumsum(dtype, device, length, is_mxrec):
    t_int = torch.randint(0, 100, (length,), dtype=dtype)
    golden = get_inclusive_result(t_int.to(device))
    result = get_inclusive_ops_result(t_int.to(device), is_mxrec)
    assert torch.equal(result, golden)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("device", ["npu:0"])
@pytest.mark.parametrize("length", [1, 10, 100, 1000, 1024, 10000])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_asynchronous_exclusive_cumsum(dtype, device, length, is_mxrec):
    t_int = torch.randint(0, 100, (length,), dtype=dtype)
    golden = get_exclusive_result(t_int.to(device))
    result = get_exclusive_ops_result(t_int.to(device), is_mxrec)
    assert torch.equal(result, golden)
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[`RecSDK/cust_op/test/asynchronous_complete_cumsum_test/torch/test_asynchronous_complete_cumsum.py`](../../../../test/asynchronous_complete_cumsum_test/torch/test_asynchronous_complete_cumsum.py)。
