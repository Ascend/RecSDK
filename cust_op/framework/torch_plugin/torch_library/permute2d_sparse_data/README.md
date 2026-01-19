**使用pytorch框架调用方式调用permute_2D_sparse_data/permute_sparse_data/permute_2D_sparse_data_input1d算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.fbgemm.permute_2D_sparse_data(Tensor permute, 
                                        Tensor lengths, 
                                        Tensor values,
                                        Tensor? weights=None,
                                        SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)

torch.ops.fbgemm.permute_sparse_data(Tensor permute, 
                                     Tensor lengths, 
                                     Tensor values,
                                     Tensor? weights=None,
                                     SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)
                               
torch.ops.fbgemm.permute_2D_sparse_data_input1d(Tensor permute, 
                                                Tensor lengths, 
                                                Tensor values,
                                                int stride,
                                                Tensor? weights=None,
                                                SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)

torch.ops.mxrec.permute_2D_sparse_data(Tensor permute,
                                       Tensor lengths, 
                                       Tensor values,
                                       Tensor? weights=None,
                                       SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)

torch.ops.mxrec.permute_sparse_data(Tensor permute,
                                    Tensor lengths, 
                                    Tensor values,
                                    Tensor? weights=None,
                                    SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)

torch.ops.mxrec.permute_2D_sparse_data_input1d(Tensor permute,
                                               Tensor lengths, 
                                               Tensor values,
                                               int stride,
                                               Tensor? weights=None,
                                               SymInt? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)
```

#### 参数说明
|  名称  | 输入/输出  | 参数类型    | 数据类型       | 数据格式                                            | 范围                  |
|  ---- |--------|---------|------------|-------------------------------------------------|---------------------|
|  permute | 输入     | Tensor  | int32      | [indices]                                       | permute中的每个值均满足: >= 0 且 < `lengths.shape[0]` |
|  lengths | 输入     | Tensor  | int32/int64 | [ [lengths], [lengths],... ]                    |           
|  values | 输入     | Tensor  | int32/int64/fp32 | [values]                                        | values的长度等于`lengths.sum()` | 
|  stride | 输入(当调用permute_2D_sparse_data_input1d需传入) | Scalar  | int64       | stride                                       | stride > 0 |
|  weights | 输入(可选) | Tensor  | fp32       | [weights]                                       | weight的长度等于`lengths.sum()` |
|  permuted_lengths_sum | 输入(可选) | SymInt  | int        | NA                                              |        (0, std::numeric_limits<int>::max()]      |
|  permuted_lengths | 输出     | Tensor  | int32/int64   | [ [permuted_lengths], [permuted_lengths], ... ] |                     |
|  permuted_values | 输出     | Tensor  | int32/int64/fp32   | [permuted_values]                               |                     |
|  permuted_weights | 输出     | Tensor  |  fp32  | [permuted_weights]                              |       |


说明：
1. 指定permuted_lengths_sum时，permuted_values/permuted_weights长度为permuted_lengths_sum，请用户自行保证数值正确; 未指定permuted_lengths_sum时，算子将计算得到permuted_lengths_sum

2. 当调用permute_2D_sparse_data_input1d算子时，需传入stride参数，且满足lengths.numel()能被stride整除。

3. 该算子实现依赖asynchronous_complete_cumsum算子，需先安装asynchronous_complete_cumsum算子


当调用permute_2D_sparse_data_input1d算子时，需传入stride参数，且满足lengths.numel()能被stride整除。

### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例
调用permute2d_sparse_data算子示例
```python
import itertools
import random
import sysconfig

import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

DEVICE = "npu:0"
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

PTYPE = [np.int32]
LTYPE = [np.int64, np.int32]
VTYPE = [np.int64, np.int32, np.float32]
WTYPE = [None, np.float32]
TYPE_LIST = list(itertools.product(PTYPE, LTYPE, VTYPE, WTYPE))

# lengths shape为[1 ~ (2T - 1), B]
# extra_t用于测试permute和lengths不等长的情况，lengths[T + extra_T, B]
T = np.random.randint(2, 30, 4)
EXTRA_T = [1, 0, -1]
B = [2048, 20480, 204800]
SHAPE_LIST = list(itertools.product(T, EXTRA_T, B))


def get_result(tensors: dict, device: str = 'cpu', is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}

    if device and device.startswith('npu'):
        torch.npu.set_device(device)
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if is_mxrec:
        results = torch.ops.mxrec.permute_2D_sparse_data(**tensors)
    else:
        results = torch.ops.fbgemm.permute_2D_sparse_data(**tensors)
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute2d_sparse_data(types, shapes, enable_permuted_sum, is_mxrec):
    """
    Params:
        permute: (T) dtype=int32
        lenghts: (T + T', B) dtype=ltype
                 L = lengths[:T].sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
    """
    ptype, ltype, vtype, wtype = types
    t, extra_t, b = shapes
    extra_t = random.randint(1, t - 1) * extra_t

    permute = np.random.choice(t + extra_t, t).astype(dtype=np.int32)
    lengths = np.ones((t + extra_t, b), dtype=ltype)
    values = np.arange(0, (t + extra_t) * b, dtype=vtype)
    weights = np.arange(0, (t + extra_t) * b, dtype=wtype) if wtype else None
    permuted_lengths_sum = lengths[permute].sum() if enable_permuted_sum else None
    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)
```

调用permute2d_sparse_data_input1d示例
```python
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("shapes", SHAPE_LIST)
@pytest.mark.parametrize("enable_permuted_sum", [True, False])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_permute2d_sparse_data_input1d(types, shapes, enable_permuted_sum, is_mxrec):
    """
    Params:
        permute: (T) dtype=int32
        lengths: (T * B) dtype=ltype (1D flattened tensor)
                 L = lengths.sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
        stride: int64_t = B (batch size, used to reshape 1D lengths to 2D [T, B] for internal 2D permutation;
                must divide lengths.size(0) evenly, e.g., lengths.size(0) % stride == 0)
    """
    ptype, ltype, vtype, wtype = types
    t, extra_t, b = shapes
    extra_t = random.randint(1, t - 1) * extra_t if extra_t > 0 else extra_t  # Consistent randomization

    permute = np.random.choice(t + extra_t, t).astype(dtype=np.int32)
    lengths_2d = np.random.randint(1, 10, size=(t + extra_t, b), dtype=ltype)
    lengths = lengths_2d.flatten()
    total_length = int(lengths.sum())
    values = np.arange(0, total_length, dtype=vtype)
    weights = np.arange(0, total_length, dtype=wtype) if wtype else None
    permuted_lengths_sum = lengths_2d[permute].sum() if enable_permuted_sum else None
    params = {
        'permute': permute,
        'lengths': lengths,
        'values': values,
        'stride': b,
        'weights': weights,
        'permuted_lengths_sum': permuted_lengths_sum
    }

    golden = get_result(params)
    result = get_result(params, DEVICE, is_mxrec)

    for gt, pred in zip(golden, result):
        assert type(gt) is type(pred)
        if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
            assert torch.allclose(gt, pred, atol=1e-5)
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[`RecSDK/cust_op/test/permute2d_sparse_data_test/torch/test_permute2d_sparse_data.py`](../../../../test/permute2d_sparse_data_test/torch/test_permute2d_sparse_data.py)。