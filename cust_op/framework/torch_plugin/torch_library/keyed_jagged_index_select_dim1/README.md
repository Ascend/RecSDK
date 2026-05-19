# 使用PyTorch框架调用方式调用keyed_jagged_index_select_dim1算子

该样例基于 PyTorch 2.6.0 和 Python 3.11.0 运行。

## Pytorch框架对外接口原型

```python
torch.ops.fbgemm.keyed_jagged_index_select_dim(Tensor values,
                                               Tensor lengths,
                                               Tensor offsets,
                                               Tensor indices,
                                               int batch_size,
                                               Tensor? weights=None,
                                               int? selected_lengths_sum=None) -> Tensor[]

torch.ops.mxrec.keyed_jagged_index_select_dim(Tensor values,
                                               Tensor lengths,
                                               Tensor offsets,
                                               Tensor indices,
                                               int batch_size,
                                               Tensor? weights=None,
                                               int? selected_lengths_sum=None) -> Tensor[]
```

### 参数说明

|  名称  | 输入/输出  | 参数类型    | 数据类型       | 数据格式                                            | 范围                  |
|  ---- |--------|---------|------------|-------------------------------------------------|---------------------|
|  values | 输入     | Tensor  | int32/int64/fp32/fp16 | [values]                                        | values的长度等于`lengths.sum()` | 
|  lengths | 输入     | Tensor  | int32/int64 | [lengths]                   |    |
|  offset | 输入     | Tensor  | int64      | [offset]                                       | 从0开始，为lengths元素的累加序列 |
|  indices | 输入     | Tensor  | int32/int64      | [indices]                                       | indices中的每个值均满足: >= 0 且 < `batch_size` |
|  weights | 输入(可选) | Tensor  | fp32/fp16       | [weights]                                       | weight的长度等于`lengths.sum()` |
|  batch_size | 输入 | Int  | `int`        | NA                                              |        dim 1 of KIT (0, std::numeric_limits\<int>::max()]      |
|  selected_lengths_sum | 输入(可选) | Int  | `int64`        | NA                                              |        [0, std::numeric_limits\<int64>::max()]      |
|  output | 输出     | Tensor[]  | Tensor[]  | [output] | 包含5个Tensor，outvalues,dtype和values相同， outlengths,dtype和lengths相同， outweight（如果weights存在，dtype和weight相同否则为None）， outputoffset,dtype和offsets相同）,savedDataT, dtype为int64 |

说明：

1. 指定selected_lengths_sum时，outvalues/outweight长度为selected_lengths_sum，请用户自行保证数值正确;未指定selected_lengths_sum时，算子将计算得到selected_lengths_sum

2. 该算子实现依赖asynchronous_complete_cumsum、select_dim1_to_permute、permute_2d_sparse_data算子，需先安装asynchronous_complete_cumsum、select_dim1_to_permute、permute_2d_sparse_data算子

3. v220版本，select_dim1_to_permute算子的indices dtype只支持int32，不支持int64 

## 运行算子样例

## 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

### 算子调用示例,以下以pytest方式调用为例

```python
import itertools
import random
import sysconfig

from pathlib import Path
import pytest
import torch
import torch_npu
import fbgemm_gpu
import numpy as np

DEVICE = "npu:0"
torch.npu.config.allow_internal_format = False
CURR_DIR = Path(__file__).resolve().parent
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

LENGTHS_TYPE = [np.int32]
VALUES_TYPE = [np.int64, np.int32, np.float32]
WEIGHTS_TYPE = [None, np.float32]
TYPE_LIST = list(itertools.product(LENGTHS_TYPE, VALUES_TYPE, WEIGHTS_TYPE))


def get_result(tensors: dict, is_mxrec: bool = False):
    tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in tensors.items()}
    torch.npu.set_device(DEVICE)
    tensors = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
    # also can use torch.ops.fbgemm.keyed_jagged_index_select_dim1
    results = torch.ops.mxrec.keyed_jagged_index_select_dim1(**tensors)
    torch_npu.npu.synchronize()
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]


# 采用fbgemm_gpu::permute_2D_sparse_data进行cpu设置验证
def get_golden(values, lengths, offsets, indices, batch_size, batch_num, weights, selected_lengths_sum):
    values = torch.from_numpy(values)
    lengths = torch.from_numpy(lengths)
    offsets = torch.from_numpy(offsets)
    indices = torch.from_numpy(indices)
    if weights is not None:
        weights = torch.from_numpy(weights)
    indiceslen = indices.size(0)
    permutelen = batch_num * indiceslen

    permute = torch.empty(permutelen, dtype=indices.dtype)
    for i in range(batch_num):
        for j in range(indiceslen):
            permute[i * indiceslen + j] = batch_size * i + indices[j].item()

    results = torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, values, weights, selected_lengths_sum)
    return [x.cpu() if isinstance(x, torch.Tensor) else x for x in results]

@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("batch_num", [1, 8, 64,100])
@pytest.mark.parametrize("batch_size", [2, 8, 64, 256])
@pytest.mark.parametrize("output_batch_size", [2, 8, 64, 256])
@pytest.mark.parametrize("enable_selected_lengths_sum", [False, True])
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_keyed_jagged_index_select_dim1(types, batch_num, batch_size, output_batch_size, enable_selected_lengths_sum, is_mxrec):
    """
    测试正常情况下的keyed_jagged_index_select_dim1算子功能
    Params:
        indices: (output_batch_size) dtype=int32
        lengths: (batch_size * batch_num) dtype=ltype
                L = lengths[:T].sum()
        values: (L) dtype=vtype
        weights: (L) dtype=fp32
        batch_size: int
        selected_lengths_sum: int
    """
    ltype, vtype, wtype = types

    indices = np.random.choice(batch_size, output_batch_size).astype(dtype=ltype)
    lengths = np.random.randint(2, 500, size=batch_size * batch_num, dtype=ltype)
    total_length = int(lengths.sum())
    cumulative_lengths = np.cumsum(lengths)
    offsets = np.zeros(len(lengths) + 1, dtype=ltype)
    offsets[1:] = cumulative_lengths

    is_float = vtype in [np.float32, np.float16]
    if is_float:
        values = np.random.rand(total_length).astype(dtype=vtype)
    else:
        values = np.random.randint(0, 2**16, (total_length,), dtype=vtype)
    weights = np.arange(0, total_length, dtype=wtype) if wtype else None

    permute = np.empty(batch_num * output_batch_size, dtype=ltype)
    for i in range(batch_num):
        for j in range(output_batch_size):
            permute[i * output_batch_size + j] = batch_size * i + indices[j]
    selected_lengths_sum = lengths[permute].sum() if enable_selected_lengths_sum else None
    params = {
        'values': values,
        'lengths': lengths,
        'offsets': offsets,
        'indices': indices,
        'batch_size': batch_size,
        'weights': weights,
        'selected_lengths_sum': selected_lengths_sum
    }

    golden = get_golden(values, lengths, offsets, indices, batch_size, batch_num, weights, selected_lengths_sum)
    result = get_result(params, is_mxrec)
    assert torch.allclose(golden[0], result[1], atol=1e-5)
    assert torch.allclose(golden[1], result[0], atol=1e-5)
    if weights is not None:
        assert torch.allclose(golden[2], result[2], atol=1e-5)
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[`RecSDK/cust_op/test/permute2d_sparse_data_test/torch/test_permute2d_sparse_data.py`](../../../../test/permute2d_sparse_data_test/torch/test_permute2d_sparse_data.py)。
