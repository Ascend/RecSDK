**使用pytorch框架调用方式调用dense_to_jagged算子**

该样例基于Pytorch2.6.0、python3.11.0运行

### Pytorch框架对外接口原型

```python
torch.ops.fbgemm.dense_to_jagged_forward(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> Tensor
torch.ops.mxrec.dense_to_jagged_forward(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> Tensor

torch.ops.fbgemm.dense_to_jagged(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> (Tensor, Tensor[])
torch.ops.mxrec.dense_to_jagged(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> (Tensor, Tensor[])

torch.ops.fbgemm.dense_to_jagged_backward(Tensor values, Tensor[] offsets, int max_lengths, float padding_value) -> Tensor
torch.ops.mxrec.dense_to_jagged_backward(Tensor values, Tensor[] offsets, int max_lengths, float padding_value) -> Tensor
```

注：dense_to_jagged_forward、dense_to_jagged接口均为dense_to_jagged算子调用接口。<br>
dense_to_jagged_backward接口为dense_to_jagged算子的反向算子jagged_to_padded_dense接口，具体调用请参考`jagged_to_padded_dense`算子调用。

#### 参数说明
|  名称  |  输入/输出  | 参数类型 |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |  ----  |
|  dense | 输入 | Tensor | bfloat16/float16/float32/int32/int64 | [dim0, dim1, dim2] | dim0 <= std::numeric_limits<int>::max() - 1 | 仅支持三维 |
|  offsets | 输入 | Tensor[] | int32/int64 | [dim0 + 1] | dim0 + 1 <= std::numeric_limits<int>::max()<br>数值必须从0开始依次递增 | 仅支持一维<br>offsets内元素需用户自行保证合法性，否则可能导致算子执行失败 |
|  total_L | 输入(可选) | SymInt | int | NA | 有值时必须等于offset[-1] | NA |
|  jagged_dense | 输出 | Tensor | bfloat16/float16/float32/int32/int64 | [jagged_dim0, dim2] | NA | dense_to_jagged_forward输出 |
|  (jagged_dense, offsets) | 输出 | Tuple | (bfloat16/float16/float32/int32/int64, int32/int64) | ([jagged_dim0, dim2], [dim0 + 1]) | NA | dense_to_jagged输出，offsets即为输入参数offsets |


### 运行算子样例

#### 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子编译"章节。

#### Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明"-"算子适配层编译"。

#### 算子调用示例,以下以pytest方式调用为例
```python
import itertools
import logging
import sysconfig

import pytest
import fbgemm_gpu
import numpy as np
import torch_npu
import torch

DEVICE = "npu:0"
logging.getLogger().setLevel(logging.INFO)
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

DENSE_DIM0 = [128, 40, 1000] # 测试不同batch大小
DENSE_DIM1 = [210, 1024]     # 固定特征维度1
DENSE_DIM2 = [1, 8, 1536]    # 固定特征维度2
DIM_LIST = list(itertools.product(DENSE_DIM0, DENSE_DIM1, DENSE_DIM2))

DENSE_DATATYPE = [torch.float32, torch.int64] # 测试不同数据类型
OFFSET_DATATYPE = [torch.int32, torch.int64]  # 偏移量数据类型
TYPE_LIST = list(itertools.product(DENSE_DATATYPE, OFFSET_DATATYPE))


@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("output_size_type", ["none", "exact"])  # 测试不同output_size场景
@pytest.mark.parametrize("is_mxrec", [True, False])
def test_dense_to_jagged(dims, types, output_size_type, is_mxrec):
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 生成随机输入数据
    denses = np.random.randn(dense_dim0, dense_dim1, dense_dim2).astype(np.float32)
    offsets = np.random.randint(0, dense_dim1, dense_dim0) # 生成随机偏移量

    # 计算实际的output_size
    actual_size = np.sum(offsets)

    # 根据测试类型设置output_size
    output_size = None
    if output_size_type == "exact":
        output_size = actual_size

    dense_datatype, offset_datatype = types
    dense_torch = torch.from_numpy(denses).to(dense_datatype).to(DEVICE)
    offsets_torch = torch.from_numpy(offsets).to(offset_datatype).to(DEVICE)

    # 计算累积偏移量
    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    # 执行核心操作：稠密张量→不规则张量
    if is_mxrec:
        jagged_embedding = torch.ops.mxrec.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]
    else:
        jagged_embedding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[`RecSDK/cust_op/test/dense_to_jagged/torch/test_dense_to_jagged.py`](../../../../../test/dense_to_jagged/torch/test_dense_to_jagged.py)。