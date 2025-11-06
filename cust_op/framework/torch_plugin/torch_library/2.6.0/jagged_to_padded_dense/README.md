**使用pytorch框架调用方式调用jagged_to_padded_dense算子**

# Pytorch框架对外接口原型

```python
torch.ops.mxrec.jagged_to_padded_dense(Tensor values, Tensor[] offsets, int max_lengths, float padding_value) -> Tensor

torch.ops.fbgemm.jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value) -> Tensor
torch.ops.mxrec.jagged_to_padded_dense(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value) -> Tensor

torch.ops.mxrec.jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, int max_lengths, float padding_value) -> Tensor
torch.ops.mxrec.jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, int[] max_lengths, float padding_value) -> Tensor
torch.ops.mxrec.jagged_to_padded_dense_backward(Tensor grad, Tensor[] offsets, int total_L) -> Tensor

torch.ops.fbgemm.jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, int max_lengths, float padding_value) -> Tensor
```

> 注：<br>
> jagged_to_padded_dense_forward, jagged_to_padded_dense_backward为早期版本，未注册自动求导，不建议使用。<br>
> jagged_to_padded_dense和dense_to_jagged互为前反向算子，dense_to_jagged具体调用请参考[`dense_to_jagged`](../dense_to_jagged/README.md)算子介绍。

# 参数说明

| 名称            | 输入/输出   | 参数类型      | 数据类型          | 数据格式                                 | 范围           | 说明                                                                     |
|---------------|---------|-----------|---------------|--------------------------------------|--------------|------------------------------------------------------------------------|
| values        | 输入      | Tensor    | float32/int64 | [dim0, dim1]                         |              |                                                                        |
| offsets       | 输入      | Tensor[]  | int32/int64   |                                      | 数值必须从0开始依次递增 | list中tensor个数只能为1, 且tensor仅支持一维<br>  offsets内元素需用户自行保证合法性，否则可能导致算子执行失败 |
| max_lengths   | 输入(属性)  | int/int[] | int           |                                      |              | max_length的元素值需大于0。类型为数组时，长度只能为1                                       |
| padding_value | 输入(属性)  | float     | float         |                                      |              |
| jagged_dense  | 输出(返回值) | Tensor    | float32/int64 | [len(offsets) - 1, max_length, dim1] |              |                                                                        |

# 运行算子样例

## 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明" - "算子编译"章节。

## Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../../README.md)中"单算子使用说明" - "算子适配层编译"章节。

## 算子调用示例

以下示例为通过python3方式调用NPU侧算子：

```python
import sysconfig

import fbgemm_gpu
import numpy as np
import torch_npu
import torch

# 加载NPU自定义算子库
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
# 设置用的卡号
DEVICE = "npu:0"
torch_npu.npu.set_device(DEVICE)

# 假设values数据由5个tensor组成，每个tensor第一维为tensor_length[i], 第2维为40
tensor_length = [8, 6, 4, 8, 1]
first_dim_sum = sum(tensor_length)  # 数值为27
input_values = torch.tensor(np.random.randn(first_dim_sum, 40).astype(np.float32))  # shape[27, 40]
# offsets为每个tensor数据在values中的偏移，从0开始
input_offsets = np.cumsum(np.array(tensor_length))
input_offsets = torch.tensor(np.insert(input_offsets, 0, 0).astype(np.int64))  # shape[6]， 数据值为：[0, 8, 14, 18, 26, 27]

input_values = input_values.to("npu")
input_offsets = input_offsets.to("npu")

# max_lengths为10，padding_value为0.0  即将前面的5个tensor每个tensor的第一维填充到10，扩充元素使用0.0填充
result = torch.ops.fbgemm.jagged_to_padded_dense(values=input_values, offsets=[input_offsets], max_lengths=[10], padding_value=0.0)
print("result shape:", result.shape, ", result data:", result)  # result shape[5, 10, 40]
```

注：上述用例为通用场景执行，更详细精度、多场景测试用例请参考用例[test_dense_to_jagged.py](../../../../../test/jagged_to_padded_dense_test/torch/test_jagged_to_padded_dense.py)。