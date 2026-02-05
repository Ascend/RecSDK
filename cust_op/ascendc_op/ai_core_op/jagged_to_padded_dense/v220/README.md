**说明**

本算子仅支持NPU调用。

# 产品支持情况

| 硬件型号           | 是否支持 |
|----------------|------|
| Atlas A2训练系列产品 | 是    |
| Atlas A3训练系列产品 | 是    |
| Atlas 推理系列产品   | 是    |

# jagged_to_padded_dense算子目录层级

```shell
jagged_to_padded_dense
|-- v220
   |-- op_host                        # 算子host侧实现
   |-- op_kernel                      # 算子kernel侧实现
   |-- jagged_to_padded_dense.json    # 算子原型配置
   |-- README.md                      # 算子说明文档
   |-- run.sh                         # 算子编译部署脚本
```

# 功能

实现将jagged tensor转为padded dense tensor功能, 对应开源API: torch.ops.fbgemm.jagged_to_padded_dense

# 算子实现原理

根据offsets将values切分为多个tensor, 再使用padding_value将每个tensor填充至max_length长度。

算子逻辑伪代码如下：

```python
import numpy as np


def jagged_to_padded_dense(values, offsets, max_lengths, padding_value):
    offsets = offsets[0]
    out = np.full((offsets.shape[0] - 1, max_lengths[0], values.shape[1]), padding_value).astype(np.float32)
    for i in range(1, offsets.shape[0]):
        copy_len = offsets[i] - offsets[i - 1]
        out[i - 1][0:copy_len, :] = values[offsets[i - 1]: offsets[i]]
    return out.astype(np.float32)


# 假设values数据由5个tensor组成，每个tensor第一维为tensor_length[i], 第2维为40
tensor_length = [8, 6, 4, 8, 1]
first_dim_sum = sum(tensor_length)  # 数值为27
input_values = np.random.randn(first_dim_sum, 40).astype(np.float32)  # shape[27, 40]
# offsets为每个tensor数据在values中的偏移，从0开始
input_offsets = np.cumsum(np.array(tensor_length))
input_offsets = np.insert(input_offsets, 0, 0).astype(np.int64)  # shape[6]， 数据值为：[0, 8, 14, 18, 26, 27]

# max_lengths为10，padding_value为0.0  即将前面的5个tensor每个tensor的第一维填充到10，扩充元素使用0.0填充
result = jagged_to_padded_dense(values=input_values, offsets=[input_offsets], max_lengths=[10], padding_value=0.0)
# result shape[5, 10, 40]
```

# 算子输入与输出

| 名称            | 输入/输出   | 参数类型      | 数据类型          | 数据格式                                  | 范围           | 说明                                                                     |
|---------------|---------|-----------|---------------|---------------------------------------|--------------|------------------------------------------------------------------------|
| values        | 输入      | Tensor    | float32/float16/bfloat16/int32/int64 | [dim0, dim1]                          |              |                                                                        |
| offsets       | 输入      | Tensor[]  | int32/int64   |                                       | 数值必须从0开始依次递增 | list中tensor个数只能为1, 且tensor仅支持一维<br>  offsets内元素需用户自行保证合法性，否则可能导致算子执行失败 |
| max_lengths   | 输入(属性)  | int/int[] | int           |                                       |              | max_length的元素值需大于0。类型为数组时，长度只能为1                                       |
| padding_value_fp32 | 输入(属性)  | float     | float         |                                       |              |
| padding_value_int64 | 输入(属性)  | int64     | int64         |                                       |              |
| jagged_dense  | 输出(返回值) | Tensor    | float32/float16/bfloat16/int32/int64 | [len(offsets) - 1, max_lengths, dim1] |              |                                                                        |

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/jagged_to_padded_dense/README.md)
