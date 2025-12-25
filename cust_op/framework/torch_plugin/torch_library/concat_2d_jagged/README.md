# 使用PyTorch框架调用concat_2d_jagged和split_2d_jagged算子

该算子当前支持两种软件版本配套：PyTorch 2.6.0和PyTorch2.7.1。concat_2d_jagged 和 split_2d_jagged是互为前、反向的Tensor拼接和切分算子。详细配套说明见[RecSDK\cust_op\README.md](../../../../README.md)。

## concat_2d_jagged算子和split_2d_jagged算子

### PyTorch框架对外接口原型

```python
torch.ops.mxrec.concat_2d_jagged(max_seqlens, tensor_a, tensor_b, offset_a, offset_b) -> Tensor

torch.ops.mxrec.split_2d_jagged(tensor, max_seqlens, offsetA, offsetB) -> Tensor, Tensor
```
### 参数说明

# concat_2d_jagged算子输入与输出
| 名称            | 输入/输出  | 数据类型                     | 数据格式         | 范围                                                                      | 说明    |
|---------------|--------|--------------------------|--------------|-------------------------------------------------------------------------|-------|
| max_seqlens      | 输入     | int                      | int          | NA                                                                      | NA    |
| tensor_a      | 输入     | bfloat16/float16/float32 | [dim0, dim1] | dim1和dtype必须与tensor_b的一致                                                | 仅支持二维 |
| tensor_b      | 输入     | bfloat16/float16/float32 | [dim0, dim1] | dim1和dtype必须与tensor_a的一致                                                | 仅支持二维 |
| offset_a      | 输入     | int                      | [dim0]       | dim0的长度必须与offset_b的长度相等,长度范围[2, 1024]                                   | 仅支持一维 |
| offset_b      | 输入     | int                      | [dim0]       | dim0的长度必须与offset_a的长度相等,长度范围[2, 1024]                                                 | 仅支持一维 |
| isReplace      | 输入(可选) | bool                     | 默认值：False    | 当前只支持默认值                                                                | NA    |
| nPrefixFromRight   | 输入(可选) | int                      | 默认值：0        | 当前只支持默认值                                                                | NA    |
| output_tensor | 输出     | bfloat16/float16/float32 | [dim0, dim1] | dim0的长度等于tensor_a和tensor_b的dim0之和，<br/>dim1的长度与tensor_a和tensor_b的dim1相等 | 结果为二维 |

# split_2d_jagged算子输入与输出
| 名称             | 输入/输出  | 数据类型                     | 数据格式         | 范围                                                    | 说明    |
|----------------|--------|--------------------------|--------------|-------------------------------------------------------|-------|
| values         | 输入     | bfloat16/float16/float32 | [dim0, dim1] | NA                                                    | 仅支持二维 |
| max_seqlens    | 输入     | int                      | int          | NA                                                    | NA    |
| offset_a       | 输入     | int                      | [dim0]       | 长度范围[2, 1024]                                         | 仅支持一维 |
| offset_b       | 输入     | int                      | [dim0]       | 长度范围[2, 1024]                                         | 仅支持一维 |
| dense_size     | 输入(可选) | int                      | 默认值：0    | 当前只支持默认值                                              | NA    |
| nPrefixToRight | 输入(可选) | int                      | 默认值：0        | 当前只支持默认值                                              | NA    |
| output_tensor1 | 输出     | bfloat16/float16/float32 | [dim0, dim1] | dim0的长度等于offset_a的最后一个偏移的值，<br/>dim1的长度与values的dim1相等 | 结果为二维 |
| output_tensor2 | 输出     | bfloat16/float16/float32 | [dim0, dim1] | dim0的长度等于offset_b的最后一个偏移的值，<br/>dim1的长度与values的dim1相等 | 结果为二维 |

### 编译与部署

算子编译与部署请参考 [RecSDK\cust_op\README.md](../../../../README.md) 中 "单算子使用说明" 章节：

### 算子调用示例

```python
import sysconfig
import torch
import torch_npu
import fbgemm_gpu
DEVICE = "npu:0"
torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# 准备输入
tensor_a = torch.tensor([[1,1,1,1,1,1,1,1],
                         [2,2,2,2,2,2,2,2],
                         [3,3,3,3,3,3,3,3]])
tensor_b = torch.tensor([[4,4,4,4,4,4,4,4],
                         [5,5,5,5,5,5,5,5],
                         [6,6,6,6,6,6,6,6],
                         [7,7,7,7,7,7,7,7],
                         ])
offset_a = torch.tensor([0, 1, 2, 3])
offset_b = torch.tensor([0, 1, 3, 4])
max_seqlens = 2  # 表示最大的offset偏移值

# 执行算子
concated_tensor = torch.ops.mxrec.concat_2d_jagged(max_seqlens, tensor_a, tensor_b, offset_a, offset_b)
# 执行split_2d_jagged以前向结果作为入参。
split_tensor_a, split_tensor_b = torch.ops.mxrec.split_2d_jagged(concated_tensor, max_seqlens, offset_a, offset_b)
```

> **提示**  
> 上述用例为通用场景执行，更详细精度、多场景测试用例，请参考完整测试文件：  
> - [`Rec SDK/cust_op/test/concat_2d_jagged_test/torch/test_concat_2d_jagged_tensor.py`](../../../../test/concat_2d_jagged_test/torch/test_concat_2d_jagged_tensor.py)