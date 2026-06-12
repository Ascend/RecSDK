# 使用PyTorch框架调用concat_2d_jagged和split_2d_jagged算子

该算子当前支持两种软件版本配套：PyTorch 2.6.0和PyTorch2.7.1。concat_2d_jagged 和 split_2d_jagged是互为前、反向的Tensor拼接和切分算子。详细配套说明见[RecSDK\cust_op\README.md](../../../../README.md)。

## concat_2d_jagged算子和split_2d_jagged算子

### PyTorch框架对外接口原型

```python
torch.ops.mxrec.concat_2d_jagged(maxSeqlen, valuesA, valuesB, offsetA, offsetB) -> Tensor

torch.ops.mxrec.split_2d_jagged(values, maxSeqlen, offsetA, offsetB) -> Tensor, Tensor
```

### 参数说明

#### concat_2d_jagged算子输入与输出

| 名称                | 输入/输出   | 参数类型     | 数据类型                           | 数据格式  | 范围                                                                  | 说明                                      |
|-------------------|---------|----------|--------------------------------|-------|---------------------------------------------------------------------|-----------------------------------------|
| maxSeqlen         | 输入      | int      | int                            | NA    | NA                                                                  | 最大序列长度，即offset中最大偏移。预留参数与开源一致,对当前功能无影响。 |
| valuesA           | 输入      | Tensor   | bfloat16/float16/float32/int32 | [dim0, dim1] | dim1和dtype必须与valuesB的一致                                             | 仅支持二维                                   |
| valuesB           | 输入      | Tensor   | bfloat16/float16/float32/int32 | [dim0, dim1] | dim1和dtype必须与valuesA的一致                                             | 仅支持二维                                   |
| offsetA           | 输入      | Tensor     | int                            | [dim0] | shape与offsetB一致,长度范围[2, 1024]                                       | 仅支持一维                                   |
| offsetB           | 输入      | Tensor     | int                            | [dim0] | shape与offsetA一致,长度范围[2, 1024]                                       | 仅支持一维                                   |
| isReplace         | 输入(可选)  | bool     | bool                           | NA    | 预留参数，当前只支持默认值Fales，传其他值不生效。                                         | NA                                      |
| nPrefixFromRight  | 输入(可选)  | int      | int                            | NA    | nPrefixFromRight >= 0                                               | 拼接时右侧tensor移动前缀的个数                                    |
| outputTensor      | 输出      | Tensor   | bfloat16/float16/float32/int32 | [dim0, dim1] | dim0的长度等于valuesA和valuesB的dim0之和，<br/>dim1的长度与valuesA和valuesB的dim1相等 | 结果为二维                                   |

说明：需确保valuesA和valuesB的dim0维长度与之对应的offset的最后一位大小相等，否则结果可能出错。

#### split_2d_jagged算子输入与输出

| 名称              | 输入/输出    | 参数类型   | 数据类型                            | 数据格式           | 范围                                                | 说明    |
|-----------------|----------|--------|---------------------------------|----------------|---------------------------------------------------|-------|
| values          | 输入       | Tensor | bfloat16/float16/float32/int32  | [dim0, dim1]   | NA                                                | 仅支持二维 |
| maxSeqlen       | 输入       | int    | int                             | NA             | NA                                                | 最大序列长度，即offset中最大偏移。预留参数与开源一致,对当前功能无影响。    |
| offsetA         | 输入       | Tensor | int                             | [dim0]      | shape与offsetB一致,长度范围[2, 1024]                                     | 仅支持一维 |
| offsetB         | 输入       | Tensor   | int                             | [dim0]      | shape与offsetA一致,长度范围[2, 1024]                                     | 仅支持一维 |
| denseSize       | 输入(可选)   | int    | int                             | NA             | 默认值：0 预留参数，当前只支持默认值，传其他值不生效。                      | NA    |
| nPrefixToRight  | 输入(可选)   | int    | int                             | NA             | 默认值：0 nPrefixToRight >= 0                         | 切分时右侧tensor的前缀个数    |
| outputTensor1   | 输出       | Tensor | bfloat16/float16/float32/int32  | [dim0, dim1]   | dim0的长度等于offsetA的最后一个偏移的值，<br/>dim1的长度与values的dim1相等 | 结果为二维 |
| outputTensor2   | 输出       | Tensor | bfloat16/float16/float32/int32  | [dim0, dim1]   | dim0的长度等于offsetB的最后一个偏移的值，<br/>dim1的长度与values的dim1相等 | 结果为二维 |

说明：需确保values的dim0维长度与offsetA和offsetB最后一位之和相等，否则结果可能出错。

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

**提示**
上述用例为通用场景执行，更详细精度、多场景测试用例，请参考完整测试文件:

[`Rec SDK/cust_op/test/concat_2d_jagged_test/torch/test_concat_2d_jagged_tensor.py`](../../../../test/concat_2d_jagged_test/torch/test_concat_2d_jagged_tensor.py)
