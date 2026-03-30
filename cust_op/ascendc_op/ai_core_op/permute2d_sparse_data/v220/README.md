**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持 |
| -------------------- |------|
| Atlas A2训练系列产品  | 是    |
| Atlas A3训练系列产品  | 是    |

# permute2d_sparse_data算子文件结构
```shell
-- permute2d_sparse_data
   |-- v220
      |-- op_host                       # 算子host侧实现
      |-- op_kernel                     # 算子kernel侧实现
      |-- permute2d_sparse_data.json    # 算子原型配置
      |-- README.md                     # 算子说明文档
      |-- run.sh                        # 算子编译部署脚本
```

# 功能

同fbgemm的permute2d_sparse_data方法, 实现了对二维稀疏数据进行重排。

# 算子实现原理

输入:
```python
permute = [0, 2]
lengths = [[1, 1],
           [1, 1],
           [1, 2],
           [0, 1]]
values = [0, 1, 2, 3, 4, 5, 6, 7]
weights = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
permuted_lengths_sum = 5
```

输出：
```python
permuted_lengths = [[1, 1],
                    [1, 2]]                     # 获取lengths[permute]
permuted_values = [0, 1, 4, 5, 6]
permuted_weights = [1.0, 1.1, 1.4, 1.5, 1.6]
```
说明：

1.permute入参为[0, 2],分别获取lengths[0]的值为[1, 1]表示两个元素,对应values中的[0，1]; lengths[2]的值为[1, 2]表示三个元素,对应values中的[4, 5, 6]。
再将[0，1]和[4, 5, 6]拼接成最终的permuted_values结果。其中lengths[1]中的[1, 1]对应values的两个元素为[2, 3]在例子中未获取。

2.permuted_weights的处理方式与permuted_values一致。

# 算子输入与输出
|  名称  | 输入/输出  | 参数类型    | 数据类型       | 数据格式                                          | 范围                  |
|  ---- |--------|---------|------------|-----------------------------------------------|---------------------|
|  permute | 输入     | Tensor  | int32/int64      | [indices]                                     | permute中的每个值均满足: >= 0 且 < `lengths.shape[0]` |
|  lengths | 输入     | Tensor  | int32/int64 | [ [lengths], [lengths],... ]                  |           
|  values | 输入     | Tensor  | int32/int64/fp32/fp16/bf16 | [values]                                      | values的长度等于`lengths.sum()` | 
|  weights | 输入 | Tensor  | fp32/fp16/bf16/double/int32/int64       | [weights] / [weights，columns]                                  | weight的长度等于`lengths.sum()`, 支持weights.dense_dim > 1 (多列)情况|
|  permuted_lengths_sum | 输入(可选) | SymInt  | int64        | NA                                            |        [0, std::numeric_limits<int64>::max()]      |
|  enableWeights | 输入 | bool  | bool | NA                                            |              |
|  permuted_lengths | 输出     | Tensor  | int32/int64   | [permuted_lengths], [permuted_lengths], ... ] |                     |
|  permuted_values | 输出     | Tensor  | int32/int64/fp32/fp16/bf16   | [permuted_values]                             |                     |
|  permuted_weights | 输出     | Tensor  |  fp32/fp16/bf16/double/int32/int64  | [permuted_weights]                            |       |


说明：指定permuted_lengths_sum时，permuted_values/permuted_weights长度为permuted_lengths_sum，请用户自行保证数值正确;
未指定permuted_lengths_sum时，算子将计算得到permuted_lengths_sum


## 算子逻辑
```
import torch
import fbgemm_gpu
def permute2d_sparse_data(permute, lengths, values, weights, permuted_lengths_sum):
    (permuted_lengths, permuted_values, permuted_weights) = (
        torch.ops.fbgemm.permute_2D_sparse_data(permute, lengths, values, weights, permuted_lengths_sum)
    )

    return permuted_lengths, permuted_values, permuted_weights

```

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/permute2d_sparse_data/README.md)
