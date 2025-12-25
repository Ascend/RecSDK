**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持 |
| -------------------- |------|
| Atlas A2训练系列产品  | 是    |
| Atlas A3训练系列产品  | 是    |
| Atlas 推理系列产品    | 否    |

# concat_2d_jagged算子目录层级

```shell
-- concat_2d_jagged
   |-- v220
      |-- op_host                 # 算子host侧实现
      |-- op_kernel               # 算子kernel侧实现
      |-- concat_2d_jagged.json    # 算子原型配置
      |-- README.md               # 算子说明文档
      |-- run.sh                  # 算子编译部署脚本
```
# 功能

将两个jagged Tensor按照offset在dim1维度上进行拼接，合并成一个tensor。

# 算子实现原理

输入:

```python
# 第一个tensor
values = [tensor([[1,1,1,1,1,1,1,1],
                  [2,2,2,2,2,2,2,2],
                  [3,3,3,3,3,3,3,3]
                  ]),
          tensor([[4,4,4,4,4,4,4,4],
                  [5,5,5,5,5,5,5,5],
                  [6,6,6,6,6,6,6,6],
                  [7,7,7,7,7,7,7,7],
                  ])] 
# 长度为2N,前N个数表示第一个tensor的offset,后N个数表示第2个tensor的offset。
offsets = [0, 1, 2, 3, 0, 1, 3, 4] 
# offset的长度，两个tensor的offset长度必须一致。
offsetLen = 4
# 拼接的tensor个数，这里仅支持两个。
jtNum = 2
```
输出：
```python
result = tensor([[1,1,1,1,1,1,1,1],
                [4,4,4,4,4,4,4,4],
                [2,2,2,2,2,2,2,2],
                [5,5,5,5,5,5,5,5],
                [6,6,6,6,6,6,6,6],
                [3,3,3,3,3,3,3,3],
                [7,7,7,7,7,7,7,7]
                 ])
```
# 算子输入与输出
| 名称     |  输入/输出 |  数据类型  | 数据格式                            | 范围                                                               | 说明                |
|--------|  - |  ----  |---------------------------------|------------------------------------------------------------------|-------------------|
| values | 输入 | bfloat16/float16/float32 | [tensor_a, tensor_b] | tensor_a与tensor_b必须为二维且第二维相等                                     | 待拼接的tensor list   |
| offsets | 输入 | int | [dim0]                          | 一维,长度为2N                                                         | 待拼接的tensor的偏移     |
| offsetLen | 输入 | int | NA                              | N                                                                | 单个tensor的offset长度 |
| jtNum  | 输入 | int | NA                              | 支持jtNum = 2                                                      | 待拼接tensor个数       |
| result | 输出 | bfloat16/float16/float32 | [dim0, dim1]                    | 结果为二维,<br/>dim0的长度等于两个tensor的dim0之和，<br/>dim1的长度与拼接tensor的dim1相等。 | NA                |

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"1.算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/concat_2d_jagged/README.md)