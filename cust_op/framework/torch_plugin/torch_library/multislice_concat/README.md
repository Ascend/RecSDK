**使用pytorch框架调用方式调用multislice_concat算子**

# Pytorch框架对外接口原型

```python
torch.ops.mxrec.multislice_concat(Tensor input, int concat_num, int[] slice_size,
int[] slice_begin, int[] slice_length) -> Tensor[]
```

# 参数说明

| 名称         | 输入/输出 | 参数类型 | 数据类型             | 数据格式           | 范围                                                            | 说明                                                                                                                                       |
| ------------ | --------- | -------- | -------------------- | ------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| input        | 输入      | Tensor   | float32/float16/bf16 | [B, D]             | B/D∈[1, 65535]                                                  | 输入的待切片Tensor                                                                                                                         |
| concat_num   | 输入      | int      | int                  | int                | concat_num∈[1, 256]                                             | 输出Tensor数量                                                                                                                             |
| slice_size   | 输入      | int[]    | int                  | [concat_num]       | concat_size[i]∈[1, 3600]<br>sum(concat_size)∈[concat_num, 3600] | 每个输出Tensor的拼接的切片数量，最多切成3600小片                                                                                           |
| slice_begin  | 输入      | int[]    | int                  | [sum(concat_size)] | concat_begin[i]∈[0, D-1]，D是input的列数                        | 每个输出Tensor的拼接的切片的起始偏移<br>slice_begin={slice_begin[0] ... slice_begin[slice_size[0] ... slice_begin[slice_size[concat_size]} |
| slice_length | 输入      | int[]    | int                  | [sum(concat_size)] | concat_length[i]∈[1, D-concat_begin[i]]，D是input的列数         | 每个输出Tensor的拼接的切片的长度                                                                                                           |
| output       | 输出      | Tensor[] | Tensor               | list[Tensor[B, K]] | 长度为concat_num的list                                          | 输出concat_num个二维Tensor，每个Tensor的第一维D与输入保持一致，第二维K为对应切片长度和                                                     |

# 算子运行样例

## 算子编译与部署

算子编译部署请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明" - "算子编译"章节。

## Pytorch编译

Pytorch框架适配层编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明" - "算子适配层编译"章节。

## 算子调用示例

算子调用示例请参考测试用例[test_multislice_concat.py](../../../../test/multislice_concat/torch/test_multislice_concat.py)
