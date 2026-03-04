**说明**

本算子仅支持NPU调用

# 产品支持情况
| 硬件型号              | 是否支持 |
| -------------------- |------|
| Atlas A2训练系列产品  | 是    |
| Atlas 推理系列产品    | 是    |

## multislice_concat算子目录层级

```shell
-- multislice_concat
    |-- v220
        |-- op_host                # 算子host侧实现
        |-- op_kernel              # 算子kernel侧实现
        |-- multislice_concat.json # 算子原型配置
        |-- README.md              # 算子说明文档
        |-- run.sh                 # 算子编译部署脚本
```

# 功能
对输入的二维Tensor在第二个维度按指定位置和长度切片，输出若干个由若干切片组成的Tensor

# 算子实现原理

## 算子计算逻辑
```python
def multislice_concat(input_data, concat_size, slice_begin, slice_length, concat_num):
    offset = 0
    result_matrices = []

    for i in range(concat_num):
        curr_concat_size = concat_size[i].item()

        all_columns =[]
        for j in range(offset, offset + curr_concat_size):
            curr_begin = slice_begin[j].item()
            curr_size = slice_length[j].item()

            cur_column = input_data[:, curr_begin:curr_begin + curr_size]
            all_columns.append(cur_column)
        new_matrix = torch.cat(all_columns, dim=1)
        result_matrices.append(new_matrix)

        offset += curr_concat_size
    
    return result_matrices
```

# 算子输入与输出

## 输入参数

| 名称  | 输入/输出 | 参数类型          | 数据类型             | 数据格式 | 范围           | 说明               |
| ----- | --------- | ----------------- | -------------------- | -------- | -------------- | ------------------ |
| input | 输入      | Tensor (REQUIRED) | float32/float16/bf16 | [B, D]   | B/D∈[1, 65535] | 输入的待切片Tensor |

## 属性参数

| 名称         | 参数类型 | 数据类型              | 范围/取值                                                       | 说明                                                                                                                                       |
| ------------ | -------- | --------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| concat_num   | Attr     | int                   | concat_num∈[1, 256]                                             | 输出Tensor数量                                                                                                                             |
| concat_size  | Attr     | int[concat_num]       | concat_size[i]∈[1, 3600]<br>sum(concat_size)∈[concat_num, 3600] | 每个输出Tensor的拼接的切片数量，最多切成3600小片                                                                                           |
| slice_begin  | Attr     | int[sum(concat_size)] | slice_begin[i]∈[0, D-1]，D是input的列数                         | 每个输出Tensor的拼接的切片的起始偏移<br>slice_begin={slice_begin[0] ... slice_begin[slice_size[0] ... slice_begin[slice_size[concat_size]} |
| slice_length | Attr     | int[sum(concat_size)] | slice_length[i]∈[1, D-slice_begin[i]]，D是input的列数           | 每个输出Tensor的拼接的切片的长度                                                                                                           |

## 输出参数

| 名称   | 输入/输出 | 参数类型          | 数据类型             | 数据格式           | 范围                   | 说明                                                                                   |
| ------ | --------- | ----------------- | -------------------- | ------------------ | ---------------------- | -------------------------------------------------------------------------------------- |
| output | 输出      | Tensor (REQUIRED) | float32/float16/bf16 | list[Tensor[B, K]] | 长度为concat_num的list | 输出concat_num个二维Tensor，每个Tensor的第一维D与输入保持一致，第二维K为对应切片长度和 |

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/multislice_concat/README.md)