**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A2训练系列产品  | 是  |
| Atlas A3训练系列产品  | 是  |

# dense_embedding_codegen_lookup_function算子目录层级
```shell
-- dense_embedding_codegen_lookup_function
   |-- v220
      |-- op_host                                            # 算子host侧实现
      |-- op_kernel                                          # 算子kernel侧实现
      |-- dense_embedding_codegen_lookup_function.json       # 算子原型配置
      |-- README.md                                          # 算子说明文档
      |-- run.sh                                             # 算子编译部署脚本
```

# 功能
算子的主要功能是实现dense_embedding_codegen_lookup_function多表查询。

# 算子实现原理

```python
def dense_embedding_codegen_lookup_function(dev_weights, weights_offsets, indices, offsets, max_D):
    result = [0] * len(indices) * max_D  # 初始化结果列表
    offset = 0
    batch_size = (len(offsets) - 1) // len(weights_offsets)  # 修正除法运算
    for i in range(len(weights_offsets)):
        for j in range(offsets[i * batch_size], offsets[(i + 1) * batch_size + 1]):
            for k in range(max_D):
                result[offset] = dev_weights[(indices[j] + weights_offsets[i] // max_D) * max_D + k]
                offset += 1
    return result
```

# 算子输入与输出
|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  dev_weights | 输入 | float32 | [total_table_size] | NA | 一维数组,所有表的权重，表的embedding_dim必须为8的整数倍 |
|  weights_offsets | 输入 | int64 | [feat_cnt] | feat_cnt >= table_num, | 一维数组 |
|  D_offsets | 输入 | int32/int64 | [feat_cnt + 1] | 数值必须从0开始依次递增 | 每个特征的embedding_dim的累加和 |
|  hash_size_cumsum | 输入 | int64 | [feat_cnt + 1] | 数值必须从0开始依次递增 | 每个特征的num_embedding累加和 |
|  indices | 输入 | int64 | NA | len(indices) = offset[-1], 每张表的索引的大小[0, num_embedding] | 查表索引，需用户自行保证合法性，否则可能导致算子执行失败 |
|  offsets | 输入 | int64 | [feat_cnt * batch_size + 1] | 数值必须从0开始依次递增 | 查表索引对应的偏移 |
|  indice_weights | 可选输入 | float32 | NA | NA | 保留参数 |
|  B_offset | 可选输入 | int64 | NA | NA | 保留参数 |
|  vbe_output_offsets_feature_rank | 可选输入 | int64 | NA | NA | 保留参数 |
|  vbe_B_offsets_rank_per_feature | 可选输入 | int64 | NA | NA | 保留参数 |
|  total_D | 属性 | int64 |  NA | NA | 所有特征的embedding_dim之和 |
|  max_D | 属性 | int64 | NA | NA | 最大的embedding_dim |
|  total_hash_size_bits | 属性 | int64 | NA | NA | hash表size和的int值用多少位bit表示 |
|  pooling_mode | 属性 | int64 | NA  | NA | poolingSum:0, poolingMean:1, poolingNone:2 |
|  feature_requires_grad | 属性 | int64 | NA | NA | 保留参数 |
|  output_dtype | 属性 | int64 | NA | NA | 保留参数 |
|  max_B | 属性 | int64 | NA | NA | 保留参数 |
|  max_B_feature_rank | 属性 | int64 | NA | NA | 保留参数 |
|  vbe_output_size | 属性 | int64 | NA | NA | 保留参数 |
|  out | 输出 | float32 | [len(indices), maxD] | NA | 查询结果 |

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/dense_embedding_codegen_lookup_function/README.md)