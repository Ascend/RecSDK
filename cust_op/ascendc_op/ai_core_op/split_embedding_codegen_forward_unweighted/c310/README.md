**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号           | 是否支持                  |
|----------------| ------------------------ |
| Atlas A5训练系列产品 | 是  |

# # split_embedding_codegen_forward_unweighted算子目录层级
```shell
-- split_embedding_codegen_forward_unweighted
   |-- c310
      |-- op_host                                            # 算子host侧实现
      |-- op_kernel                                          # 算子kernel侧实现
      |-- split_embedding_codegen_forward_unweighted.json    # 算子原型配置
      |-- README.md                                          # 算子说明文档
      |-- run.sh                                             # 算子编译部署脚本
```

# 功能
算子的主要功能是实现多表查询。

# 算子实现原理

```
# with bag sum or mean
def split_embedding_codegen_forword_unweighted(dev_weights, weights_offsets, D_offsets, indices, offsets, total_D, pool_mode):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0]-1) // feat_cnt
    results = np.zeros((batch_size, total_D)).astype(np.float32)
    for i in range(feat_cnt):
        embed_dim = D_offsets[i+1] - D_offsets[i]
        for b in range(batch_size):
            this_offset = offsets[i*batch_size+b]
            next_offset = offsets[i*batch_size+b+1]
            this_indics = indices[this_offset: next_offset]
            # sum
            if pool_mode == 0:
                seq_lens = 1
            # mean
            else:
                seq_lens = len(this_indics)
            for j in this_indics:
                this_embed_index = weights_offsets[i]+j*embed_dim
                this_embed = dev_weights[this_embed_index: this_embed_index+embed_dim]
                results[b, D_offsets[i]:D_offsets[i+1]] = results[b, D_offsets[i]:D_offsets[i+1]] + this_embed/seq_lens
    return result.astype(np.float32)

# with no bag
def split_embedding_nobag_codegen_forword_unweighted(dev_weights, weights_offsets, indices, offsets, total_D):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0]-1) // feat_cnt
    out_D0 = len(indices)
    out_D1 = total_D // feat_cnt  # EC模式下要求每张表的dim一致
    results = np.zeros((out_D0, out_D1)).astype(np.float32)
    result_indx = 0
    for i in range(len(offsets)-1):
        # 待查indics
        this_indice = indices[offsets[i]:offsets[i+1]]
        # 待查表
        weights_indx = i // batch_size
        for j in this_indice:
            this_embed_index = weights_offsets[weights_indx] + j * out_D1
            this_embed = dev_weights[this_embed_index: this_embed_index + out_D1]
            results[result_indx] = this_embed
            result_indx += 1
    return result.astype(np.float32)
```

# 算子输入与输出
|  名称  |  输入/输出  |  数据类型  |  数据格式  |  范围  |  说明  |
|  ---- |  ---- |  ----  |  ----  |  ----  |  ----  |
|  dev_weights | 输入 | float32 | [total_table_size] | NA | 一维数组,所有表的权重，表的embedding_dim必须为8的整数倍 |
|  uvm_weight | 输入 | float32 | NA | NA  | 保留参数|
|  lxu_cache_weight | 输入 | float32 | NA | NA | 保留参数 |
|  weights_pacements | 输入 | int32 | NA | NA | 保留参数 |
|  weights_offsets | 输入 | int64 | [feat_cnt] | feat_cnt >= table_num, | 一维数组 |
|  D_offsets | 输入 | int64 | [feat_cnt + 1] | 数值必须从0开始依次递增 | 每个特征的embedding_dim的累加和 |
|  indices | 输入 | int64 | NA | len(indices) = offset[-1], 每张表的索引的大小[0, num_embedding] | 查表索引，需用户自行保证合法性，否则可能导致算子执行失败 |
|  hash_indices | 输入 | int64 | NA | len(indices) = offset[-1], 每张表的索引的大小[0, num_embedding] | 查表索引，需用户自行保证合法性，否则可能导致算子执行失败 |
|  offsets | 输入 | float32/int64 | [ [feat_cnt * batch_size + 1]] | 数值必须从0开始依次递增 | 查表索引对应的偏移 |
|  lxu_cache_locations | 输入 | int32 | NA | NA |保留参数 |
|  total_D | 属性 | int64 |  NA | NA | 所有特征的embedding_dim之和 |
|  max_D | 属性 | int64 | NA | NA | 最大的embedding_dim |
|  pooling_mode | 属性 | int64 | NA  | NA | poolingSum:0, poolingMean:1, poolingNone:2 |
|  output_dtype | 属性 | int64 | NA | NA |保留参数 |
|  is_experimental | 属性 | bool |  NA | NA | 保留参数 |
|  out | 输出 | float32 |poolingSum/poolingMean: [batch_size, total_D] poolingNone:[len(indices), maxD] | NA | NA |

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/split_embedding_codegen_forward_unweighted/README.md)