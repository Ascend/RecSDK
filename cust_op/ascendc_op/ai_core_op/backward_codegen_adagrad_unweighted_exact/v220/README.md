**说明**

本算子仅支持NPU调用。

# 产品支持情况
| 硬件型号              | 是否支持                  |
| -------------------- | ------------------------ |
| Atlas A2训练系列产品  | 是  |
| Atlas A3训练系列产品  | 是  |

# backward_codegen_adagrad_unweighted_exact算子目录层级
```shell
-- backward_codegen_adagrad_unweighted_exact
   |-- v220
      |-- op_host                                            # 算子host侧实现
      |-- op_kernel                                          # 算子kernel侧实现
      |-- backward_codegen_adagrad_unweighted_exact.json    # 算子原型配置
      |-- README.md                                          # 算子说明文档
      |-- run.sh                                             # 算子编译部署脚本
```

# 算子输入与输出
| 名称                          |  输入/输出  | 数据类型    |  数据格式  | 范围                       | 说明                                                |
|-----------------------------|  ---- |---------|  ----  |--------------------------|---------------------------------------------------|
| grad_output                 | 输入 | float32 | poolingSum/poolingMean: [batch_size, total_D] poolingNone:[len(indices), maxD] | NA                       | 查询向量的反向的梯度                                        |
| dev_weights                 | 输入 | float32 | [total_table_size] | NA                       | 一维数组,所有表的权重，表的embedding_dim必须为8的整数倍               |
| uvm_weights                 | 输入 | float32 | NA | NA                       | 预留参数不支持配置                                         |
| lxu_cache_weights           | 输入 | float32 | NA | NA                       | 预留参数不支持配置                                         |
| weights_placements          | 输入 | int32   | NA | NA                       | 一维数组, 每个特征对应的表偏移起始                                |
| weights_offsets             | 输入 | int64   | [feat_cnt] | feat_cnt >= table_num    | 一维数组, 每个特征对应的表偏移起始                                |
| D_offsets                   | 输入 | int32   | [feat_cnt + 1] | 数值必须从0开始依次递增             | 每个特征的embedding_dim的累加和                            |
| hash_size_cumsum            | 输入 | int64   | [feat_cnt] | 数值必须从0开始依次递增             | 每个特征的num_embedding累加和                             |
| indices                     | 输入 | int64   | NA | 每张表的索引[0, num_embedding] | 查表索引,一维数组 len(indices) = offset[-1], 由用户保证输入数据正确性 |
| offsets                     | 输入 | int64   | [feat_cnt * batch_size + 1] | 数值必须从0开始依次递增             | 查表索引对应的偏移, 由用户保证输入数据正确性                           |
| lxu_cache_weights           | 输入 | int32   | NA | NA                       | 预留参数不支持配置                                         |
| momentum1_dev               | 输入 | float32 | [total_table_size] | NA                       | 一阶动量，用于adagrad和adam优化器                            |
| momentum1_uvm               | 输入 | float32 | [total_table_size] | NA                       | 保留参数                                              |
| momentum1_placements        | 输入 | int32   | NA | NA                       | 保留参数                                              |
| momentum1_offsets           | 输入 | int64   | NA | NA                       | 保留参数                                              |
| momentum2_dev               | 输入 | float32 | NA | NA                       | 二阶动量，用于adam优化器                                    |
| momentum2_uvm               | 输入 | float32 | NA | NA                       | 保留参数                                              |
| momentum2_placements        | 输入 | int32   | NA | NA                       | 保留参数                                              |
| momentum2_offsets           | 输入 | int32   | NA | NA                       | 保留参数                                              |
| hash_indices                | 可选输入 | int64   | NA | [0, num_embedding]       | 映射后的查表索引, 一维数组 len(indices) = offset[-1]          |
| unique_id                   | 可选输入 | float32 | NA | [0, num_embedding]       | 去重后查表索引, 一维数组 len(unique_id) = offset[-1]         |
| unique_hash_size            | 可选输入 | int64   | NA | NA                       | 查表索引对应的偏移                                         |
| unique_inverse              | 可选输入 | int64   | NA | [0, num_embedding]       | 去重后的索引和原始索引的对应关系                                  |
| table_indice_offsets        | 可选输入 | float32 | [feat_cnt + 1] | NA                       | 每个特征的查表下标数累加和                                     |
| max_D                       | 属性 | int64   | NA | NA                       | 最大的embedding_dim                                  |
| total_hash_size_bits        | 属性 | bool    | NA | NA                       | hash表size和的int值用多少位bit表示                          |
| pooling_mode                | 属性 | int64   | NA | NA                       | poolingSum:0, poolingMean:1, poolingNone:2        |
| BT_block_size               | 属性 | int64   | NA | NA                       | 保留参数                                              |
| max_segment_length_per_warp | 属性 | int64   | NA | NA                       | 保留参数                                              |
| stochastic_rounding         | 属性 | int64   | NA | NA                       | 保留参数                                              |
| info_B_num_bits             | 属性 | int64   | NA | NA                       | 保留参数                                              |
| info_B_mask_int64           | 属性 | int64   | NA | NA                       | 保留参数                                              |
| use_uniq_cache_locations    | 属性 | int64   | NA | NA                       | 保留参数                                              |
| use_homogeneous_placements  | 属性 | int64   | NA | NA                       | 保留参数                                              |
| optim_type                  | 属性 | int64   | NA | NA                       | 优化器类型，adagrad:1, adam:2, sgd:3                    |
| eps                         | 属性 | float32 | NA | NA                       | 一个非常小的常数，防止分母为0，用于adam和adagrad优化器                 |
| learning_rate               | 属性 | float32 | NA | NA                       | 学习率                                               |
| beta1                       | 属性 | float32 | NA | NA                       | 衰减率，通常为0.9                                        |
| beta2                       | 属性 | float32 | NA | NA                       | 衰减率，通常为0.999                                      |
| iter                        | 属性 | float32 | NA | NA                       | 迭代次数用于adam优化器                                     |
| use_optimize                | 属性 | bool    | NA | true or false            | 是否更新参数,默认true                                     |
| out                         | 输出 | float32 | [len(unique_id), maxD] | NA                       | 查表索引的梯度累加和                                        |
| momentum1_dev_out           | 输出 | float32 | [total_table_size] | NA                       | 更新后的一阶动量                                          |
| momentum2_dev_out           | 输出 | float32 | [total_table_size] | NA                       | 更新后的二阶动量                                          |
| weights_dev_out             | 输出 | float32 | [total_table_size] | NA                       | 更新后的表的权重                                          |

# 算子实现原理
## backward_codegen_adagrad_unweighted_exact实现原理

```python3
import numpy as np


def backward_codegen_adagrad_unweighted_exact(grad_output, dev_weights, weights_offsets, D_offsets, indices, offsets,
                                              momentum1_dev, eps, learning_rate, maxD, hash_size_cumsum):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0] - 1) // feat_cnt
    results = np.zeros(dev_weights.shape).astype(np.float32)
    hash_table = [0 for i in range(hash_size_cumsum[-1])]

    this_offset_i = 0
    for i, ind in enumerate(indices):
        if i >= offsets[this_offset_i + 1]:
            this_offset_i = this_offset_i + 1
        table_index = this_offset_i // grad_output.shape[0]
        index_in_all_table = hash_size_cumsum[table_index] + ind
        hash_table[index_in_all_table] = i

    for i in range(batch_size):
        for j in range(feat_cnt):
            offset_this = offsets[j * batch_size + i]
            offset_this_i = offsets[j * batch_size + i + 1]

            this_grad = grad_output[i, D_offsets[j]:D_offsets[j + 1]]
            this_table_D = D_offsets[j + 1] - D_offsets[j]
            table_index = j
            for k in range(offset_this, offset_this_i):
                ind = indices[k]
                # this_table_ind = weights_offsets[j]+ind*this_table_D
                index_in_all_table = hash_size_cumsum[table_index] + ind
                output_ind = hash_table[index_in_all_table]

                results[output_ind * maxD: output_ind * maxD + this_table_D] += this_grad

    this_offset_i = 0
    grad = np.zeros_like(dev_weights)
    for i in range(indices.shape[0]):
        if (i >= offsets[this_offset_i + 1]):
            this_offset_i = this_offset_i + 1

        table_index = this_offset_i // grad_output.shape[0]
        true_ind = indices[i]
        index_in_all_table = hash_size_cumsum[table_index] + true_ind

        if (i != hash_table[index_in_all_table]):
            continue

        this_weight_offset = weights_offsets[table_index]
        this_embed_d = D_offsets[table_index + 1] - D_offsets[table_index]
        table_offset_of_this_index = this_weight_offset + this_embed_d * true_ind
        grad[table_offset_of_this_index:table_offset_of_this_index + this_embed_d] = results[
                                                                                     i * maxD:i * maxD + this_embed_d]

    m = momentum1_dev + grad ** 2
    ada_learning_rate = learning_rate / (np.sqrt(m) + eps)
    delta = ada_learning_rate * grad
    return grad, m, dev_weights - delta

```

## backward_codegen_adam_unweighted_exact实现原理

```python3

import numpy as np


def backward_codegen_adam_unweighted_exact(grad_output,
                                           dev_weights,
                                           weights_offsets,
                                           D_offsets,
                                           indices,
                                           offsets,
                                           momentum1_dev,
                                           momentum2_dev,
                                           eps,
                                           learning_rate,
                                           beta1,
                                           beta2,
                                           iter,
                                           maxD,
                                           hash_size_cumsum):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0] - 1) // feat_cnt
    results = np.zeros(dev_weights.shape).astype(np.float32)
    hash_table = [0 for i in range(hash_size_cumsum[-1])]

    this_offset_i = 0
    for i, ind in enumerate(indices):
        if i >= offsets[this_offset_i + 1]:
            this_offset_i = this_offset_i + 1
        table_index = this_offset_i // grad_output.shape[0]
        index_in_all_table = hash_size_cumsum[table_index] + ind
        hash_table[index_in_all_table] = i

    for i in range(batch_size):
        for j in range(feat_cnt):
            offset_this = offsets[j * batch_size + i]
            offset_this_i = offsets[j * batch_size + i + 1]

            this_grad = grad_output[i, D_offsets[j]:D_offsets[j + 1]]
            this_table_D = D_offsets[j + 1] - D_offsets[j]
            table_index = j
            for k in range(offset_this, offset_this_i):
                ind = indices[k]
                # this_table_ind = weights_offsets[j]+ind*this_table_D
                index_in_all_table = hash_size_cumsum[table_index] + ind
                output_ind = hash_table[index_in_all_table]

                results[output_ind * maxD: output_ind * maxD + this_table_D] += this_grad

    this_offset_i = 0
    grad = np.zeros_like(dev_weights)
    for i in range(indices.shape[0]):
        if i >= offsets[this_offset_i + 1]:
            this_offset_i = this_offset_i + 1

        table_index = this_offset_i // grad_output.shape[0]
        true_ind = indices[i]
        index_in_all_table = hash_size_cumsum[table_index] + true_ind

        if i != hash_table[index_in_all_table]:
            continue

        this_weight_offset = weights_offsets[table_index]
        this_embed_d = D_offsets[table_index + 1] - D_offsets[table_index]
        table_offset_of_this_index = this_weight_offset + this_embed_d * true_ind
        grad[table_offset_of_this_index:table_offset_of_this_index + this_embed_d] = results[
                                                                                     i * maxD:i * maxD + this_embed_d]

    m1 = beta1 * momentum1_dev + (1 - beta1) * grad
    m2 = beta2 * momentum2_dev + (1 - beta2) * np.square(grad)

    v_bias_corr = m1 / (1 - beta1 ** iter)
    s_bias_corr = m2 / (1 - beta2 ** iter)

    delta = learning_rate * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    return grad, m1, m2, dev_weights - delta

```

##  backward_codegen_sgd_unweighted_exact实现原理

```python3
import numpy as np


def backward_codegen_sgd_unweighted_exact(grad_output,
                                          dev_weights,
                                          weights_offsets,
                                          D_offsets,
                                          indices,
                                          offsets,
                                          learning_rate,
                                          maxD,
                                          hash_size_cumsum):
    feat_cnt = weights_offsets.shape[0]
    batch_size = (offsets.shape[0] - 1) // feat_cnt
    results = np.zeros(dev_weights.shape).astype(np.float32)
    hash_table = [0 for i in range(hash_size_cumsum[-1])]

    this_offset_i = 0
    for i, ind in enumerate(indices):
        if i >= offsets[this_offset_i + 1]:
            this_offset_i = this_offset_i + 1
        table_index = this_offset_i // grad_output.shape[0]
        index_in_all_table = hash_size_cumsum[table_index] + ind
        hash_table[index_in_all_table] = i

    for i in range(batch_size):
        for j in range(feat_cnt):
            offset_this = offsets[j * batch_size + i]
            offset_this_i = offsets[j * batch_size + i + 1]

            this_grad = grad_output[i, D_offsets[j]:D_offsets[j + 1]]
            this_table_D = D_offsets[j + 1] - D_offsets[j]
            table_index = j
            for k in range(offset_this, offset_this_i):
                ind = indices[k]
                # this_table_ind = weights_offsets[j]+ind*this_table_D
                index_in_all_table = hash_size_cumsum[table_index] + ind
                output_ind = hash_table[index_in_all_table]

                results[output_ind * maxD: output_ind * maxD + this_table_D] += this_grad

    this_offset_i = 0
    grad = np.zeros_like(dev_weights)
    for i in range(indices.shape[0]):
        if i >= offsets[this_offset_i + 1]:
            this_offset_i = this_offset_i + 1

        table_index = this_offset_i // grad_output.shape[0]
        true_ind = indices[i]
        index_in_all_table = hash_size_cumsum[table_index] + true_ind

        if i != hash_table[index_in_all_table]:
            continue

        this_weight_offset = weights_offsets[table_index]
        this_embed_d = D_offsets[table_index + 1] - D_offsets[table_index]
        table_offset_of_this_index = this_weight_offset + this_embed_d * true_ind
        grad[table_offset_of_this_index:table_offset_of_this_index + this_embed_d] = results[
                                                                                     i * maxD:i * maxD + this_embed_d]

    delta = learning_rate * grad
    return grad, dev_weights - delta

```

# 算子编译部署

算子编译请参考[RecSDK\cust_op\README.md](../../../../README.md)中"单算子使用说明"-"算子编译"章节。

注：详细算子调用示例参考Pytorch框架下[README.md](../../../../framework/torch_plugin/torch_library/split_embedding_codegen_forward_unweighted/README.md)