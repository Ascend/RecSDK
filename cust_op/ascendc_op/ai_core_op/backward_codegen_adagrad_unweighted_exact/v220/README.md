# backward_codegen_adagrad_unweighted_exact算子及样例说明
本算子仅支持NPU调用

## backward_codegen_adagrad_unweighted_exact算子文件结构

```shell
├── backward_codegen_adagrad_unweighted_exact.json    # 算子原型配置
├── op_host    # backward_codegen_adagrad_unweighted_exact算子Host侧实现
├── op_kernel  # backward_codegen_adagrad_unweighted_exact算子Kernel侧实现
├── README.md  # backward_codegen_adagrad_unweighted_exact算子说明文档
└── run.sh     # backward_codegen_adagrad_unweighted_exact算子安装脚本
```

## backward_codegen_adagrad_unweighted_exact融合算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的backward_codegen_adagrad_unweighted_exact, 实现了将反向梯度计算后，使用adagrad算法，将weights和momentum1进行更新
b) 算子参数说明：

* grad_output: 查询向量的反向的梯度；
* dev_weights: 预留参数不支持配置；
* uvm_weights: 预留参数不支持配置；
* lxu_cache_weights: 预留参数不支持配置;
* weights_placements: 预留参数不支持配置;
* weights_offsets: 每张表的偏移量;
* D_offsets: 每张表embeding dim的offsets;
* hash_size_cumsum: 表size的偏移;
* indices: 查询表的indics;
* offsets: indices对应的偏移;
* lxu_cache_locations: 预留参数不支持配置;
* hash_indices: 稀疏表查表的indics，可选参数;
* momentum1_dev: 输出值;
* momentum1_uvm: 预留参数不支持配置;
* momentum1_placements: 预留参数不支持配置;
* momentum1_offsets: 预留参数不支持配置;
* unique_offsets: 每张表去重后的偏移量，可选参数;
* unique_ids: 稀疏表查表的索引值，可选参数;
* unique_inverse: 查询表的索引对应的unique_ids位置，可选参数;
* indice_size_cumsum: 每张表的查表索引个数累加和，可选参数；

* max_D: 表中最大的Embedding Dim;
* total_hash_size_bits: hash表size和的int值用多少位bit表示;
* pooling_mode: pooling的方式Sum或者Mean;
* BT_block_size: 预留参数不支持配置;
* max_segment_length_per_warp: 预留参数不支持配置;
* stochastic_rounding: 预留参数不支持配置;
* info_B_num_bits: 预留参数不支持配置;
* info_B_mask_int64: 预留参数不支持配置;
* info_B_mask_int64: 预留参数不支持配置;
* use_uniq_cache_locations: 预留参数不支持配置;
* use_homogeneous_placements: 预留参数不支持配置;
* eps: adagrad的eps;
* learning_rate: 学习率;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
*

支持的输入数据类型：grad_output、dev_weights、momentum1_dev为float32类型，hash_size_cumsum、indices、offsets为int64，D_offsets为int32；max_D、total_hash_size_bits、pooling_mode为int类型。eps、learning_rate为float类型

* grad_output的dims为[batchsize, total]，dev_weights的dims为所有表的[embed_dim * embed_size]
  ，weights_offsets为表的个数[ num_embed ], weights_offsets的dims为[ num_embed+1 ], D_offsets的dim为[ num_embed+1 ],
  hash_size_cumsum为[ num_embed+1 ]。indices的dim0为offset最后一位的值。offsets为[batchsize, num_embed]，embed_dim长度需为8的倍数。
* 配置unique_ids时，需同时配置unique_offsets, unique_inverse，配置后将使用unique信息进行参数更新。

2. 算子逻辑

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

## backward_codegen_adam_unweighted_exact融合算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的backward_codegen_adam_unweighted_exact,
实现了将反向梯度计算后，使用adam算法，将weights、momentum1和momentum2进行更新

b) 算子参数说明：

* grad_output: 查询向量的反向的梯度；
* dev_weights: 预留参数不支持配置；
* uvm_weights: 预留参数不支持配置；
* lxu_cache_weights: 预留参数不支持配置;
* weights_placements: 预留参数不支持配置;
* weights_offsets: 每张表的偏移量;
* D_offsets: 每张表embeding dim的offsets;
* hash_size_cumsum: 表size的偏移;
* indices: 查询表的indics;
* offsets: indices对应的偏移;
* lxu_cache_locations: 预留参数不支持配置;
* hash_indices: 稀疏表查表的indics，可选参数;
* momentum1_dev: 输出值;
* momentum1_uvm: 预留参数不支持配置;
* momentum1_placements: 预留参数不支持配置;
* momentum1_offsets: 预留参数不支持配置;
* momentum2_dev: 输出值;
* momentum2_uvm: 预留参数不支持配置;
* momentum2_placements: 预留参数不支持配置;
* momentum2_offsets: 预留参数不支持配置;
* unique_offsets: 每张表去重后的偏移量，可选参数;
* unique_ids: 稀疏表查表的索引值，可选参数;
* unique_inverse: 查询表的索引对应的unique_ids位置，可选参数;
* indice_size_cumsum: 每张表的查表索引个数累加和，可选参数；


* max_D: 表中最大的Embedding Dim;
* total_hash_size_bits: hash表size和的int值用多少位bit表示;
* pooling_mode: pooling的方式Sum或者Mean;
* BT_block_size: 预留参数不支持配置;
* max_segment_length_per_warp: 预留参数不支持配置;
* stochastic_rounding: 预留参数不支持配置;
* info_B_num_bits: 预留参数不支持配置;
* info_B_mask_int64: 预留参数不支持配置;
* info_B_mask_int64: 预留参数不支持配置;
* use_uniq_cache_locations: 预留参数不支持配置;
* use_homogeneous_placements: 预留参数不支持配置;
* eps: adam的eps;
* learning_rate: 学习率;
* beta1
* beta2
* iter
* weight_decay

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
*

支持的输入数据类型：grad_output、dev_weights、momentum1_dev、momentum2_dev为float32类型，hash_size_cumsum、indices、offsets为int64，D_offsets为int32；max_D、total_hash_size_bits、pooling_mode为int类型。eps、learning_rate、beta1、beta2为float类型

* grad_output的dims为[batchsize, total]，dev_weights的dims为所有表的[embed_dim * embed_size]
  ，weights_offsets为表的个数[ num_embed ], weights_offsets的dims为[ num_embed+1 ], D_offsets的dim为[ num_embed+1 ],
  hash_size_cumsum为[ num_embed+1 ]。
* indices的dim0为offset最后一位的值。offsets为[batchsize, num_embed]，embed_dim长度需为8的倍数。
* 配置unique_ids时，需同时配置unique_offsets, unique_inverse，配置后将使用unique信息进行参数更新。

2. 算子逻辑

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

## backward_codegen_sgd_unweighted_exact融合算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的backward_codegen_sgd_unweighted_exact, 实现了将反向梯度计算后，使用sgd算法，将weights进行更新

b) 算子参数说明：

* grad_output: 查询向量的反向的梯度；
* dev_weights: 预留参数不支持配置；
* uvm_weights: 预留参数不支持配置；
* lxu_cache_weights: 预留参数不支持配置;
* weights_placements: 预留参数不支持配置;
* weights_offsets: 每张表的偏移量;
* D_offsets: 每张表embeding dim的offsets;
* hash_size_cumsum: 表size的偏移;
* indices: 查询表的indics;
* offsets: indices对应的偏移;
* lxu_cache_locations: 预留参数不支持配置;
* hash_indices: 稀疏表查表的indics，可选参数;


* max_D: 表中最大的Embedding Dim;
* total_hash_size_bits: hash表size和的int值用多少位bit表示;
* pooling_mode: pooling的方式Sum或者Mean;
* BT_block_size: 预留参数不支持配置;
* max_segment_length_per_warp: 预留参数不支持配置;
* stochastic_rounding: 预留参数不支持配置;
* info_B_num_bits: 预留参数不支持配置;
* info_B_mask_int64: 预留参数不支持配置;
* info_B_mask_int64: 预留参数不支持配置;
* use_uniq_cache_locations: 预留参数不支持配置;
* use_homogeneous_placements: 预留参数不支持配置;
* learning_rate: 学习率;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.2.RC1.alpha001及之后版本；
*

支持的输入数据类型：grad_output、dev_weights为float32类型，hash_size_cumsum、indices、offsets为int64，D_offsets为int32；max_D、total_hash_size_bits、pooling_mode为int类型。learning_rate为float类型

* grad_output的dims为[batchsize, total]，dev_weights的dims为所有表的[embed_dim * embed_size]
  ，weights_offsets为表的个数[ num_embed ], weights_offsets的dims为[ num_embed+1 ], D_offsets的dim为[ num_embed+1 ],
  hash_size_cumsum为[ num_embed+1 ]。
* indices的dim0为offset最后一位的值。offsets为[batchsize, num_embed]，embed_dim长度需为8的倍数。

2. 算子逻辑

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
## 算子使用说明
请参考:[RecSDK-Torch 自定义算子说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/README.md)
