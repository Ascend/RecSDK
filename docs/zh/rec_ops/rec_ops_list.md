
# RecOps 算子列表

RecOps 是 Rec SDK 基于 Ascend C 开发的推荐场景自定义算子集，为各框架组件（tf_rec_v1、tf_rec_v2、torch_rec_v1、torch_rec_v2）提供基础算子能力。

## 简介

| 算子名称 | 功能介绍 | 支持硬件 |
|---------|---------|---------|
| [concat_jagged_tensor](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor/v220/README.md) | 将两个jagged Tensor按照offset在dim1维度上进行拼接，合并成一个tensor。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [concat_jagged_tensor_grad](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor_grad/v220/README.md) | 将一个Tensor按照offset切分成两个不一定等长的tensor。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [cust_op_by_addr](../../../cust_op/ascendc_op/ai_core_op/cust_op_by_addr/c310/README.md) | 用addr地址作为入参进行嵌入表查询，替换tf.gather算子，支持动态扩容。 | ascend910b, ascend910_93, ascend950 |
| [disentangle_attention](../../../cust_op/ascendc_op/ai_core_op/disentangle_attention/v220/README.md) | 实现DeBERTa模型中的解耦注意力(disentangle attention)功能。 | ascend910b, ascend910_93, ascend950 |
| [fused_lazy_adam](../../../cust_op/ascendc_op/ai_core_op/fused_lazy_adam/v220/README.md) | 实现LazyAdam优化器反向更新时m、v、variable三项数据的计算和更新。 | ascend910b, ascend910_93, ascend950 |
| [fused_sgd](../../../cust_op/ascendc_op/ai_core_op/fused_sgd/v220/README.md) | 实现SGD优化器反向更新时参数的计算和更新。 | ascend910b, ascend910_93, ascend950 |
| [gather_for_rank1](../../../cust_op/ascendc_op/ai_core_op/gather_for_rank1/v220/README.md) | 实现shape为1的index_select操作，从一维张量中根据索引选择元素。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_dense_backward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward/v220/README.md) | 实现HSTU融合算子的反向传播，计算Q、K、V和attn_bias的梯度。 | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_backward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward_fuxi/v220/README.md) | 实现HSTU-Fuxi融合算子的反向传播，计算注意力机制中的梯度。 | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_forward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward/README.md) | 使用HSTU融合算子实现推荐场景中的注意力机制，支持GQA和dim不等特性。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_dense_forward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward_fuxi/v220/README.md) | 基于HSTU融合算子实现推荐场景Alpha-Fuxi模型中的注意力机制，支持timestamp_bias和position_bias，支持normal/jagged/paged三种layout。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_backward_v2](../../../cust_op/ascendc_op/ai_core_op/hstu_v2/BWD_README.md) | HSTU V2融合算子的反向传播实现，计算Query、Key、Value以及RAB注意力分数的梯度。 | ascend950 |
| [index_select_for_rank1_backward](../../../cust_op/ascendc_op/ai_core_op/index_select_for_rank1_backward/v220/README.md) | 实现index_select的反向传播，计算梯度。 | ascend910b, ascend910_93, ascend950 |
| [in_linear_silu](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu/v220/README.md) | 用于HSTU Attention前将合并归一化后的UVQK进行Linear、Silu操作后拆分成User、Value、Query、Key四个Tensor。 | ascend910b, ascend910_93, ascend950 |
| [in_linear_silu_backward](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu_backward/v220/README.md) | in_linear_silu算子的反向传播实现，计算输入x、weight和bias的梯度。 | ascend910b, ascend910_93, ascend950 |
| [lccl](../../../cust_op/ascendc_op/ai_core_op/lccl/v220/README.md) | 利用AICore直接访问对端片上内存的能力，使用内存语义进行集合通信（AllToAll、AllUss、GatherAll）。 | ascend910b, ascend910_93 |
| [ln_mul](../../../cust_op/ascendc_op/ai_core_op/ln_mul/v220/README.md) | 对输入X进行LayerNorm，然后与gamma和beta计算，最后与输入U相乘。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [multislice_concat](../../../cust_op/ascendc_op/ai_core_op/multislice_concat/v220/README.md) | 对输入的二维Tensor在第二个维度按指定位置和长度切片，输出若干个切片组成的Tensor。 | ascend910b, ascend950 |
| [norm_multiply_dropout](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout/v220/README.md) | 实现layer_norm + multiply + dropout计算逻辑的融合算子。 | ascend910b, ascend910_93, ascend950 |
| [norm_multiply_dropout_backward](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout_backward/v220/README.md) | 实现layer_norm + multiply + dropout计算的反向求导逻辑。 | ascend910b, ascend910_93, ascend950 |
| [pcie_through](../../../cust_op/ascendc_op/ai_core_op/pcie_through/v220/README.md) | 利用pcie_through在host和device交换数据量较大的场景，提升换入换出的性能。 | ascend910b |
| [relative_attn_bias_backward](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_backward/v220/README.md) | 针对HSTU模型rab的time部分，计算时间戳参数反向传播中的梯度值。 | ascend910b, ascend910_93, ascend950 |
| [relative_attn_bias_pos](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_pos/v220/README.md) | 针对HSTU模型rab的pos部分计算。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [relative_attn_bias_time](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_time/v220/README.md) | 针对HSTU模型rab的time部分计算。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [reverse_sequence](../../../cust_op/ascendc_op/ai_core_op/reverse_sequence/v220/README.md) | 将输入数据根据指定长度参数(seq_lengths)进行第二维的逆序操作。 | ascend910b, ascend910_93, ascend950 |
| [token_mixing](../../../cust_op/ascendc_op/ai_core_op/token_mixing/v220/README.md) | 实现x与其转置x_t相加后的归一化操作。 | ascend910b, ascend950 |

## 算子目录结构

```text
cust_op/ascendc_op/ai_core_op/
├── concat_jagged_tensor/                              # Jagged张量拼接
├── concat_jagged_tensor_grad/                         # Jagged张量拼接反向
├── cust_op_by_addr/                                   # 地址查找
├── disentangle_attention/                             # 解耦注意力
├── fused_lazy_adam/                                   # LazyAdam优化器
├── fused_sgd/                                         # SGD优化器
├── gather_for_rank1/                                  # 一维Gather
├── hstu_dense_backward/                               # HSTU反向
├── hstu_dense_backward_fuxi/                          # HSTU-Fuxi反向
├── hstu_dense_forward/                                # HSTU前向
├── hstu_dense_forward_fuxi/                           # HSTU-Fuxi前向
├── hstu_v2/                                           # HSTU V2反向
├── index_select_for_rank1_backward/                   # IndexSelect反向
├── in_linear_silu/                                    # Linear+Silu融合
├── in_linear_silu_backward/                           # Linear+Silu反向
├── lccl/                                              # 集合通信
├── ln_mul/                                            # LayerNorm+Multiply融合
├── multislice_concat/                                 # 多切片拼接
├── norm_multiply_dropout/                             # Norm+Multiply+Dropout融合
├── norm_multiply_dropout_backward/                    # Norm+Multiply+Dropout反向
├── pcie_through/                                      # PCIe数据传输
├── relative_attn_bias_backward/                       # 相对位置偏置反向
├── relative_attn_bias_pos/                            # 相对位置偏置位置
├── relative_attn_bias_time/                           # 相对位置偏置时间
├── reverse_sequence/                                  # 序列反转
└── token_mixing/                                      # Token混合
```
