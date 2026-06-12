
# RecOps 算子列表

RecOps 是 Rec SDK 基于 Ascend C 开发的推荐场景自定义算子集，为各框架组件（tf_rec_v1、tf_rec_v2、torch_rec_v1、torch_rec_v2）提供基础算子能力。

## 基本概念

| 术语 | 说明 |
|------|------|
| **Jagged Tensor** | 一种不规则张量，各行长度可以不同，常用于变长序列场景 |
| **dim** | Dimension，指张量的维度 |
| **LayerNorm** | 层归一化操作，对隐藏层进行均值方差归一化 |
| **Dropout** | 训练时随机丢弃部分神经元输出的正则化技术 |
| **Silu** | Sigmoid Linear Unit 激活函数，公式为 `x * sigmoid(x)` |
| **GQA** | Grouped Query Attention，分组查询注意力机制 |
| **HSTU** | Hierarchical Sparse Transformer Unit，分层稀疏 Transformer 计算单元，用于推荐场景的注意力机制 |
| **Alpha-Fuxi** | 推荐场景模型名称，HSTU-Fuxi 算子基于该模型实现注意力机制 |
| **Paged Attention** | 分页注意力机制，将注意力计算的键值对分页管理，用于高效处理变长序列，节省内存访问开销 |

## 算子列表

| 算子名称 | 类型 | 功能介绍 | 支持硬件 |
|---------|------|---------|---------|
| [concat_jagged_tensor](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor/v220/README.md) | 融合算子 | 将两个变长 Tensor 按位置偏移在第二维拼接成一个完整 Tensor。输入：待拼接的两个变长 Tensor、每行的起始位置列表；输出：拼接后的完整 Tensor。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [concat_jagged_tensor_grad](../../../cust_op/ascendc_op/ai_core_op/concat_jagged_tensor_grad/v220/README.md) | 融合算子 | 将一个完整 Tensor 按位置偏移切分成两个变长 Tensor（concat_jagged_tensor 的反向操作）。输入：待切分的 Tensor、分割位置信息；输出：切分后的两个变长 Tensor。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [in_linear_silu](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu/v220/README.md) | 融合算子 | 将融合特征经线性变换+Silu激活后拆分为 User、Value、Query、Key 四个向量（用于 HSTU 注意力机制的前处理）。输入：融合特征、权重、偏置、分组长度；输出：User 向量、Value 向量、Query 向量、Key 向量。 | ascend910b, ascend910_93, ascend950 |
| [in_linear_silu_backward](../../../cust_op/ascendc_op/ai_core_op/in_linear_silu_backward/v220/README.md) | 融合算子 | in_linear_silu 反向传播：计算融合特征、权重、偏置的梯度。输入：融合特征、权重、偏置、User/Value/Query/Key 的梯度；输出：融合特征梯度、权重梯度、偏置梯度。 | ascend910b, ascend910_93, ascend950 |
| [ln_mul](../../../cust_op/ascendc_op/ai_core_op/ln_mul/v220/README.md) | 融合算子 | LayerNorm + Multiply 融合：对输入张量做归一化，再与缩放系数和偏置计算，最后与另一个张量相乘。输入：待归一化张量、相乘张量、缩放系数、偏置；输出：归一化结果与相乘张量的乘积。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [norm_multiply_dropout](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout/v220/README.md) | 融合算子 | LayerNorm + Multiply + Dropout 融合操作：对输入做归一化，与另一张量相乘后进行 Dropout。输入：待归一化张量、相乘张量、归一化参数、Dropout 比例；输出：归一化+相乘+Dropout 后的结果。 | ascend910b, ascend910_93, ascend950 |
| [norm_multiply_dropout_backward](../../../cust_op/ascendc_op/ai_core_op/norm_multiply_dropout_backward/v220/README.md) | 融合算子 | norm_multiply_dropout 反向传播：计算各输入张量的梯度。输入：输出梯度、待归一化张量、相乘张量、归一化参数、Dropout 比例；输出：相乘张量梯度、待归一化张量梯度、归一化参数梯度。 | ascend910b, ascend910_93, ascend950 |
| [token_mixing](../../../cust_op/ascendc_op/ai_core_op/token_mixing/v220/README.md) | 融合算子 | Token 混合归一化：将张量与其转置相加后做 LayerNorm。输入：三维张量、缩放系数、偏置；输出：归一化结果。 | ascend910b, ascend950 |
| [disentangle_attention](../../../cust_op/ascendc_op/ai_core_op/disentangle_attention/v220/README.md) | 注意力算子 | 实现 DeBERTa 模型中的解耦注意力机制，将内容注意力与位置注意力分离计算。输入：Query/Key/Value 向量、位置编码的 Key/Query、相对位置索引、注意力掩码；输出：注意力输出、注意力概率矩阵、注意力权重。 | ascend910b, ascend910_93, ascend950 |
| [hstu_v2](../../../cust_op/ascendc_op/ai_core_op/hstu_v2/BWD_README.md) | 注意力算子 | HSTU V2 反向传播：计算 Query、Key、Value 及相对位置注意力的梯度。输入：输出梯度、Query、Key、Value、序列偏移量、相对位置注意力；输出：Query梯度、Key梯度、Value梯度、位置偏置梯度。 | ascend950 |
| [hstu_dense_backward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward/v220/README.md) | 注意力算子 | HSTU 反向传播：计算 Query、Key、Value 和注意力偏置的梯度。输入：输出梯度、Query、Key、Value、mask、attn_bias；输出：Query梯度、Key梯度、Value梯度、attn_bias梯度。 | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_backward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_backward_fuxi/v220/README.md) | 注意力算子 | HSTU-Fuxi 反向传播：计算 Query、Key、Value、时间戳偏置、位置偏置的梯度。输入：输出梯度、Query、Key、Value、时间戳偏置梯度、位置偏置梯度；输出：Query梯度、Key梯度、Value梯度、时间戳偏置梯度、位置偏置梯度。 | ascend910b, ascend910_93, ascend950 |
| [hstu_dense_forward](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward/README.md) | 注意力算子 | HSTU 前向注意力：实现推荐场景注意力机制，支持 GQA 和 dim 不等特性。输入：Query、Key、Value、mask、attn_bias；输出：注意力输出。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [hstu_dense_forward_fuxi](../../../cust_op/ascendc_op/ai_core_op/hstu_dense_forward_fuxi/v220/README.md) | 注意力算子 | HSTU-Fuxi 前向注意力：实现 Alpha-Fuxi 模型注意力，支持时间戳偏置、位置偏置三种布局。输入：Query、Key、Value、时间戳偏置、位置偏置、mask；输出：注意力输出。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [relative_attn_bias_backward](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_backward/v220/README.md) | 注意力算子 | 相对位置偏置反向传播：计算 HSTU 模型中时间戳偏置参数的梯度。输入：时间戳偏置梯度、时间戳桶索引、桶数量；输出：时间戳权重梯度。 | ascend910b, ascend910_93, ascend950 |
| [relative_attn_bias_pos](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_pos/v220/README.md) | 注意力算子 | 相对位置偏置前向（position 部分）：计算 HSTU 模型中位置偏置。输入：相对位置编码、标识矩阵、有效长度；输出：位置偏置。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [relative_attn_bias_time](../../../cust_op/ascendc_op/ai_core_op/relative_attn_bias_time/v220/README.md) | 注意力算子 | 相对位置偏置前向（timestamp 部分）：计算 HSTU 模型中时间戳偏置。输入：时间戳、时间戳权重、桶划分因子；输出：时间戳偏置、桶索引。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [fused_lazy_adam](../../../cust_op/ascendc_op/ai_core_op/fused_lazy_adam/v220/README.md) | 优化器算子 | LazyAdam 优化器融合：计算并更新一阶矩估计、二阶矩估计、参数。输入：梯度、索引、一阶矩、二阶矩、学习率；输出：更新后的参数。 | ascend910b, ascend910_93, ascend950 |
| [fused_sgd](../../../cust_op/ascendc_op/ai_core_op/fused_sgd/v220/README.md) | 优化器算子 | SGD 优化器融合：计算并更新参数。输入：梯度、索引、学习率；输出：更新后的参数。 | ascend910b, ascend910_93, ascend950 |
| [gather_for_rank1](../../../cust_op/ascendc_op/ai_core_op/gather_for_rank1/v220/README.md) | 张量算子 | 一维索引选择：根据索引从一维张量中选取元素。输入：一维数据、索引列表；输出：按索引选取的元素。 | ascend910b, ascend910_93, ascend310p, ascend950 |
| [index_select_for_rank1_backward](../../../cust_op/ascendc_op/ai_core_op/index_select_for_rank1_backward/v220/README.md) | 张量算子 | 一维索引选择反向传播：聚合梯度到原始张量。输入：输出梯度、原始数据、索引；输出：原始数据梯度。 | ascend910b, ascend910_93, ascend950 |
| [multislice_concat](../../../cust_op/ascendc_op/ai_core_op/multislice_concat/v220/README.md) | 张量算子 | 二维切片拼接：按位置和长度切片并拼接成多个张量。输入：二维张量、切片配置；输出：拼接后的多个张量。 | ascend910b, ascend950 |
| [cust_op_by_addr](../../../cust_op/ascendc_op/ai_core_op/cust_op_by_addr/c310/README.md) | 查表算子 | 地址查表嵌入查询：用地址查找嵌入向量，替换 tf.gather，支持动态扩容。输入：地址列表、嵌入维度；输出：嵌入向量。 | ascend910b, ascend910_93, ascend950 |
| [lccl](../../../cust_op/ascendc_op/ai_core_op/lccl/v220/README.md) | 通信算子 | 集合通信：多卡间直接访问对端显存进行 AllToAll/AllUss/GatherAll 操作。输入：分布式张量、通信矩阵、共享内存；输出：聚合后的张量。 | ascend910b, ascend910_93 |
| [pcie_through](../../../cust_op/ascendc_op/ai_core_op/pcie_through/v220/README.md) | 通信算子 | PCIe 直通：大数据量 host-device 间传输，提升换入换出性能。输入：换入索引、换出索引、表数据、共享内存地址；输出：执行状态。 | ascend910b |
| [reverse_sequence](../../../cust_op/ascendc_op/ai_core_op/reverse_sequence/v220/README.md) | 序列算子 | 序列逆序：按指定长度对第二维进行逆序。输入：三维数据、每个序列的长度；输出：逆序后的数据。 | ascend910b, ascend910_93, ascend950 |

## 算子目录结构

```text
cust_op/ascendc_op/ai_core_op/
# ===== 融合算子 =====
├── concat_jagged_tensor/                              # Jagged Tensor 拼接
├── concat_jagged_tensor_grad/                         # Jagged Tensor 拼接反向
├── in_linear_silu/                                    # Linear + Silu 融合
├── in_linear_silu_backward/                           # Linear + Silu 反向
├── ln_mul/                                            # LayerNorm + Multiply 融合
├── norm_multiply_dropout/                             # Norm + Multiply + Dropout 融合
├── norm_multiply_dropout_backward/                    # Norm + Multiply + Dropout 反向
├── token_mixing/                                      # Token 混合归一化

# ===== 注意力算子 =====
├── disentangle_attention/                             # 解耦注意力 (DeBERTa)
├── hstu_dense_forward/                                # HSTU 前向
├── hstu_dense_backward/                               # HSTU 反向
├── hstu_dense_forward_fuxi/                           # HSTU-Fuxi 前向
├── hstu_dense_backward_fuxi/                          # HSTU-Fuxi 反向
├── hstu_v2/                                           # HSTU V2
├── relative_attn_bias_pos/                            # 相对位置偏置 (position)
├── relative_attn_bias_time/                           # 相对位置偏置 (timestamp)
├── relative_attn_bias_backward/                       # 相对位置偏置反向

# ===== 优化器算子 =====
├── fused_lazy_adam/                                   # LazyAdam 优化器
├── fused_sgd/                                         # SGD 优化器

# ===== 张量算子 =====
├── gather_for_rank1/                                  # 一维 index_select
├── index_select_for_rank1_backward/                   # index_select 反向
├── multislice_concat/                                 # 多切片拼接

# ===== 查表算子 =====
├── cust_op_by_addr/                                   # 地址查表（嵌入表查询）

# ===== 通信算子 =====
├── lccl/                                              # 集合通信 (AllToAll/AllUss/GatherAll)
├── pcie_through/                                      # PCIe 直通数据传输

# ===== 序列算子 =====
└── reverse_sequence/                                  # 序列逆序
```

## 版本配套

| 框架组件 | 基础框架 | 适配状态 | 支持算子 |
|----------|---------|---------|---------|
| tf_rec_v1 | TensorFlow | 非全下沉 | 基础算子集（查表、优化器、通信） |
| tf_rec_v2 | TensorFlow | 全下沉 | 全部算子（支持动态扩容） |
| torch_rec_v1 | PyTorch + TorchRec | 非全下沉 | 基础算子集（查表、优化器、通信） |
| torch_rec_v2 | PyTorch + TorchRec | 全下沉 | 全部算子（支持 Paged Attention） |

> **说明**：
>
> - **非全下沉**：指部分计算任务在 NPU 上执行，部分在 CPU 上执行的混合模式，适用于基础推荐场景
> - **全下沉**：指所有计算任务都下沉到 NPU 上执行，以获得更好的性能，支持高级特性
> - 各框架组件的具体安装和使用说明，请参见对应文档：
>   - [tf_rec_v1 文档](../tensorflow/tf_rec_v1/recsdk_tf_installation_guide.md)
>   - [tf_rec_v2 文档](../tensorflow/tf_rec_v2/recsdk_tf_installation_guide.md)
>   - [torch_rec_v1 文档](../torch/torch_rec_v1/recsdk_torch_installation_guide.md)
>   - [torch_rec_v2 文档](../torch/torch_rec_v2/recsdk_torch_installation_guide.md)
