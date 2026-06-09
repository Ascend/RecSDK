# 接口说明<a name="ZH-CN_TOPIC_0000002302229620"></a>

## 概述

Rec SDK Torch 通过 Python 接口进行应用开发，从代码调用角度上来说所有 Python 侧接口都可以被调用。

本章节仅列出业务提供的对外接口，其余未进行说明的接口用户请勿直接调用。

Rec SDK Torch 是基于 TorchRec 接口的扩展，Rec SDK Torch 的接口依赖于 TorchRec 提供的类和方法，但是并不能支持 TorchRec 的所有功能。本章节介绍基于 Rec SDK Torch 搭建模型时使用的 TorchRec 接口限制。

>[!NOTICE]
>
>1. 由于 TorchRec 原生接口不在 Rec SDK Torch 的管理范围内，因此将不会对后续接口中涉及到的 TorchRec 原生接口做参数合法性校验，需用户自行保证参数正确性。
>2. 当前 API 列表是从接口类型维度进行展示，如需从功能特性维度查看 API 使用，请参见[功能特性介绍](../migration_and_training.md#functional_features_description)。
>3. 部分API参数说明中包含“不支持用户自定义”的描述，表示该参数不支持用户自定义，只能使用默认值，传入非默认值时将抛出异常。

## API 分类概览

Rec SDK Torch 的 API 按接口类型分为以下类别，各文档中带有"（TorchRec）"标记的接口为 TorchRec 开源接口，其余为 Rec SDK Torch 自有接口：

| 类别 | 文档 | 说明 |
|------|------|------|
| 创表接口 | [table_creation_apis.md](table_creation_apis.md) | 稀疏表的配置与创建，包含纯显存模式和多级缓存模式两类创表方式。 |
| 数据接口 | [data_apis.md](data_apis.md) | 稀疏数据的表示和封装，用于向稀疏表传入查询 ID 和特征信息。 |
| 分表接口 | [subtable_apis.md](subtable_apis.md) | 分布式环境下稀疏表的分片策略和执行稀疏表分表。 |
| 优化器接口 | [optimizers_apis.md](optimizers_apis.md) | 稀疏表参数的优化器配置，支持梯度累积功能。 |
| pipeline 接口 | [pipeline_apis.md](pipeline_apis.md) | 训练流水线的创建和执行，支持纯显存和多级缓存两种流水线模式。 |
| 多级缓存管理 | [multilevel_cache_management_apis.md](multilevel_cache_management_apis.md) | 多级缓存模式下的模型保存/加载及权重初始化类型。 |
| 准入淘汰管理 | [access_and_elimination_management_apis.md](access_and_elimination_management_apis.md) | 稀疏表特征的准入和淘汰策略配置。 |
| 自定义算子 | [specialized_operator.md](specialized_operator.md) | NPU 加速的自定义算子，部分已绑定到开源 API。 |

## 推荐阅读顺序

对于首次使用 Rec SDK Torch 的用户，建议按以下顺序阅读 API 文档：

1. **本页（接口说明）** — 了解 API 整体结构和核心概念
2. **[创表接口](table_creation_apis.md)** — 理解稀疏表的配置和创建方式，这是搭建模型的第一步
3. **[数据接口](data_apis.md)** — 掌握如何向稀疏表传入查询数据
4. **[分表接口](subtable_apis.md)** — 了解分布式环境下如何分片稀疏表
5. **[优化器接口](optimizers_apis.md)** — 配置稀疏表参数的优化器
6. **[pipeline 接口](pipeline_apis.md)** — 创建训练流水线，串联查表、前向/反向传播
7. **[多级缓存管理](multilevel_cache_management_apis.md)** / **[准入淘汰管理](access_and_elimination_management_apis.md)** — 进阶功能，按需阅读
8. **[自定义算子](specialized_operator.md)** — 了解可用的 NPU 加速算子

## 核心术语

| 术语 | 说明 |
|------|------|
| 稀疏表（Embedding Table） | 用于存储大规模稀疏特征（如用户 ID、物品 ID）的 Embedding 向量的数据结构。 |
| embedding_dim | 稀疏表的列数，即每个特征的 Embedding 向量维度，取值范围 [8, 4096] 且需为 8 的倍数。 |
| num_embeddings | 稀疏表的行数，即最大特征数量，取值范围 [1, 10 亿]。 |
| Pooling | 将同一特征的多个 Embedding 向量聚合为一个向量的操作，支持 SUM（求和）、MEAN（取平均）、NONE（不做 Pooling）。 |
| EBC（EmbeddingBagCollection） | 带 Pooling 的稀疏表，查表后自动对同一特征的多条 Embedding 做聚合。纯显存模式使用 HashEmbeddingBagCollection，多级缓存模式使用 EmbCacheEmbeddingBagCollection。 |
| EC（EmbeddingCollection） | 不带 Pooling 的稀疏表，查表后返回原始 Embedding 列表。多级缓存模式使用 EmbCacheEmbeddingCollection。 |
| 纯显存模式 | 稀疏表数据全部存放在 NPU 显存中，通过 HybridTrainPipelineSparseDist 进行训练。 |
| 多级缓存模式 | 稀疏表数据分布在 CPU 内存和 NPU 显存之间，通过 EmbCacheTrainPipelineSparseDist 进行训练，支持更大规模的 Embedding 表。 |
| row_wise | 按行分表策略，将稀疏表的不同行分配到不同 NPU 卡上。 |
| data_parallel | 数据并行分表策略，每张 NPU 卡保留完整的稀疏表副本。 |
| meta 设备 | PyTorch 的虚拟设备类型，用于延迟实际内存分配。在创建稀疏表时指定 "meta" 可先构建模型结构，待 DistributedModelParallel 分表后再实际分配内存。 |
| 准入（Admit） | 控制新特征 ID 是否被加入到稀疏表中，可通过重复次数阈值或展示/点击分数阈值来过滤低频特征。 |
| 淘汰（Evict） | 将长时间未访问或分数较低的特征 ID 从稀疏表中移除，以控制表的内存占用。 |
| JaggedTensor | 持有稀疏 ID 和特征长度的数据结构，每个样本的特征 ID 数量可以不同。 |
| KeyedJaggedTensor | 在 JaggedTensor 基础上增加特征名称键（key），用于区分不同特征组。 |
| pipeline | 训练流水线，用于迭代数据集并进行训练。可将训练流程中部分不存在依赖关系的操作并行执行，提高训练效率。 |

## 接口调用约定

### 导入路径

文档中的代码示例使用以下导入路径约定：

- **纯显存模式接口**：从 `hybrid_torchrec` 导入，如 `from hybrid_torchrec import HashEmbeddingBagConfig`。
- **多级缓存模式接口**：从 `torchrec_embcache` 导入，如 `from torchrec_embcache.distributed import EmbCacheEmbeddingBagConfig`。
- **TorchRec 原生接口**：从 `torchrec` 导入，如 `from torchrec.distributed.planner import EmbeddingShardingPlanner`。

### 返回值约定

- 成功：返回对应的对象或 None
- 失败：抛出异常。具体异常类型和原因请参考各 API 的参数约束说明。
