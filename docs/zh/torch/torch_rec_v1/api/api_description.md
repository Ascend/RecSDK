# 接口说明<a name="ZH-CN_TOPIC_0000002302229620"></a>

## 概述

Rec SDK Torch 通过 Python 接口进行应用开发，从代码调用角度上来说所有 Python 侧接口都可以被调用。

本章节仅列出业务提供的对外接口，其余未进行说明的接口，用户请勿直接调用。

Rec SDK Torch 是基于 TorchRec 接口的扩展，Rec SDK Torch 的接口依赖于 TorchRec 提供的类和方法，但是并不能支持 TorchRec 的所有功能。本章节介绍基于 Rec SDK Torch 搭建模型时对 TorchRec 接口限制。

> [!NOTICE]
>
> 1. 由于 TorchRec 原生接口不在 Rec SDK Torch 的管理范围内，因此将不会对后续接口中涉及到的 TorchRec 原生接口做参数合法性校验，需用户自行保证参数正确性。
> 2. 当前 API 列表是从接口类型维度进行展示，如需从功能特性维度查看 API 使用，请参见[功能特性介绍](../migration_and_training.md#functional_features_description)。
> 3. 部分API参数说明中包含“不支持用户自定义”的描述，表示该参数不支持用户自定义，只能使用默认值，传入非默认值时将抛出异常。
> 4. 部分API参数说明中包含“仅支持TorchRec v1.2.0版本“等类似说明，表示仅在配套的TorchRec版本中才支持传入该参数。
> 5. 相关术语请参见[核心术语](../introduction.md#core_terms)。

## API 分类概览

Rec SDK Torch 的 API 按接口类型分为以下类别，各文档中带有"（TorchRec）"标记的接口为 TorchRec 开源接口，其余为 Rec SDK Torch 自有接口：

| 类别 | 文档 | 说明 |
|------|------|------|
| 创表接口 | [table_creation_apis.md](table_creation_apis.md) | 稀疏表的配置与创建，包含纯显存模式和多级缓存模式两类创表方式。 |
| 数据接口 | [data_apis.md](data_apis.md) | 稀疏数据的表示和封装，用于向稀疏表传入查询 ID 和特征信息。 |
| 分表接口 | [subtable_apis.md](subtable_apis.md) | 分布式环境下稀疏表的分片策略和分表执行。 |
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

## 接口调用约定

### 导入路径

文档中的代码示例使用以下导入路径约定：

- **纯显存模式接口**：从 `hybrid_torchrec` 导入，如 `from hybrid_torchrec import HashEmbeddingBagConfig`。
- **多级缓存模式接口**：从 `torchrec_embcache` 导入，如 `from torchrec_embcache.distributed import EmbCacheEmbeddingBagConfig`。
- **TorchRec 原生接口**：从 `torchrec` 导入，如 `from torchrec.distributed.planner import EmbeddingShardingPlanner`。

### 返回值约定

- 成功：返回对应的对象或 None
- 失败：抛出异常。具体异常类型和原因请参考各 API 的参数约束说明。
