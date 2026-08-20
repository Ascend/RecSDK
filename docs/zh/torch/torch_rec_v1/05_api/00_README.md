# API列表

注：当前API列表是从接口类型维度进行展示，如需从功能特性维度查看API使用，请参见[功能特性介绍](../04_migration_and_training/migration_and_training.md#functional_features_description)。

## API概览

- [接口说明](01_api_description.md)
- [创表接口](02_table_creation_apis.md)
- [数据接口](03_data_apis.md)
- [优化器接口](04_optimizers_apis.md)
- [分表接口](05_subtable_apis.md)
- [pipeline接口](06_pipeline_apis.md)
- [多级缓存管理接口](07_multilevel_cache_management_apis.md)
- [准入淘汰管理接口](08_access_and_elimination_management_apis.md)
- [自定义算子](09_specialized_operator.md)

## API详情

注：后续API中带有`（TorchRec）`标记的表示该API为TorchRec原生API，其他API为Rec SDK Torch提供。

### 创表接口

|API | 功能描述|
|--|--|
|[HashEmbeddingBagConfig](02_table_creation_apis.md#hashembeddingbagconfig)|HashEmbeddingBagCollection的入参，用于配置表的大小、dim、数据类型等。|
|[HashEmbeddingBagCollection](02_table_creation_apis.md#hashembeddingbagcollection)|创建带Pooling和哈希映射的稀疏表对象。|
|[EmbCacheEmbeddingBagConfig](02_table_creation_apis.md#embcacheembeddingbagconfig)|EmbCacheEmbeddingBagCollection的入参，用于配置表的大小、dim、数据类型等。|
|[EmbCacheEmbeddingBagCollection](02_table_creation_apis.md#embcacheembeddingbagcollection)|创建带pooling、哈希映射和多级缓存的稀疏表对象。|
|[EmbCacheEmbeddingConfig](02_table_creation_apis.md#embcacheembeddingconfig)|EmbCacheEmbeddingCollection的配置类接口，用于配置表的大小、dim、数据类型等。|
|[EmbCacheEmbeddingCollection](02_table_creation_apis.md#embcacheembeddingcollection)|创建带哈希映射和多级缓存的稀疏表对象。|

### 数据接口

|API | 功能描述|
|--|--|
|[JaggedTensor（TorchRec）](03_data_apis.md#jaggedtensortorchrec)|持有稀疏id和特征长度的类，用于查表。|
|[KeyedJaggedTensor（TorchRec）](03_data_apis.md#keyedjaggedtensortorchrec)|通过引入键（通常是特征名称）来扩展JaggedTensor，以标记不同的特征组。|

### 优化器接口

|API | 功能描述|
|--|--|
|[apply_optimizer_in_backward（TorchRec）](04_optimizers_apis.md#apply_optimizer_in_backwardtorchrec)|指定模型在反向传播时使用的优化器。一般用于指定稀疏表参数反向传播时使用的优化器。|
|[in_backward_optimizer_filter（TorchRec）](04_optimizers_apis.md#in_backward_optimizer_filtertorchrec)|用于过滤掉被指定为backward_optimizer的参数。|
|[KeyedOptimizerWrapper（TorchRec）](04_optimizers_apis.md#keyedoptimizerwrappertorchrec)|用于封装过滤掉表的参数的优化器。|
|[CombinedOptimizer（TorchRec）](04_optimizers_apis.md#combinedoptimizertorchrec)|将多个优化器合并为一个。|

### 分表接口

|API | 功能描述|
|--|--|
|[ShardingEnv（TorchRec）](05_subtable_apis.md#shardingenvtorchrec)|保存分布式相关参数。|
|[Topology（TorchRec）](05_subtable_apis.md#topologytorchrec)|保存分布式环境网络设备拓扑参数。|
|[ParameterConstraints（TorchRec）](05_subtable_apis.md#parameterconstraintstorchrec)|指定分表计划的查询范围。|
|[get_default_hybrid_sharders](05_subtable_apis.md#get_default_hybrid_sharders)|获取分表器。|
|[EmbeddingShardingPlanner（TorchRec）](05_subtable_apis.md#embeddingshardingplannertorchrec)|创建分表计划器，用于搜索最合适的分表计划。|
|[EmbCacheEmbeddingBagCollectionSharder](05_subtable_apis.md#embcacheembeddingbagcollectionsharder)|创建EmbCacheEmbeddingBagCollectionSharder分表器，用于将EmbCacheEmbeddingBagCollection分片到不同的设备上。|
|[EmbCacheEmbeddingCollectionSharder](05_subtable_apis.md#embcacheembeddingcollectionsharder)|创建EmbCacheEmbeddingCollectionSharder分表器，用于将EmbCacheEmbeddingCollection分片到不同的设备上。|
|[DistributedModelParallel（TorchRec）](05_subtable_apis.md#distributedmodelparalleltorchrec)|将传入的Module变为分布式的Module，并执行分表计划。|

### pipeline接口

|API | 功能描述|
|--|--|
|[HybridTrainPipelineSparseDist](06_pipeline_apis.md#hybridtrainpipelinesparsedist)|创建纯显存模式的分布式训练数据流水线调度器。|
|[EmbCacheTrainPipelineSparseDist](06_pipeline_apis.md#embcachetrainpipelinesparsedist)|创建多级缓存模式的分布式训练数据流水线调度器。|

### 多级缓存管理接口

|API | 功能描述|
|--|--|
|[InitializerType](07_multilevel_cache_management_apis.md#initializertype)|权重初始化类型枚举，定义嵌入表权重的初始化方式。|
|[Saver](07_multilevel_cache_management_apis.md#saver)|多级缓存稀疏表保存加载功能类，提供多级缓存稀疏表数据（稀疏表Embedding，Embedding对应的优化器参数等）的保存、加载接口。|

### 准入淘汰管理接口

|API | 功能描述|
|--|--|
|[JaggedTensorWithTimestamp](08_access_and_elimination_management_apis.md#jaggedtensorwithtimestamp)|该接口是一个扩展自JaggedTensor的类，用于表示带有时间戳信息的Jagged Tensor。该类在JaggedTensor的基础上增加了一个\_timestamps属性，存储与values对应的时间戳信息。用于特征淘汰时计算时间。|
|[KeyedJaggedTensorWithTimestamp](08_access_and_elimination_management_apis.md#keyedjaggedtensorwithtimestamp)|该接口是一个扩展自KeyedJaggedTensor的类，用于表示带有时间戳信息的Keyed Jagged Tensor。该类在KeyedJaggedTensor的基础上增加了一个\_timestamps属性，存储与values对应的时间戳信息。用于特征淘汰时计算时间。|
|[AdmitAndEvictPolicyType](08_access_and_elimination_management_apis.md#admitandevictpolicytype)|准入淘汰策略类型枚举，定义嵌入表特征准入和淘汰的策略类型。|
|[ShowClickParams](08_access_and_elimination_management_apis.md#showclickparams)|该接口表示基于展示点击（show/click）策略的参数配置。该类提供了配置展示点击准入和淘汰功能的参数，允许用户根据展示次数和点击次数控制特征的准入和淘汰行为。|
|[AdmitAndEvictConfig](08_access_and_elimination_management_apis.md#admitandevictconfig)|该接口表示单个嵌入表的准入和淘汰配置。该类提供了配置嵌入表特征准入和淘汰功能的参数，允许用户根据特定条件控制特征的准入和淘汰行为。支持两种策略类型：基于计数的策略（POLICY_COUNT）和基于展示点击的策略（POLICY_SHOWCLICK）。|

### 自定义算子

Rec SDK Torch提供了部分自定义算子，用于处理稀疏表数据和加速模型训练，详情请参考[自定义算子](09_specialized_operator.md)中的自定义算子列表。
