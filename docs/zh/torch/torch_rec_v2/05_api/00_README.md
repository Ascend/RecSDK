# API列表

## API概览

- [接口说明](01_api_description.md)
- [创表接口](02_table_creation_apis.md)
- [数据接口](03_data_apis.md)
- [分表接口](04_subtable_apis.md)
- [保存与加载接口](05_dump_load_apis.md)
- [自定义算子](06_specialized_operator.md)
- [API支持度](07_api_compatibility.md)

## API详情

注：后续API中带有`（TorchRec）`标记的表示该API为TorchRec原生API，其他API为Rec SDK Torch提供。

### 创表接口

|API | 功能描述|
|--|--|
|[EmbeddingConfig（TorchRec）](02_table_creation_apis.md#embeddingconfig)|EmbeddingCollection的入参，用于配置表的大小、dim、数据类型等。|
|[EmbeddingCollection（TorchRec）](02_table_creation_apis.md#embeddingcollection)|创建单机表对象。|

### 数据接口

|API | 功能描述|
|--|--|
|[JaggedTensor（TorchRec）](03_data_apis.md#jaggedtensortorchrec)|持有稀疏id和特征长度的类，用于查表。|
|[KeyedJaggedTensor（TorchRec）](03_data_apis.md#keyedjaggedtensortorchrec)|通过引入键（通常是特征名称）来扩展JaggedTensor，以标记不同的特征组。|

### 分表接口

|API | 功能描述|
|--|--|
|[ShardingEnv（TorchRec）](04_subtable_apis.md#shardingenvtorchrec)|保存分布式相关参数。|
|[Topology（TorchRec）](04_subtable_apis.md#topologytorchrec)|保存分布式环境网络设备拓扑参数。|
|[DynamicEmbeddingCollectionSharder](04_subtable_apis.md#ZH-CN_TOPIC_0000002461958569)|创建不带Pooling的动态嵌入表分表器。|
|[DynamicEmbeddingBagCollectionSharder](04_subtable_apis.md#ZH-CN_TOPIC_0000002461958570)|创建带Pooling的动态嵌入表分表器。|
|[DynamicEmbParameterConstraints](04_subtable_apis.md#ZH-CN_TOPIC_0000002336148869)|配置动态嵌入表的分表约束。|
|[DynamicEmbTableOptions](04_subtable_apis.md#ZH-CN_TOPIC_0000002338277269)|配置动态嵌入表参数。|
|[DynamicEmbInitializerMode](04_subtable_apis.md#ZH-CN_TOPIC_0000002304198202)|动态嵌入表中嵌入向量的初始化模式枚举。|
|[DynamicEmbInitializerArgs](04_subtable_apis.md#ZH-CN_TOPIC_0000002508694909)|配置动态嵌入表中嵌入向量的初始化参数。|
|[DynamicEmbScoreStrategy](04_subtable_apis.md#dynamicembscorestrategy)|动态嵌入表的评分策略枚举，用于稀疏特征淘汰。|
|[EmbOptimType](04_subtable_apis.md#ZH-CN_TOPIC_0000002476574952)|动态嵌入表的优化器类型枚举。|
|[DynamicEmbeddingEnumerator](04_subtable_apis.md#dynamicembeddingenumerator)|枚举动态嵌入表的分片方案。|
|[DynamicEmbeddingShardingPlanner](04_subtable_apis.md#dynamicembeddingshardingplanner)|创建动态嵌入表分表计划器，用于搜索分表计划。|
|[DynamicEmbCheckMode](04_subtable_apis.md#ZH-CN_TOPIC_0000002396563024)|动态嵌入表索引插入的安全检查模式枚举。|

### 保存与加载接口

|API | 功能描述|
|--|--|
|[DynamicEmbDump](05_dump_load_apis.md#ZH-CN_TOPIC_0000002428320084)|将模型中的动态嵌入表数据并行保存至文件系统。|
|[set_score](05_dump_load_apis.md#ZH-CN_TOPIC_0000002430202770)|设置模型中动态嵌入表的分数。|
|[get_score](05_dump_load_apis.md#ZH-CN_TOPIC_0000002430202771)|获取模型中动态嵌入表的分数。|
|[DynamicEmbLoad](05_dump_load_apis.md#ZH-CN_TOPIC_0000002430202769)|从文件系统加载动态嵌入表数据并写入模型。|
|[incremental_dump](05_dump_load_apis.md#ZH-CN_TOPIC_0000002430202869)|根据分数阈值增量导出模型中的动态嵌入表数据。|

### 自定义算子

Rec SDK Torch提供了部分自定义算子，用于处理动态稀疏表数据和加速模型训练，详情请参考[自定义算子](06_specialized_operator.md)中的自定义算子列表。
