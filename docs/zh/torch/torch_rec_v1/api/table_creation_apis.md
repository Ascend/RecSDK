# 创表接口<a name="ZH-CN_TOPIC_0000002336148893"></a>

## HashEmbeddingBagConfig<a name="ZH-CN_TOPIC_0000002336148933"></a>

**功能描述<a name="section634582619155"></a>**

HashEmbeddingBagCollection的入参，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```cpp
@dataclass
class HashEmbeddingBagConfig:
 def __init__(**kwargs):
```

**参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|num_embeddings|int|必选|稀疏表的行数。取值范围：[1, 10亿]。|
|embedding_dim|int|必选|稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。|
|name|str|必选|稀疏表的名称。只能包含数字、字母和下划线。|
|data_type|torchrec.types.DataType|可选|稀疏表的数据类型。仅支持默认值为DataType.FP32。|
|feature_names|List[str]|必选|稀疏表查询的特征名称。只能包含数字、字母和下划线。|
|weight_init_max|float|可选|仅支持默认值为None或1.0，不支持用户自定义。|
|weight_init_min|float|可选|仅支持默认值为None或0.0，不支持用户自定义。|
|num_embeddings_post_pruning|int|可选|仅支持默认值为None，不支持用户自定义。|
|init_fn|Callable|可选|支持传入nn.Parameter类型的函数。用户需自行保证该函数的正确性。默认值为None。|
|need_pos|bool|可选|仅支持默认值为False，不支持用户自定义。|
|pooling|torchrec.modules.embedding_configs.PoolType|可选|pool操作的类型。取值范围：<li>SUM：求和。</li><li>MEAN：取平均。</li><li>NONE：不做pool操作。</li>默认为SUM。|


**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。


## HashEmbeddingBagCollection<a name="ZH-CN_TOPIC_0000002302389408"></a>

**功能描述<a name="section634582619155"></a>**

创建带Pooling和哈希映射的单机表对象。

**函数原型<a name="section1483104721911"></a>**

```cpp
class HashEmbeddingBagCollection:
 def __init__(**kwargs):
```

**参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|tables|List[HashEmbeddingBagConfig \| EmbeddingBagConfig]|必选|稀疏表配置文件列表。列表长度的取值范围：[1, 10000]。<p>参数范围参考HashEmbeddingBagConfig。</p>|
|is_weighted|bool|可选|仅支持默认值为False。|
|device|str或者torch.device|可选|稀疏表的设备。默认为torch.device("cpu")。<li>如果为str取值范围：<ul><li>"npu"：npu设备。</li><li>"meta"：meta设备。</li><li>"cpu"：cpu设备。cpu设备不支持分布式表，只支持单机表。</li></ul></li><li>如果为torch.device取值范围：<ul><li>torch.device("npu")：npu设备。</li><li>torch.device("meta")：meta设备。</li><li>torch.device("cpu")：cpu设备。cpu设备不支持分布式表，只支持单机表。</li></ul></li>|


**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```cpp
ebc = HashEmbeddingBagCollection(device="npu", tables=table_configs)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。


## EmbCacheEmbeddingBagConfig<a name="ZH-CN_TOPIC_0000002396403104"></a>

**功能描述<a name="section634582619155"></a>**

EmbCacheEmbeddingBagConfig是EmbCacheEmbeddingBagCollection的配置类接口，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheEmbeddingBagConfig:
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选| 说明                                                       |
|--|--|--|----------------------------------------------------------|
|num_embeddings|int|必选| 稀疏表的行数。取值范围：[1, 10亿]。使用row_wise时至少一张稀疏表的行数≥使用的卡数。        |
|embedding_dim|int|必选| 稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。                         |
|name|str|必选| 稀疏表的名称。只能包含数字、字母和下划线。                                    |
|data_type|torchrec.types.DataType|可选| 稀疏表的数据类型。仅支持默认值为DataType.FP32。                           |
|feature_names|List[str]|必选| 稀疏表查询的特征名称。只能包含数字、字母和下划线。                                |
|weight_init_max|float|可选| 仅支持默认值为None或1.0，不支持用户自定义。                                |
|weight_init_min|float|可选| 仅支持默认值为None或0.0，不支持用户自定义。                                |
|num_embeddings_post_pruning|int|可选| 仅支持默认值为None，不支持用户自定义。                                    |
|init_fn|Callable|可选| 支持传入nn.Parameter类型的函数。用户需自行保证该函数的正确性。默认值为None。           |
|need_pos|bool|可选| 仅支持默认值为False，不支持用户自定义。                                   |
|pooling|torchrec.modules.embedding_configs.PoolType|可选| pool操作的类型。取值范围：<li>SUM：求和。</li><li>MEAN：取平均。</li>默认为SUM。 |
|weight_init_mean|float|可选| 权重初始化均值，用于UNIFORM初始化类型，默认值0.0。                           |
|weight_init_stddev|float|可选| 权重初始化标准差，用于UNIFORM初始化类型，默认值0.05。                         |
|initializer_type|InitializerType|可选| 权重初始化类型，支持LINEAR、TRUNCATED_NORMAL、UNIFORM，默认值LINEAR。     |
|admit_and_evict_config|AdmitAndEvictConfig|可选| 特征准入和淘汰配置，默认不启用准入和淘汰功能。预留参数，当前暂不支持。                      |
|is_incremental|bool|可选| 开启增量保存和加载功能                      |


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheEmbeddingBagConfig对象。
-   失败：抛出异常。


## EmbCacheEmbeddingBagCollection<a name="ZH-CN_TOPIC_0000002396562992"></a>

**功能描述<a name="section634582619155"></a>**

创建带pooling、哈希映射和多级缓存的单机表对象。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheEmbeddingBagCollection:
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|tables|List[EmbCacheEmbeddingBagConfig\|EmbeddingConfig]|必选|稀疏表配置文件列表。列表长度的取值范围为[1，10000]。|
|world_size|int|必选|分布式训练world_size大小，取值范围为[1，10000]。|
|batch_size|int|必选|批次大小，取值范围为[1，102400]。|
|multi_hot_sizes|List[int]|必选|每个特征的多热编码大小列表。该参数列表的长度必须与tables的列表长度相同，取值范围为[1，10000]；列表中多热编码大小的取值范围为[1，102400]。|
|is_weighted|bool|可选|仅支持默认值False。|
|need_accumulate_offset|bool|可选|是否需要累积偏移量，默认值True。|
|device|torch.device|可选|默认为CPU，计算设备和HashEmbeddingBagCollection一致。|
|embedding_optimizer_cls|Type[torch.optim.Optimizer]|可选|嵌入优化器类型，默认值torch.optim.Adagrad。取值范围为：<li>torch.optim.Adagrad：表示Adagrad优化器。</li><li>torch.optim.Adam：表示Adam优化器。</li><li>torch.optim.SGD：表示SGD优化器。</li>|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheEmbeddingBagCollection对象。
-   失败：抛出异常。


## EmbCacheEmbeddingConfig<a name="ZH-CN_TOPIC_0000002430082769"></a>

**功能描述<a name="section634582619155"></a>**

EmbCacheEmbeddingConfig是EmbCacheEmbeddingCollection的配置类接口，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheEmbeddingConfig:
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|num_embeddings|int|必选|稀疏表的行数。取值范围：[1, 10亿]。使用row_wise时至少一张稀疏表的行数≥使用的卡数。|
|embedding_dim|int|必选|稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。|
|name|str|必选|稀疏表的名称。只能包含数字、字母和下划线。|
|data_type|torchrec.types.DataType|可选|稀疏表的数据类型。仅支持默认值为DataType.FP32。|
|feature_names|List[str]|必选|稀疏表查询的特征名称。只能包含数字、字母和下划线。|
|weight_init_max|float|可选|仅支持默认值为None或1.0，不支持用户自定义。|
|weight_init_min|float|可选|仅支持默认值为None或0.0，不支持用户自定义。|
|num_embeddings_post_pruning|int|可选|仅支持默认值为None，不支持用户自定义。|
|init_fn|Callable|可选|支持传入nn.Parameter类型的函数。用户需自行保证该函数的正确性。默认值为None。|
|need_pos|bool|可选|仅支持默认值为False，不支持用户自定义。|
|weight_init_mean|float|可选|权重初始化均值，用于UNIFORM初始化类型，默认值0.0。|
|weight_init_stddev|float|可选|权重初始化标准差，用于UNIFORM初始化类型，默认值0.05。|
|initializer_type|InitializerType|可选|权重初始化类型，支持LINEAR、TRUNCATED_NORMAL、UNIFORM，默认值LINEAR。|
|admit_and_evict_config|AdmitAndEvictConfig|可选|特征准入和淘汰配置，默认不启用准入和淘汰功能。|
|is_incremental|bool|可选| 开启增量保存和加载功能                      |


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheEmbeddingConfig对象。
-   失败：抛出异常。


## EmbCacheEmbeddingCollection<a name="ZH-CN_TOPIC_0000002430202745"></a>

**功能描述<a name="section634582619155"></a>**

创建带哈希映射和多级缓存的单机表对象。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheEmbeddingCollection:
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|tables|List[EmbCacheEmbeddingConfig\|EmbeddingConfig]|必选|稀疏表配置列表。列表长度的取值范围为[1，10000]。|
|world_size|int|必选|分布式训练world_size大小，取值范围为[1，10000]。|
|batch_size|int|必选|批次大小，取值范围为[1，102400]。|
|multi_hot_sizes|List[int]|必选|每个特征的多热编码大小列表。该参数列表的长度必须与tables的列表长度相同，取值范围为[1，10000]；列表中多热编码大小的取值范围为[1，102400]。|
|need_indices|bool|可选|是否需要索引，默认值False。|
|need_accumulate_offset|bool|可选|是否需要累积偏移量，默认值True。|
|device|torch.device|可选|默认为CPU，计算设备和EmbCacheEmbeddingBagCollection一致。|
|embedding_optimizer_cls|Type[torch.optim.Optimizer]|可选|嵌入优化器类型，默认值torch.optim.Adagrad。取值范围为：<li>torch.optim.Adagrad：表示Adagrad优化器。</li><li>torch.optim.Adam：表示Adam优化器。</li><li>torch.optim.SGD：表示SGD优化器。</li>|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheEmbeddingCollection对象。
-   失败：抛出异常。


