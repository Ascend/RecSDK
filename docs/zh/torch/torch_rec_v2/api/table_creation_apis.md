# 创表接口<a name="ZH-CN_TOPIC_0000002336148893"></a>

## EmbeddingConfig<a name="ZH-CN_TOPIC_0000002336148933"></a>

>[!NOTICE] 须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

EmbeddingCollection的入参，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```cpp
@dataclass
class EmbeddingConfig:
    def __init__(**kwargs):
```

**参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|num_embeddings|int|必选|稀疏表的行数。取值范围：[1, 10亿]。其中最小值需要满足：≥使用的卡数。|
|embedding_dim|int|必选|稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。|
|name|str|必选|稀疏表的名称。只能包含数字、字母和下划线。长度范围：[1,4096]。|
|data_type|torchrec.types.DataType|可选|稀疏表的数据类型。仅支持默认值为DataType.FP32。|
|feature_names|List[str]|必选|稀疏表查询的特征名称。只能包含数字、字母和下划线。|
|weight_init_max|float|可选|权重初始化最大值。仅支持默认值为None，不支持用户自定义。|
|weight_init_min|float|可选|权重初始化最小值。仅支持默认值为None，不支持用户自定义。|
|num_embeddings_post_pruning|int|可选|推理剪枝后稀疏表数量。仅支持默认值为None，不支持用户自定义。|
|init_fn|Callable|可选|初始化函数。支持传入nn.Parameter类型的函数。用户需自行保证该函数的正确性。默认值为None。|
|need_pos|bool|可选|位置权重。仅支持默认值为False，不支持用户自定义。|


**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。


## EmbeddingCollection<a name="ZH-CN_TOPIC_0000002302389408"></a>

>[!NOTICE] 须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

创建单机表对象。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbeddingCollection:
    def __init__(**kwargs):
```

**参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|tables|List[EmbeddingConfig]|必选|稀疏表配置文件列表。<p>参数范围参考EmbeddingConfig。</p>|
|device|Optional[torch.device]|可选|稀疏表的设备。默认为torch.device("cpu")。</ul></li><li>如果为torch.device取值范围：<ul><li>torch.device("npu")：npu设备。</li><li>torch.device("meta")：meta设备。</li><li>torch.device("cpu")：cpu设备。cpu设备不支持分布式表，只支持单机表。</li></ul></li>|
|need_indices|bool|可选|是否需要索引，默认值为False。|


**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```cpp
ec = EmbeddingCollection(device="npu", tables=table_configs)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。

