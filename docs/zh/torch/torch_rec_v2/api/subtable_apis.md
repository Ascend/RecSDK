# 分表接口<a name="ZH-CN_TOPIC_0000002336268665"></a>

## ShardingEnv（TorchRec）<a name="ZH-CN_TOPIC_0000002336148941"></a>

>[!NOTICE] 须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

保存分布式相关参数。

**函数原型<a name="section1483104721911"></a>**

```cpp
class ShardingEnv:
    def __init__(**kwargs):
def from_process_group(cls, pg: dist.ProcessGroup) -> "ShardingEnv":
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|world_size|int|必选|使用的卡数。取值范围：[1，8]|
|rank|int|必选|当前的卡号。取值范围：[0，world_size -1]|
|pg|dist.ProcessGroup|必选|分布式通讯链接。取值范围：只支持backend为hccl和gloo的链接。<div class="notice"><span class="noticetitle">须知</span><div class="notebody">“hccl”在PyTorch里面的backend_name为custom。</div></div>|
|output_dtensor|bool|可选|仅支持默认值为False，不支持用户自定义。|


**使用示例<a name="section193151694205"></a>**

```cpp
import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv
host_gp = dist.new_group(backend="gloo")
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
```


**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## Topology（TorchRec）<a name="ZH-CN_TOPIC_0000002336268737"></a>

>[!NOTICE] 须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

保存分布式环境的参数。

**函数原型<a name="section1483104721911"></a>**

```cpp
class Topology:
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选| 说明                                                |
|--|--|--|---------------------------------------------------|
|world_size|int|必选| 使用的卡数。取值范围：[1，8]                                  |
|compute_device|str|必选| 设备名称。当使用NPU设备时取值为"npu"，即npu设备。                    |
|hbm_cap|int|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|ddr_cap|int|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|local_world_size|int|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|hbm_mem_bw|float|可选| 当使用NPU设备时仅支持默认值为(897 * 1024 * 1024 * 1024 / 1000)，不支持用户自定义。 |
|ddr_mem_bw|float|可选| 当使用NPU设备时仅支持默认值为(51 * 1024 * 1024 * 1024 / 1000)，不支持用户自定义。 |
|hbm_to_ddr_mem_bw|float|可选| 当使用NPU设备时仅支持默认值为(32 * 1024 * 1024 * 1024 / 1000)，不支持用户自定义。 |
|intra_host_bw|float|可选| 当使用NPU设备时仅支持默认值为(600 * 1024 * 1024 * 1024 / 1000)，不支持用户自定义。 |
|inter_host_bw|float|可选| 当使用NPU设备时仅支持默认值为(12.5 * 1024 * 1024 * 1024 / 1000)，不支持用户自定义。 |
|bwd_compute_multiplier|float|可选| 当使用NPU设备时仅支持默认值为2，不支持用户自定义。                                |
|custom_topology_data|torchrec.distribute.planner.types.CustomTopologyData|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|weighted_feature_bwd_compute_multiplier|float|可选| 当使用NPU设备时仅支持默认值为1，不支持用户自定义。                                |
|uneven_sharding_perf_multiplier|float|可选| 当使用NPU设备时仅支持默认值为1，不支持用户自定义。                                |


**使用示例<a name="section193151694205"></a>**

```cpp
from torchrec.distributed.planner import Topology,
topo = Topology(world_size=world_size, compute_device="npu")
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## DynamicEmbeddingCollectionSharder <a name="ZH-CN_TOPIC_0000002461958569"></a>

**功能描述<a name="section634582619155"></a>**

稀疏表分表器，继承自TorchRec的EmbeddingCollectionSharder，其使用方法与原生类完全相同。此API主要通过重写索引去重流程使其与动态嵌入表相兼容。

**函数原型<a name="section1483104721911"></a>**

```cpp
class DynamicEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(**kwargs):
```


**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|fused_params|Optional[Dict[str, Any]]|可选|用于配置嵌入操作中融合的参数，如优化器状态、学习率等。默认为None。|
|qcomm_codecs_registry|Optional[Dict[str, QuantizedCommCodecs]]|可选|用于注册量化通信编码器，量化通信可以减少通信量。默认为None。|
|use_index_dedup|bool|可选|是否使用local unique。默认为False。|

**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder
eb_sharder = DynamicEmbeddingCollectionSharder(
    fused_params=fused_params,
    use_index_dedup=False,
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。



## DynamicEmbParameterConstraints <a name="ZH-CN_TOPIC_0000002336148869"></a>

**功能描述<a name="section634582619155"></a>**

动态嵌入表的约束。继承自TorchRec基础的ParameterConstraints。

**函数原型<a name="section1483104721911"></a>**

```cpp
class DynamicEmbParameterConstraints(ParameterConstraints):
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                                                                                                                                                                 |
|--|--|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|use_dynamicemb|bool|可选|是否启用动态稀疏表。当使用NPU设备时仅支持默认值为True，不支持用户自定义。|
|dynamicemb_options|DynamicEmbTableOptions|可选|配置动态稀疏表的具体选项。参考DynmaicEmbTableOptions的取值范围|
|sharding_type|List[str]| 可选    | 分表的类型。当使用NPU设备时为必选参数，取值范围：<li>"row_wise"：按照行号进行分表。</li>说明</span><div class="notebody">不支持混合使用不同的分表类型。</div></div> |
|compute_kernels|List[str]| 可选    | 计算的kernel类型。当使用NPU设备时为必选参数，取值范围：<li>"fused"：采用合表的方式查询。该方式仅在sharding_type为"row_wise"时使用。</li>                                    |
|min_partition|List[int]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|pooling_factors|List[float]| 可选    | 当使用NPU设备时仅支持默认值为POOLING_FACTOR，不支持用户自定义。                                                                                                                                                                    |
|num_poolings|List[float]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|batch_sizes|List[int]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|is_weighted|bool| 可选    | 当使用NPU设备时仅支持默认值为False，不支持用户自定义。                                                                                                                                                                             |
|cache_params|torchrec.distributed.types.CacheParams| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|enforce_hbm|bool| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|stochastic_rounding|bool| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|bounds_check_mode|enum.IntEnum| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|feature_names|List[str]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|output_dtype|Enum| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|device_group|str| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|key_value_params|torchrec.distributed.types.KeyValueParams| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |


**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
constraints = {
    "table0": DynamicEmbParameterConstraints(
        sharding_types=[ShardingType.ROW_WISE.value],
        compute_kernels=["fused"],
        dynamicemb_options=DynamicEmbTableOptions(),
    ),
}
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbTableOptions <a name="ZH-CN_TOPIC_0000002338277269"></a>

**功能描述<a name="section634582619155"></a>**

动态嵌入表参数类，用于配置各个动态嵌入表的参数，这些参数将作为DynamicEmbParameterConstraints的输入。

**函数原型<a name="section1483104721911"></a>**

```cpp
@dataclass
class DynamicEmbTableOptions(_ContextOptions):
```

**参数说明<a name="section182631461211"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|training|bool|可选|指示动态嵌入表是否处于训练模式或评估模式的标志。默认为True。如果处于训练模式，dynamicemb会将嵌入和优化器状态一起存储在底层的键值表中。|
|initializer_args|DynamicEmbInitializerArgs|可选|训练模式下，用于初始化动态嵌入向量的参数。默认为DynamicEmbInitializerArgs实例，参考DynamicEmbInitializerArgs的取值范围。|
|eval_initializer_args|DynamicEmbInitializerArgs|可选|评估模式下，用于初始化动态嵌入向量的参数。默认为DynamicEmbInitializerArgs实例，仅支持DynamicEmbInitializerMode.CONSTANT。|
|caching|bool|可选|是否启用缓存模式。仅支持默认值为False，不支持用户自定义。|
|init_capacity|Optional[int]|可选|单个NPU上表的初始容量，如果未设置，默认为分片后的max_capacity；如果设置将向上取值到2的幂。取值范围：[0,MAX_INT32)，请用户自行保证内存使用情况。|
|max_load_factor|float|可选|触发rehash的最大负载因子。默认为0.5，取值范围(0.0,1.0)。|
|score_strategy|DynamicEmbScoreStrategy|可选|为每一个键分配一个评分，用于淘汰策略。默认为DynamicEmbScoreStrategy.TIMESTAMP。目前暂不支持DynamicEmbScoreStrategy.CUSTOMIZED。|
|bucket_capacity|int|可选|HKV中每个桶的容量，默认为128。如果设置，它将向上取整到2的幂。当桶已满时，桶中分数最小的键将被淘汰，其槽位将用于存放新键；桶容量越大，基于分数的淘汰就越准确，但也会导致性能损失。取值范围[16,1024]，用户自行保证内存使用情况。|
|safe_check_mode|DynamicEmbCheckMode|可选|是否启用插入安全检查。默认为DynamicEmbCheckMode.IGNORE。|
|global_hbm_for_values|int|可选|用于存储嵌入+优化器状态的NPU内存总量（单位：字节），默认为0的情况下在planner中会被设置为“值的类型的字节数 * 分表的行数（对齐到2的幂次）* 分表的列数”。|
|external_storage|Storage|可选|外部存储/参数服务器，用于替代默认的KeyValueTable。仅支持默认为None，不支持用户自定义。|
|index_type|torch.dtype|可选|稀疏特征的索引类型。仅支持默认值为torch.int64，不支持用户自定义。|


**使用示例<a name="section106984023511"></a>**

```cpp
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
table_options = [
    DynamicEmbTableOptions(
        index_type=torch.int64,
        optimizer_type=EmbOptimType.ADAM,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.NORMAL,
        ),
    )
]
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbInitializerMode <a name="ZH-CN_TOPIC_0000002304198202"></a>

**功能描述<a name="section634582619155"></a>**

枚举类。动态嵌入表中各嵌入向量的初始化方法，支持均匀/正态/常量初始化，默认采用均匀分布。

**函数原型<a name="section1483104721911"></a>**

```cpp
class DynamicEmbInitializerMode(enum.Enum):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                    |
|--|--|-------|-------------------------------------------------------|
|NORMAL|str|-| 使用正态分布（Normal Distribution）初始化嵌入向量|
|UNIFORM|str|-|使用均匀分布（Uniform Distribution）初始化嵌入向量|
|CONSTANT|str|-|所有嵌入向量的值都初始化为一个给定常量|

**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
initializer_args=DynamicEmbInitializerArgs(
    mode=DynamicEmbInitializerMode.NORMAL,
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbInitializerArgs <a name="ZH-CN_TOPIC_0000002508694909"></a>

**功能描述<a name="section634582619155"></a>**

数据类，DynamicEmbInitializerMode中每个随机初始化的参数。

**函数原型<a name="section1483104721911"></a>**

```cpp
@dataclass
class DynamicEmbInitializerArgs:
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                  |
|--|--|-------|-----------------------------------------------------|
|mode|DynamicEmbInitializerMode|可选|初始化模式。可选NORMAL、UNIFORM、CONSTANT，默认为UNIFORM。|
|mean|float|可选|正态分布的均值，默认为 0.0。|
|std_dev|float|可选|正态分布的标准差，默认为 1.0。|
|lower|float|可选|均匀分布的下界，默认为 None。|
|upper|float|可选|均匀分布的上界，默认为 None。|
|value|float|可选|常量初始化时使用的固定值，默认为 0.0。|


**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
initializer_args=DynamicEmbInitializerArgs(
    mode=DynamicEmbInitializerMode.CONSTANT,
    value=0.1,
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbScoreStrategy<a name="ZH-CN_TOPIC_0000002338384297"></a>

**功能描述<a name="section634582619155"></a>**

枚举类，HKV评分机制，用于稀疏特征的定制化淘汰。

**函数原型<a name="section1483104721911"></a>**

```cpp
class DynamicEmbScoreStrategy(enum.IntEnum):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                             |
|--|--|-------|----------------------------------------------------------------|
|TIMESTAMP|int|-|使用设备的纳秒级全局时间戳作为评分|
|STEP|int|-|使用嵌入表内部的步数计数器（step）作为评分|
|CUSTOMIZED|int|-|用户完全自定义评分|
|LFU|int|-|根据嵌入项被访问的频率自动计算评分|


**使用示例<a name="section193151694205"></a>**

```cpp
table_options = DynamicEmbTableOptions(
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。



## EmbOptimType <a name="ZH-CN_TOPIC_0000002476574952"></a>

**功能描述<a name="section634582619155"></a>**

枚举类，动态嵌入表优化器类型。

**函数原型<a name="section1483104721911"></a>**

```cpp
@enum.unique
class EmbOptimType(enum.Enum):
```

**参数说明<a name="section1367815197580"></a>**

|参数名|类型| 可选/必选 | 说明                                                             |
|--|--|-------|----------------------------------------------------------------|
|ADAM|str|-|嵌入表Adam优化器|
|ADAMW|str|-|嵌入表AdamW优化器|


**使用示例<a name="section1045492782314"></a>**

```cpp
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import 
table_options = [
    DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        optimizer_type=EmbOptimType.ADAM,
    )
]
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbeddingEnumerator<a name="ZH-CN_TOPIC_0000002396403112"></a>

**功能描述<a name="section634582619155"></a>**

继承自TorchRec的EmbeddingEnumerator，使用方法与其完全相同。该类别在为分片计划执行枚举时会区分常规/动态嵌入表。  

**函数原型<a name="section1483104721911"></a>**

```cpp
class DynamicEmbeddingEnumerator(EmbeddingEnumerator):
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|topology|Topology|必选|描述 GPU 与主机内存的拓扑结构，用于优化分布式或跨设备内存访问。参考Topology的取值范围。|
|batch_size|Optional[int]|可选|训练时的批处理大小，与 TorchRec 中的同名参数行为一致。默认为512，取值范围：[1,1000000]。|
|constraints|Optional[Dict[str, DynamicEmbParameterConstraints]]|可选|动态嵌入表参数的约束配置字典，键为参数名，值为对应的约束对象。用于定义初始化方式、缓存策略、容量等，默认为None。参考DynamicEmbParameterConstraints的取值范围。|
|estimator|Optional[Union[ShardEstimator, List[ShardEstimator]]]|可选|用于估算分片大小的评估器，支持单个或多个评估器，行为与TorchRec中一致。仅支持默认值为None，不支持用户自定义。|
|use_exact_enumerate_order|Optional[bool]|可选|是否严格按照模型参数名的字典序（named_children）来枚举可分片参数。若为 True，则保证参数遍历顺序与模型结构一致。仅支持默认值为False，不支持用户自定义。|

**使用示例<a name="section1045492782314"></a>**

```cpp
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
enumerator=DynamicEmbeddingEnumerator(
    topology=topology,
    constraints=constraints,
)

```


**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbeddingShardingPlanner<a name="ZH-CN_TOPIC_0000002396403120"></a>

**功能描述<a name="section634582619155"></a>**

对TorchRec的EmbeddingShardingPlanner进行封装，与EmbeddingShardingPlanner不同的是该封装类需要接受额外eb_configs参数来规划动态嵌入表的容量配置。

**函数原型<a name="section1483104721911"></a>**

```cpp
class DynamicEmbeddingShardingPlanner:
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|eb_configs|List[BaseEmbeddingConfig]|必选|TorchRec模型中所有嵌入表配置的列表，用于动态嵌入规划。当前仅支持EmbeddingConfig，参考EmbeddingConfig的取值范围。|
|constraints|Dict[str, DynamicEmbParameterConstraints]|必选|每个嵌入表的约束配置字典，键为表名，值为约束对象。参考DynamicEmbParameterConstraints的取值范围。|
|topology|Optional[Topology]|可选|GPU和主机内存的拓扑结构，若为None，将使用默认拓扑，默认为None。参考Topology的取值范围。|
|batch_size|Optional[int]|可选|训练时的批处理大小。若为None，默认设置为 512。取值范围：[1,1000000]。|
|enumerator|Optional[Enumerator]|可选|用于分片的枚举器。若为None，将使用默认枚举器。默认为None。|
|storage_reservation|Optional[StorageReservation]|可选|存储预留信息，用于规划时预留部分内存。仅支持默认值为None，不支持用户自定义。|
|proposer|Optional[Union[Proposer, List[Proposer]]]|可选|分片方案提议器。仅支持默认值为None，不支持用户自定义。|
|partitioner|Optional[Partitioner]|可选|用于对嵌入表进行分片的分区器。仅支持默认值为None，不支持用户自定义。|
|performance_model|Optional[PerfModel]|可选|性能模型，用于评估不同分片方案的效率。仅支持默认值为None，不支持用户自定义。|
|stats|Optional[Union[Stats, List[Stats]]]|可选|统计信息收集器。仅支持默认值为None，不支持用户自定义。|
|debug|bool|可选|是否启用调试模式，启用后将输出更多日志信息。仅支持默认值为True，不支持用户自定义。|

**使用示例<a name="section193151694205"></a>**

```cpp
eb_planner = DynamicEmbeddingShardingPlanner(
    eb_configs=eb_configs,
    topology=topology,
    constraints=constraints,
    batch_size=batch_size,
    enumerator=DynamicEmbeddingEnumerator(
        topology=self.topology,
        constraints=constraints,
    ),
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## DynamicEmbCheckMode <a name="ZH-CN_TOPIC_0000002396563024"></a>

**功能描述<a name="section634582619155"></a>**

枚举类,嵌入表索引插入安全检查模式。当动态嵌入表容量较小时，单个包含大量索引的特征可能导致哈希表无法插入索引的问题。启用安全检查后，可以观察到此类情况（包括索引插入失败的次数及每次失败的索引数量），从而判断动态嵌入表容量是否设置过低。

**函数原型<a name="section1483104721911"></a>**

```cpp
@enum.unique
class DynamicEmbCheckMode(enum.IntEnum):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|ERROR|int|-|若有索引无法成功插入，则抛出运行时错误，提示失败数量。|
|WARNING|int|-|若有索引无法成功插入，则输出警告信息，显示失败数量。|
|IGNORE|int|-|不进行插入是否成功的检查。|

**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbTableOptions,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
table_options = [
    DynamicEmbTableOptions(
        safe_check_mode = DynamicEmbCheckMode.IGNORE
    ),
]
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。