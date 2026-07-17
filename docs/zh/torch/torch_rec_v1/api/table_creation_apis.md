# 创表接口<a name="ZH-CN_TOPIC_0000002336148893"></a>

Rec SDK Torch提供两类稀疏表配置与创建接口，分别对应纯显存模式和多级缓存模式，并按是否带Pooling划分为EBC与EC两种查表模式。各Config与Collection的对应关系如下表所示。

**表 1**  Config与Collection对应关系

| 模式 | 是否带Pooling | Config配置类 | Collection创建类 |
|------|------------|-------------|------------------|
| 纯显存模式 | 是（EBC） | HashEmbeddingBagConfig | HashEmbeddingBagCollection |
| 多级缓存模式 | 是（EBC） | EmbCacheEmbeddingBagConfig | EmbCacheEmbeddingBagCollection |
| 多级缓存模式 | 否（EC） | EmbCacheEmbeddingConfig | EmbCacheEmbeddingCollection |

## HashEmbeddingBagConfig<a name="ZH-CN_TOPIC_0000002336148933"></a>

**功能描述<a name="section634582619155"></a>**

HashEmbeddingBagCollection的入参，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```python
@dataclass
class HashEmbeddingBagConfig(EmbeddingBagConfig):
    num_embeddings: int
    embedding_dim: int
    name: str
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    num_embeddings_post_pruning: Optional[int] = None
    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    need_pos: bool = False
    pooling: PoolingType = PoolingType.SUM
```

**参数说明<a name="section1643017411155"></a>**

| 参数名                         | 类型                                          | 可选/必选 | 说明                                                                                        |
|-----------------------------|---------------------------------------------|-------|-------------------------------------------------------------------------------------------|
| num_embeddings              | int                                         | 必选    | 稀疏表的行数。取值范围：[1, 10亿]。                                                                     |
| embedding_dim               | int                                         | 必选    | 稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。                                                          |
| name                        | str                                         | 必选    | 稀疏表的名称。只能包含数字、字母和下划线。                                                                     |
| data_type                   | torchrec.types.DataType                     | 可选    | 稀疏表的数据类型。仅支持默认值为DataType.FP32。                                                            |
| feature_names               | List[str]                                   | 必选    | 稀疏表查询的特征名称列表。特征名称字符串中只能包含数字、字母和下划线。                                                                 |
| weight_init_max             | float                                       | 可选    | 仅支持默认值为None或1.0，不支持用户自定义。                                                                 |
| weight_init_min             | float                                       | 可选    | 仅支持默认值为None或0.0，不支持用户自定义。                                                                 |
| num_embeddings_post_pruning | int                                         | 可选    | 仅支持默认值为None，不支持用户自定义。                                                                     |
| init_fn                     | Callable                                    | 可选    | 支持传入torch.nn.Parameter类型参数的函数。用户需自行保证该函数的正确性。默认值为None。                                            |
| need_pos                    | bool                                        | 可选    | 仅支持默认值为False，不支持用户自定义。                                                                    |
| pooling                     | torchrec.PoolingType | 可选    | pooling操作的类型。取值范围：<ul><li>SUM：求和。</li><li>MEAN：取平均。</li><li>NONE：不做pooling操作。</li></ul> 默认为SUM。 |

**返回值说明**

- 成功：返回HashEmbeddingBagConfig对象。
- 失败：抛出异常。

**使用示例**

```python
import torch
import torchrec
from hybrid_torchrec import HashEmbeddingBagConfig


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)

emb_config = HashEmbeddingBagConfig(
    name="table0",
    embedding_dim=128,
    num_embeddings=1000,
    feature_names=["feat0"],
    pooling=torchrec.PoolingType.MEAN,
    init_fn=weight_init,  # type: ignore
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。

## HashEmbeddingBagCollection<a name="ZH-CN_TOPIC_0000002302389408"></a>

**功能描述<a name="section634582619155"></a>**

创建带Pooling和哈希映射的稀疏表对象。

**函数原型<a name="section1483104721911"></a>**

```python
class HashEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    def __init__(
        self,
        tables: Union[List[HashEmbeddingBagConfig], List[EmbeddingBagConfig]],
        is_weighted: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
```

**参数说明<a name="section1643017411155"></a>**

| 参数名         | 类型                                                 | 可选/必选 | 说明                                                                                                                                                                                                                                                                                                      |
|-------------|----------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables      | List[HashEmbeddingBagConfig \| EmbeddingBagConfig] | 必选    | 稀疏表配置列表。列表长度的取值范围：[1, 10000]。<p>参数范围参考HashEmbeddingBagConfig。</p>                                                                                                                                                                                                                                     |
| is_weighted | bool                                               | 可选    | 仅支持默认值为False。                                                                                                                                                                                                                                                                                           |
| device      | Union[str, torch.device]                                  | 可选    | 稀疏表的设备。默认为torch.device("cpu")。<br>如果为str取值范围：<ul><li>"npu"：npu设备。</li><li>"meta"：meta设备。</li><li>"cpu"：cpu设备。cpu设备不支持分布式表，只支持单机表。</li></ul><br>如果为torch.device取值范围：<ul><li>torch.device("npu")：npu设备。</li><li>torch.device("meta")：meta设备。</li><li>torch.device("cpu")：cpu设备。cpu设备不支持分布式表，只支持单机表。</li></ul> |

**返回值说明**

- 成功：返回HashEmbeddingBagCollection对象。
- 失败：抛出异常。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```python
import torch
import torchrec
from hybrid_torchrec.modules.hash_embeddingbag import HashEmbeddingBagCollection, HashEmbeddingBagConfig


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


embedding_dims: list[int] = [64, 64, 64]
num_embeddings: list[int] = [400, 4000, 400]
table_num: int = len(num_embeddings)
embedding_configs: list[HashEmbeddingBagConfig] = []
table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
feat_names: list[str] = [f"feat{i}" for i in range(table_num)]
for i in range(table_num):
    emb_config = HashEmbeddingBagConfig(
        name=table_names[i],
        embedding_dim=embedding_dims[i],
        num_embeddings=num_embeddings[i],
        feature_names=[feat_names[i]],
        pooling=torchrec.PoolingType.MEAN,
        init_fn=weight_init,  # type: ignore
    )
    embedding_configs.append(emb_config)
sparse_ebc: HashEmbeddingBagCollection = HashEmbeddingBagCollection(device="meta", tables=embedding_configs)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。

## EmbCacheEmbeddingBagConfig<a name="ZH-CN_TOPIC_0000002396403104"></a>

**功能描述<a name="section634582619155"></a>**

EmbCacheEmbeddingBagConfig是EmbCacheEmbeddingBagCollection的配置类接口，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```python
@dataclass
class EmbCacheEmbeddingBagConfig(EmbeddingBagConfig):
    num_embeddings: int
    embedding_dim: int
    name: str
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    num_embeddings_post_pruning: Optional[int] = None
    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    need_pos: bool = False
    pooling: PoolingType = PoolingType.SUM
    weight_init_mean: Optional[float] = 0.0
    weight_init_stddev: Optional[float] = 0.05
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )
    is_incremental: bool = False
```

**参数说明**

| 参数名                         | 类型                                          | 可选/必选 | 说明                                                                |
|-----------------------------|---------------------------------------------|-------|-------------------------------------------------------------------|
| num_embeddings              | int                                         | 必选    | 稀疏表的行数。取值范围：[1, 10亿]。使用row_wise时至少一张稀疏表的行数≥使用的卡数。                 |
| embedding_dim               | int                                         | 必选    | 稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。多个稀疏表的embedding_dim需保持一致。                                  |
| name                        | str                                         | 必选    | 稀疏表的名称。只能包含数字、字母和下划线。                                             |
| data_type                   | torchrec.types.DataType                     | 可选    | 稀疏表的数据类型。仅支持默认值为DataType.FP32。                                    |
| feature_names               | List[str]                                   | 必选    | 稀疏表查询的特征名称列表。特征名称字符串中只能包含数字、字母和下划线。                                         |
| weight_init_max             | float                                       | 可选    | 仅支持默认值为None或1.0，不支持用户自定义。                                         |
| weight_init_min             | float                                       | 可选    | 仅支持默认值为None或0.0，不支持用户自定义。                                         |
| num_embeddings_post_pruning | int                                         | 可选    | 仅支持默认值为None，不支持用户自定义。                                             |
| init_fn                     | Callable                                    | 可选    | 支持传入torch.nn.Parameter类型参数的函数。用户需自行保证该函数的正确性。默认值为None。**当前配置类用于多级缓存EBC模式，Embedding初始化需通过`initializer_type`参数设置，`init_fn`参数不生效**。                    |
| need_pos                    | bool                                        | 可选    | 仅支持默认值为False，不支持用户自定义。                                            |
| pooling                     | torchrec.PoolingType | 可选    | pooling操作的类型。取值范围：<ul><li>SUM：求和。</li><li>MEAN：取平均。</li></ul>默认为SUM。 |
| weight_init_mean            | float                                       | 可选    | 权重初始化均值，用于InitializerType.TRUNCATED_NORMAL初始化类型，默认值0.0。                                    |
| weight_init_stddev          | float                                       | 可选    | 权重初始化标准差，用于InitializerType.TRUNCATED_NORMAL初始化类型，默认值0.05。                                  |
| initializer_type            | InitializerType                             | 可选    | 权重初始化类型，支持LINEAR、TRUNCATED_NORMAL、UNIFORM，默认值LINEAR。详细说明请参见[InitializerType](multilevel_cache_management_apis.md#initializertype)。<br>各取值类型说明：<br>InitializerType.LINEAR：配合weight_init_min、weight_init_max参数，进行Embedding的线性初始化。<br>InitializerType.TRUNCATED_NORMAL：配合weight_init_min、weight_init_max、weight_init_mean、weight_init_stddev参数，进行Embedding的截断正态分布初始化。<br>InitializerType.UNIFORM：配合weight_init_min、weight_init_max参数，进行Embedding的均匀分布初始化。              |
| admit_and_evict_config      | AdmitAndEvictConfig                         | 可选    | 特征准入和淘汰配置，**预留参数，当前配置类暂不支持该参数**。                               |
| is_incremental              | bool                                        | 可选    | 是否开启训练过程中增量数据处理功能，需结合增量保存/加载功能使用。默认值为False，表示不开启。增量保存/加载功能请参见[Saver](multilevel_cache_management_apis.md#saver)。                                                       |

**返回值说明**

- 成功：返回EmbCacheEmbeddingBagConfig对象。
- 失败：抛出异常。

**使用示例**

```python
from torchrec_embcache.distributed import EmbCacheEmbeddingBagConfig, InitializerType

emb_config = EmbCacheEmbeddingBagConfig(
    name="table0",
    embedding_dim=128,
    num_embeddings=1000,
    feature_names=["feat0"],
    # init_fn=weight_init,  # 注意：多级缓存模式，不支持自定义初始化函数
    initializer_type=InitializerType.TRUNCATED_NORMAL,  # Embedding初始化方式通过initializer_type参数设置
    weight_init_mean=0.0,
    weight_init_stddev=0.05,
)
```

## EmbCacheEmbeddingBagCollection<a name="ZH-CN_TOPIC_0000002396562992"></a>

**功能描述<a name="section634582619155"></a>**

创建带Pooling、哈希映射的多级缓存稀疏表对象。

**函数原型<a name="section1483104721911"></a>**

```python
class EmbCacheEmbeddingBagCollection(EmbeddingBagCollection):
    def __init__(
        self,
        tables: List[EmbCacheEmbeddingBagConfig | EmbeddingBagConfig],
        world_size: int,
        batch_size: int,
        multi_hot_sizes: List[int],
        is_weighted: bool = False,
        need_accumulate_offset: bool = True,
        device: Optional[torch.device] = None,
        embedding_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
    ) -> None:
```

**参数说明**

| 参数名                     | 类型                                                | 可选/必选 | 说明                                                                                                                                                             |
|-------------------------|---------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables                  | List[EmbCacheEmbeddingBagConfig\|EmbeddingBagConfig] | 必选    | 稀疏表配置列表。列表长度的取值范围为[1，10000]。                                                                                                                                 |
| world_size              | int                                               | 必选    | 分布式训练world_size大小，取值范围为[1，10000]。                                                                                                                              |
| batch_size              | int                                               | 必选    | 批次大小，取值范围为[1，102400]。                                                                                                                                          |
| multi_hot_sizes         | List[int]                                         | 必选    | 每个特征的多热编码大小列表，用于计算训练中所需的最小device内存大小。该参数列表的长度必须与tables的列表长度相同，取值范围为[1，10000]；列表中多热编码大小的取值范围为[1，102400]。                                                                               |
| is_weighted             | bool                                              | 可选    | 仅支持默认值False。                                                                                                                                                   |
| need_accumulate_offset  | bool                                              | 可选    | 是否需要累积偏移量，默认值True。                                                                                                                                             |
| device                  | torch.device                                      | 可选    | 计算设备，默认值torch.device("cpu")。                                                                                                                                   |
| embedding_optimizer_cls | Type[torch.optim.Optimizer]                       | 可选    | 嵌入优化器类型，用于计算训练中所需的最小device内存大小，默认值torch.optim.Adagrad。取值范围：<ul><li>torch.optim.Adagrad：表示Adagrad优化器。</li><li>torch.optim.Adam：表示Adam优化器。</li><li>torch.optim.SGD：表示SGD优化器。</li><li>torchrec.optim.Adagrad：Adagrad优化器。</li><li>torchrec.optim.Adam：Adam优化器。</li><li>torchrec.optim.SGD：SGD优化器。</li></ul> <br>优化器类型参数需和调用[apply_optimizer_in_backward](optimizers_apis.md#TOPIC_0000002302229708)时传入的优化器类型一致。 |

**返回值说明**

- 成功：返回EmbCacheEmbeddingBagCollection对象。
- 失败：抛出异常。

**使用示例**

```python
import torch
from typing import List
from torchrec_embcache.distributed import EmbCacheEmbeddingBagCollection, EmbCacheEmbeddingBagConfig
from torchrec_embcache.distributed import EmbCacheEmbeddingBagConfig, InitializerType

emb_config = EmbCacheEmbeddingBagConfig(
    name="table0",
    embedding_dim=128,
    num_embeddings=1000,
    feature_names=["feat0"],
    initializer_type=InitializerType.TRUNCATED_NORMAL,  # Embedding初始化方式通过initializer_type参数设置
    weight_init_mean=0.0,
    weight_init_stddev=0.05,
)
embedding_configs: List[EmbCacheEmbeddingBagConfig] = [emb_config]
world_size: int = 1
batch_size: int = 128
table_num = len(embedding_configs)
embedding_optimizer_cls = torch.optim.Adagrad
sparse_ebc: torch.nn.Module = EmbCacheEmbeddingBagCollection(
    embedding_configs, # type: ignore
    world_size,
    batch_size,
    multi_hot_sizes=[1] * table_num,
    device=torch.device("meta"),
    embedding_optimizer_cls=embedding_optimizer_cls,
)
```

## EmbCacheEmbeddingConfig<a name="ZH-CN_TOPIC_0000002430082769"></a>

**功能描述<a name="section634582619155"></a>**

EmbCacheEmbeddingConfig是EmbCacheEmbeddingCollection的配置类接口，用于配置表的大小、dim、数据类型等。

**函数原型<a name="section1483104721911"></a>**

```python
@dataclass
class EmbCacheEmbeddingConfig(EmbeddingConfig):
    num_embeddings: int
    embedding_dim: int
    name: str
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    num_embeddings_post_pruning: Optional[int] = None
    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    need_pos: bool = False
    weight_init_mean: Optional[float] = 0.0
    weight_init_stddev: Optional[float] = 0.05
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )
    is_incremental: bool = False
```

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|num_embeddings|int|必选|稀疏表的行数。取值范围：[1, 10亿]。使用row_wise时至少一张稀疏表的行数≥使用的卡数。|
|embedding_dim|int|必选|稀疏表的列数。取值范围：[8, 4096]。取值需要为8的倍数。多个稀疏表的embedding_dim需保持一致。|
|name|str|必选|稀疏表的名称。只能包含数字、字母和下划线。|
|data_type|torchrec.types.DataType|可选|稀疏表的数据类型。仅支持默认值为DataType.FP32。|
|feature_names|List[str]|必选|稀疏表查询的特征名称列表。特征名称字符串中只能包含数字、字母和下划线。|
|weight_init_max|float|可选|仅支持默认值为None或1.0，不支持用户自定义。|
|weight_init_min|float|可选|仅支持默认值为None或0.0，不支持用户自定义。|
|num_embeddings_post_pruning|int|可选|仅支持默认值为None，不支持用户自定义。|
|init_fn|Callable|可选|支持传入torch.nn.Parameter类型参数的函数。默认值为None。**当前配置类用于多级缓存EC模式，Embedding初始化需通过`initializer_type`参数设置，`init_fn`参数不生效。**|
|need_pos|bool|可选|仅支持默认值为False，不支持用户自定义。|
|weight_init_mean|float|可选|权重初始化均值，用于InitializerType.TRUNCATED_NORMAL初始化类型，默认值0.0。|
|weight_init_stddev|float|可选|权重初始化标准差，用于InitializerType.TRUNCATED_NORMAL初始化类型，默认值0.05。|
|initializer_type|InitializerType|可选|权重初始化类型，支持LINEAR、TRUNCATED_NORMAL、UNIFORM，默认值LINEAR。<br>各取值类型说明：<br>InitializerType.LINEAR：配合weight_init_min、weight_init_max参数，进行Embedding的线性初始化。<br>InitializerType.TRUNCATED_NORMAL：配合weight_init_min、weight_init_max、weight_init_mean、weight_init_stddev参数，进行Embedding的截断正态分布初始化。<br>InitializerType.UNIFORM：配合weight_init_min、weight_init_max参数，进行Embedding的均匀分布初始化。|
|admit_and_evict_config|AdmitAndEvictConfig|可选|特征准入和淘汰配置，默认不启用准入和淘汰功能。|
|is_incremental|bool|可选| 是否开启训练过程中增量数据处理功能，需结合增量保存/加载功能使用。默认值为False，表示不开启。增量保存/加载功能请参见[Saver](multilevel_cache_management_apis.md#saver)。|

**返回值说明**

- 成功：返回EmbCacheEmbeddingConfig对象。
- 失败：抛出异常。

**使用示例**

```python
from typing import List

import torch
from torchrec_embcache.distributed import EmbCacheEmbeddingCollection, EmbCacheEmbeddingConfig, InitializerType

emb_config = EmbCacheEmbeddingConfig(
    name="table0",
    embedding_dim=128,
    num_embeddings=1000,
    feature_names=["feat0"],
    initializer_type=InitializerType.TRUNCATED_NORMAL,
    weight_init_mean=0.0,
    weight_init_stddev=0.05,
)
embedding_configs: List[EmbCacheEmbeddingConfig] = [emb_config]
embedding_optimizer_cls = torch.optim.Adagrad
table_num = len(embedding_configs)
world_size = 1
batch_size = 128
sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
    embedding_configs,  # type: ignore
    world_size,
    batch_size,
    multi_hot_sizes=[1] * table_num,
    device=torch.device("meta"),
    embedding_optimizer_cls=embedding_optimizer_cls,
)
```

## EmbCacheEmbeddingCollection<a name="ZH-CN_TOPIC_0000002430202745"></a>

**功能描述<a name="section634582619155"></a>**

创建带哈希映射的多级缓存模式稀疏表对象。

**函数原型<a name="section1483104721911"></a>**

```python
class EmbCacheEmbeddingCollection(EmbeddingCollection):
    def __init__(
        self,
        tables: List[EmbCacheEmbeddingConfig | EmbeddingConfig],
        world_size: int,
        batch_size: int,
        multi_hot_sizes: List[int],
        need_indices: bool = False,
        need_accumulate_offset: bool = True,
        device: Optional[torch.device] = None,
        embedding_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adagrad,
    ) -> None:
```

**参数说明**

| 参数名                     | 类型                                             | 可选/必选 | 说明                                                                                                                                                            |
|-------------------------|------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables                  | List[EmbCacheEmbeddingConfig\|EmbeddingConfig] | 必选    | 稀疏表配置列表。列表长度的取值范围为[1，10000]。                                                                                                                                  |
| world_size              | int                                            | 必选    | 分布式训练world_size大小，取值范围为[1，10000]。                                                                                                                             |
| batch_size              | int                                            | 必选    | 批次大小，取值范围为[1，102400]。                                                                                                                                         |
| multi_hot_sizes         | List[int]                                      | 必选    | 每个特征的多热编码大小列表，用于计算训练中所需的最小device内存大小。该参数列表的长度必须与tables的列表长度相同，取值范围为[1，10000]；列表中多热编码大小的取值范围为[1，102400]。                                                                              |
| need_indices            | bool                                           | 可选    | 是否需要索引，默认值False。                                                                                                                                              |
| need_accumulate_offset  | bool                                           | 可选    | 是否需要累积偏移量，默认值True。                                                                                                                                            |
| device                  | torch.device                                   | 可选    | 计算设备，默认值torch.device("cpu")。                                                                                                                                  |
| embedding_optimizer_cls | Type[torch.optim.Optimizer]                    | 可选    | 嵌入优化器类型，用于计算训练中所需的最小device内存大小，默认值torch.optim.Adagrad。取值范围：<ul><li>torch.optim.Adagrad：表示Adagrad优化器。</li><li>torch.optim.Adam：表示Adam优化器。</li><li>torch.optim.SGD：表示SGD优化器。</li><li>torchrec.optim.Adagrad：Adagrad优化器。</li><li>torchrec.optim.Adam：Adam优化器。</li><li>torchrec.optim.SGD：SGD优化器。</li><li>torchrec.optim.AccumulateAdagrad：带梯度累积功能的Adagrad优化器。</li><li>torchrec.optim.AccumulateAdam：带梯度累积功能的Adam优化器。</li><li>torchrec.optim.AccumulateSGD：带梯度累积功能的SGD优化器。</li></ul> <br>优化器类型参数需和调用[apply_optimizer_in_backward](optimizers_apis.md#TOPIC_0000002302229708)时传入的优化器类型一致。 |

**返回值说明**

- 成功：返回EmbCacheEmbeddingCollection对象。
- 失败：抛出异常。

**使用示例**

```python
from typing import List

import torch
from torchrec_embcache.distributed import EmbCacheEmbeddingCollection, EmbCacheEmbeddingConfig

# embedding_configs为EmbCacheEmbeddingConfig对象列表，此处省略详细定义。
embedding_configs: List[EmbCacheEmbeddingConfig] = ......
# 此处的优化器类型需和torchrec.optim.apply_optimizer_in_backward时传入的优化器类型一致
embedding_optimizer_cls = torch.optim.Adagrad
table_num = len(embedding_configs)
sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
    embedding_configs,  # type: ignore
    2,
    128,
    multi_hot_sizes=[1] * table_num,
    device=torch.device("meta"),
    embedding_optimizer_cls=embedding_optimizer_cls,
)
```
