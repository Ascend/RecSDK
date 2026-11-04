# Table Creation APIs

Rec SDK Torch provides two types of sparse table configuration and creation APIs, corresponding to full-device memory mode and multi-level cache mode. The APIs are divided into two table lookup modes, EBC and EC, based on whether pooling is used. The following table describes the mapping between the Config and Collection classes.

**Table 1** Mapping between Config and Collection

| Mode | Pooling | Config Class | Collection Class |
|------|---------|--------------|------------------|
| Full-device memory mode | Yes (EBC) | HashEmbeddingBagConfig | HashEmbeddingBagCollection |
| Multi-level cache mode | Yes (EBC) | EmbCacheEmbeddingBagConfig | EmbCacheEmbeddingBagCollection |
| Multi-level cache mode | No (EC) | EmbCacheEmbeddingConfig | EmbCacheEmbeddingCollection |

## `HashEmbeddingBagConfig`

**Description**

Input to `HashEmbeddingBagCollection`. Use it to configure table size, `dim`, data type, and other settings.

**Function Prototype**

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
    input_dim: Optional[int] = None
    total_num_buckets: Optional[int] = None
    use_virtual_table: bool = False
    virtual_table_eviction_policy: Optional[VirtualTableEvictionPolicy] = None
    enable_embedding_update: bool = False
```

**Parameters**

| Parameter                 | Type                           | Mandatory/Optional| Description                                                                                       |
|-----------------------------|---------------------------------------------|-------|-------------------------------------------------------------------------------------------|
| num_embeddings              | int                                         | Mandatory   | Number of rows in the sparse table. The value ranges from 1 to 1 billion.                                                                    |
| embedding_dim               | int                                         | Mandatory   | Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8.                                                         |
| name                        | str                                         | Mandatory   | Name of the sparse table. The value can contain only numbers, letters, and underscores (_).                                                                    |
| data_type                   | torchrec.types.DataType                     | Optional   | Data type of the sparse table. Only the default value `DataType.FP32` is supported.                                                           |
| feature_names               | List[str]                                   | Mandatory   | List of feature names queried by the sparse table. Each feature name can contain only numbers, letters, and underscores (_).                                                                |
| weight_init_max             | float                                       | Optional   | Only the default value `None` or 1.0 is supported.                                                                |
| weight_init_min             | float                                       | Optional   | Only the default value `None` or 0.0 is supported.                                                                |
| num_embeddings_post_pruning | int                                         | Optional   | Only the default value `None` is supported.                                                                    |
| init_fn                     | Callable                                    | Optional   | You can pass a function that takes a parameter of the `torch.nn.Parameter` type. You need to ensure that the function is correct. The default value is `None`.                                           |
| need_pos                    | bool                                        | Optional   | Only the default value `False` is supported.                                                                   |
| pooling                     | torchrec.PoolingType                       | Optional   | Type of the pooling operation. The value can be:<ul><li>`SUM`: Sum. </li><li>`MEAN`: Average. </li><li>`NONE`: Does not perform pooling. </li></ul> The default value is `SUM`.|
| input_dim                   | int                                         | Optional   | Input dimension. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.2.0 and v1.5.0.                                                  |
| total_num_buckets           | int                                         | Optional   | Total number of buckets, which remains unchanged during the model lifecycle. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| use_virtual_table           | bool                                        | Optional   | Indicates whether to use a virtual table. When an NPU device is used, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| virtual_table_eviction_policy | torchrec.modules.embedding_configs.VirtualTableEvictionPolicy | Optional   | Eviction policy of the virtual table. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| enable_embedding_update     | bool                                        | Optional   | Indicates whether to enable Embedding update. When an NPU device is used, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |

**Returns**

- Success: A `HashEmbeddingBagConfig` object is returned.
- Failure: An exception is thrown.

**Example**

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

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

## `HashEmbeddingBagCollection`

**Description**

Creates a sparse table object with pooling and hash mapping.

**Function Prototype**

```python
class HashEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    def __init__(
        self,
        tables: Union[List[HashEmbeddingBagConfig], List[EmbeddingBagConfig]],
        is_weighted: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
```

**Parameters**

| Parameter        | Type                                                | Mandatory/Optional| Description                                                                                                                                                                                                                                                                                                     |
|-------------|----------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables      | List[HashEmbeddingBagConfig \| EmbeddingBagConfig] | Mandatory   | List of sparse table configurations. The list length must be in the range [1, 10000]. <p>See `HashEmbeddingBagConfig` for parameter details.</p>                                                                                                                                                                                                                                     |
| is_weighted | bool                                               | Optional   | Only the default value `False` is supported.                                                                                                                                                                                                                                                                                          |
| device      | Union[str, torch.device]                           | Optional   | Device for the sparse table. The default value is `torch.device("cpu")`.<br>If the type is a string, supported values are:<ul><li>`"npu"`: NPU device. </li><li>`"meta"`: Meta device. </li><li>`"cpu"`: CPU device. The CPU device does not support distributed tables and supports only single-node tables.</li></ul><br>If the type is `torch.device`, supported values are:<ul><li>`torch.device("npu")`: NPU device. </li><li>`torch.device("meta")`: Meta device. </li><li>`torch.device("cpu")`: CPU device. The CPU device does not support distributed tables and supports only single-node tables.</li></ul>|

**Returns**

- Success: A `HashEmbeddingBagCollection` object is returned.
- Failure: An exception is thrown.

**Example**

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

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

## `EmbCacheEmbeddingBagConfig`

**Description**

Configuration class API for `EmbCacheEmbeddingBagCollection`. It configures table size, `dim`, data type, and other settings.

**Function Prototype**

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
    input_dim: Optional[int] = None
    total_num_buckets: Optional[int] = None
    use_virtual_table: bool = False
    virtual_table_eviction_policy: Optional[VirtualTableEvictionPolicy] = None
    enable_embedding_update: bool = False
    weight_init_mean: Optional[float] = 0.0
    weight_init_stddev: Optional[float] = 0.05
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )
    is_incremental: bool = False
```

**Parameters**

| Parameter                        | Type                                         | Mandatory/Optional| Description                                                               |
|-----------------------------|---------------------------------------------|-------|-------------------------------------------------------------------|
| num_embeddings              | int                                         | Mandatory   | Number of rows in the sparse table. The value ranges from 1 to 1 billion. When `row_wise` is used, at least one sparse table must have a row count greater than or equal to the number of used devices.                |
| embedding_dim               | int                                         | Mandatory   | Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8. The `embedding_dim` values of multiple sparse tables must be consistent.                                 |
| name                        | str                                         | Mandatory   | Name of the sparse table. The value can contain only numbers, letters, and underscores (_).                                            |
| data_type                   | torchrec.types.DataType                     | Optional   | Data type of the sparse table. Only the default value `DataType.FP32` is supported.                                   |
| feature_names               | List[str]                                   | Mandatory   | List of feature names queried by the sparse table. Each feature name can contain only numbers, letters, and underscores (_).                                        |
| weight_init_max             | float                                       | Optional   | Only the default value `None` or 1.0 is supported.                                        |
| weight_init_min             | float                                       | Optional   | Only the default value `None` or 0.0 is supported.                                        |
| num_embeddings_post_pruning | int                                         | Optional   | Only the default value `None` is supported.                                            |
| init_fn                     | Callable                                    | Optional   | You can pass a function that takes a parameter of the `torch.nn.Parameter` type. You need to ensure that the function is correct. The default value is `None`. **The current configuration class is used in multi-level cache EBC mode. Set Embedding initialization with the `initializer_type` parameter. The `init_fn` parameter does not take effect.**                   |
| need_pos                    | bool                                        | Optional   | Only the default value `False` is supported.                                           |
| pooling                     | torchrec.PoolingType                        | Optional   | Type of the pooling operation. The value can be:<ul><li>`SUM`: Sum. </li><li>`MEAN`: Average. </li></ul>The default value is `SUM`.|
| input_dim                   | int                                         | Optional   | Input dimension. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.2.0 and v1.5.0.                                                  |
| total_num_buckets           | int                                         | Optional   | Total number of buckets, which remains unchanged during the model lifecycle. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| use_virtual_table           | bool                                        | Optional   | Indicates whether to use a virtual table. When an NPU device is used, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| virtual_table_eviction_policy | torchrec.modules.embedding_configs.VirtualTableEvictionPolicy | Optional   | Eviction policy of the virtual table. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| enable_embedding_update     | bool                                        | Optional   | Indicates whether to enable Embedding update. When an NPU device is used, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.                                                  |
| weight_init_mean            | float                                       | Optional   | Mean of weight initialization. Use it for the `InitializerType.TRUNCATED_NORMAL` initialization type. The default value is 0.0.                                   |
| weight_init_stddev          | float                                       | Optional   | Standard deviation of weight initialization. Use it for the `InitializerType.TRUNCATED_NORMAL` initialization type. The default value is 0.05.                                 |
| initializer_type            | InitializerType                             | Optional   | Weight initialization type. Supported values are `LINEAR`, `TRUNCATED_NORMAL`, and `UNIFORM`. The default value is `LINEAR`. For details, see [InitializerType](07_multilevel_cache_management_apis.md#initializertype).<br>Descriptions of the values:<br>`InitializerType.LINEAR`: Initializes Embedding linearly using the `weight_init_min` and `weight_init_max` parameters.<br>`InitializerType.TRUNCATED_NORMAL`: Initializes Embedding with the truncated normal distribution using the `weight_init_min`, `weight_init_max`, `weight_init_mean`, and `weight_init_stddev` parameters.<br>`InitializerType.UNIFORM`: Initializes Embedding with the uniform distribution using the `weight_init_min` and `weight_init_max` parameters.             |
| admit_and_evict_config      | AdmitAndEvictConfig                         | Optional   | Feature admission and eviction configuration. **This is a reserved parameter and is not supported by the current configuration class.**                               |
| is_incremental              | bool                                        | Optional   | Indicates whether to enable incremental data processing during training. Use it with the incremental saving/loading feature. The default value is `False`, which indicates that this feature is disabled. For the incremental saving/loading feature, see [Saver](07_multilevel_cache_management_apis.md#saver).                                                       |

**Returns**

- Success: An `EmbCacheEmbeddingBagConfig` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec_embcache.distributed import EmbCacheEmbeddingBagConfig, InitializerType

emb_config = EmbCacheEmbeddingBagConfig(
    name="table0",
    embedding_dim=128,
    num_embeddings=1000,
    feature_names=["feat0"],
    # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions
    initializer_type=InitializerType.TRUNCATED_NORMAL,  # Set the embedding initialization method with the initializer_type parameter
    weight_init_mean=0.0,
    weight_init_stddev=0.05,
)
```

## `EmbCacheEmbeddingBagCollection`

**Description**

Creates a multi-level cache sparse table object with pooling and hash mapping.

**Function Prototype**

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

**Parameters**

| Parameter                    | Type                                               | Mandatory/Optional| Description                                                                                                                                                            |
|-------------------------|---------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables                  | List[EmbCacheEmbeddingBagConfig\|EmbeddingBagConfig] | Mandatory   | List of sparse table configurations. The list length must be in the range [1, 10000].                                                                                                                                |
| world_size              | int                                               | Mandatory   | `world_size` for distributed training. The value range is [1, 10000].                                                                                                                             |
| batch_size              | int                                               | Mandatory   | Batch size. The value range is [1, 102400].                                                                                                                                         |
| multi_hot_sizes         | List[int]                                         | Mandatory   | List of multi-hot encoding sizes for each feature. Use it to calculate the minimum device memory required for training. The length of this list must match the length of the `tables` list. The list length must be in the range [1, 10000]. The multi-hot encoding size range in the list is [1, 102400].                                                                              |
| is_weighted             | bool                                              | Optional   | Only the default value `False` is supported.                                                                                                                                                  |
| need_accumulate_offset  | bool                                              | Optional   | Indicates whether to accumulate offsets. The default value is `True`.                                                                                                                                            |
| device                  | torch.device                                      | Optional   | Computing device. The default value is `torch.device("cpu")`.                                                                                                                                  |
| embedding_optimizer_cls | Type[torch.optim.Optimizer]                       | Optional   | Type of the embedding optimizer. Use it to calculate the minimum device memory required for training. The default value is `torch.optim.Adagrad`. Value range:<ul><li>`torch.optim.Adagrad`: Adagrad optimizer </li><li>`torch.optim.Adam`: Adam optimizer </li><li>`torch.optim.SGD`: SGD optimizer</li><li>`torchrec.optim.Adagrad`: Adagrad optimizer</li><li>`torchrec.optim.Adam`: Adam optimizer</li><li>`torchrec.optim.SGD`: SGD optimizer</li></ul><br>The optimizer type parameter must be the same as the optimizer type passed when calling [apply_optimizer_in_backward](04_optimizers_apis.md#apply_optimizer_in_backward-torchrec). |

**Returns**

- Success: An `EmbCacheEmbeddingBagCollection` object is returned.
- Failure: An exception is thrown.

**Example**

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
    initializer_type=InitializerType.TRUNCATED_NORMAL,  # Set the embedding initialization method with the initializer_type parameter
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

## `EmbCacheEmbeddingConfig`

**Description**

Configuration class API for `EmbCacheEmbeddingCollection`. It configures table size, `dim`, data type, and other settings.

**Function Prototype**

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
    input_dim: Optional[int] = None
    total_num_buckets: Optional[int] = None
    use_virtual_table: bool = False
    virtual_table_eviction_policy: Optional[VirtualTableEvictionPolicy] = None
    enable_embedding_update: bool = False
    weight_init_mean: Optional[float] = 0.0
    weight_init_stddev: Optional[float] = 0.05
    initializer_type: InitializerType = field(default=InitializerType.LINEAR)
    admit_and_evict_config: Optional[AdmitAndEvictConfig] = field(
        default_factory=lambda: AdmitAndEvictConfig()
    )
    is_incremental: bool = False
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|num_embeddings|int|Mandatory|Number of rows in the sparse table. The value ranges from 1 to 1 billion. When `row_wise` is used, at least one sparse table must have a row count greater than or equal to the number of used devices.|
|embedding_dim|int|Mandatory|Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8. The `embedding_dim` values of multiple sparse tables must be consistent.|
|name|str|Mandatory|Name of the sparse table. The value can contain only numbers, letters, and underscores (_).|
|data_type|torchrec.types.DataType|Optional|Data type of the sparse table. Only the default value `DataType.FP32` is supported.|
|feature_names|List[str]|Mandatory|List of feature names queried by the sparse table. Each feature name can contain only numbers, letters, and underscores (_).|
|weight_init_max|float|Optional|Only the default value `None` or 1.0 is supported.|
|weight_init_min|float|Optional|Only the default value `None` or 0.0 is supported.|
|num_embeddings_post_pruning|int|Optional|Only the default value `None` is supported.|
|init_fn|Callable|Optional|You can pass a function that takes a parameter of the `torch.nn.Parameter` type. The default value is `None`. **The current configuration class is used in multi-level cache EC mode. Set Embedding initialization with the `initializer_type` parameter. The `init_fn` parameter does not take effect.**|
|need_pos|bool|Optional|Only the default value `False` is supported.|
|input_dim|int|Optional|Input dimension. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.2.0 and v1.5.0.|
|total_num_buckets|int|Optional|Total number of buckets, which remains unchanged during the model lifecycle. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.|
|use_virtual_table|bool|Optional|Indicates whether to use a virtual table. When an NPU device is used, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.|
|virtual_table_eviction_policy|torchrec.modules.embedding_configs.VirtualTableEvictionPolicy|Optional|Eviction policy of the virtual table. When an NPU device is used, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.|
|enable_embedding_update|bool|Optional|Indicates whether to enable Embedding update. When an NPU device is used, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0.|
|weight_init_mean|float|Optional|Mean of weight initialization. Use it for the `InitializerType.TRUNCATED_NORMAL` initialization type. The default value is 0.0.|
|weight_init_stddev|float|Optional|Standard deviation of weight initialization. Use it for the `InitializerType.TRUNCATED_NORMAL` initialization type. The default value is 0.05.|
|initializer_type|InitializerType|Optional|Weight initialization type. Supported values are `LINEAR`, `TRUNCATED_NORMAL`, and `UNIFORM`. The default value is `LINEAR`.<br>Descriptions of the values:<br>`InitializerType.LINEAR`: Initializes Embedding linearly using the `weight_init_min` and `weight_init_max` parameters.<br>`InitializerType.TRUNCATED_NORMAL`: Initializes Embedding with the truncated normal distribution using the `weight_init_min`, `weight_init_max`, `weight_init_mean`, and `weight_init_stddev` parameters.<br>`InitializerType.UNIFORM`: Initializes Embedding with the uniform distribution using the `weight_init_min` and `weight_init_max` parameters.|
|admit_and_evict_config|AdmitAndEvictConfig|Optional|Feature admission and eviction configuration. Admission and eviction are disabled by default.|
|is_incremental|bool|Optional|Indicates whether to enable incremental data processing during training. Use it with the incremental saving/loading feature. The default value is `False`, which indicates that this feature is disabled. For the incremental saving/loading feature, see [Saver](07_multilevel_cache_management_apis.md#saver).|

**Returns**

- Success: An `EmbCacheEmbeddingConfig` object is returned.
- Failure: An exception is thrown.

**Example**

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

## `EmbCacheEmbeddingCollection`

**Description**

Creates a sparse table object with hash mapping in multi-level cache mode.

**Function Prototype**

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

**Parameters**

| Parameter                    | Type                                            | Mandatory/Optional| Description                                                                                                                                                           |
|-------------------------|------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables                  | List[EmbCacheEmbeddingConfig\|EmbeddingConfig] | Mandatory   | List of sparse table configuration objects. The list length must be in the range [1, 10000].                                                                                                                                 |
| world_size              | int                                            | Mandatory   | `world_size` for distributed training. The value range is [1, 10000].                                                                                                                            |
| batch_size              | int                                            | Mandatory   | Batch size. The value range is [1, 102400].                                                                                                                                        |
| multi_hot_sizes         | List[int]                                      | Mandatory   | List of multi-hot encoding sizes for each feature. Use it to calculate the minimum device memory required for training. The length of this list must match the length of the `tables` list. The list length must be in the range [1, 10000]. The multi-hot encoding size range in the list is [1, 102400].                                                                             |
| need_indices            | bool                                           | Optional   | Indicates whether indexes are required. The default value is `False`.                                                                                                                                             |
| need_accumulate_offset  | bool                                           | Optional   | Indicates whether to accumulate offsets. The default value is `True`.                                                                                                                                           |
| device                  | torch.device                                   | Optional   | Computing device. The default value is `torch.device("cpu")`.                                                                                                                                 |
| embedding_optimizer_cls | Type[torch.optim.Optimizer]                    | Optional   | Type of the embedding optimizer. Use it to calculate the minimum device memory required for training. The default value is `torch.optim.Adagrad`. Value range:<ul><li>`torch.optim.Adagrad`: Adagrad optimizer </li><li>`torch.optim.Adam`: Adam optimizer </li><li>`torch.optim.SGD`: SGD optimizer</li><li>`torchrec.optim.Adagrad`: Adagrad optimizer</li><li>`torchrec.optim.Adam`: Adam optimizer</li><li>`torchrec.optim.SGD`: SGD optimizer</li><li>`torchrec.optim.AccumulateAdagrad`: Adagrad optimizer with gradient accumulation</li><li>`torchrec.optim.AccumulateAdam`: Adam optimizer with gradient accumulation</li><li>`torchrec.optim.AccumulateSGD`: SGD optimizer with gradient accumulation</li></ul><br>The optimizer type parameter must be the same as the optimizer type passed when calling [apply_optimizer_in_backward](04_optimizers_apis.md#apply_optimizer_in_backward-torchrec). |

**Returns**

- Success: An `EmbCacheEmbeddingCollection` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from typing import List

import torch
from torchrec_embcache.distributed import EmbCacheEmbeddingCollection, EmbCacheEmbeddingConfig

# embedding_configs is a list of EmbCacheEmbeddingConfig objects. The detailed definitions are omitted here
embedding_configs: List[EmbCacheEmbeddingConfig] = ......
# The optimizer type here must be the same as the optimizer type passed to torchrec.optim.apply_optimizer_in_backward
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
