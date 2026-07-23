# Table Creation APIs

## `HashEmbeddingBagConfig`

**Description**

Input to `HashEmbeddingBagCollection`. Use it to configure table size, `dim`, data type, and other settings.

**Function Prototype**

```python
@dataclass
class HashEmbeddingBagConfig:
    def __init__(**kwargs):
```

**Parameters**

| Parameter                 | Type                           | Mandatory/Optional| Description                                                                                       |
|-----------------------------|---------------------------------------------|-------|-------------------------------------------------------------------------------------------|
| num_embeddings              | int                                         | Mandatory   | Number of rows in the sparse table. The value ranges from 1 to 1 billion.                                                                    |
| embedding_dim               | int                                         | Mandatory   | Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8.                                                         |
| name                        | str                                         | Mandatory   | Name of the sparse table. The value can contain only numbers, letters, and underscores (_).                                                                    |
| data_type                   | torchrec.types.DataType                     | Optional   | Data type of the sparse table. Only the default value `DataType.FP32` is supported.                                                           |
| feature_names               | List[str]                                   | Mandatory   | Names of the features queried by the sparse table. The value can contain only numbers, letters, and underscores (_).                                                                |
| weight_init_max             | float                                       | Optional   | Only the default value `None` or 1.0 is supported.                                                                |
| weight_init_min             | float                                       | Optional   | Only the default value `None` or 0.0 is supported.                                                                |
| num_embeddings_post_pruning | int                                         | Optional   | Only the default value `None` is supported.                                                                    |
| init_fn                     | Callable                                    | Optional   | A function of the `nn.Parameter` type can be passed. You need to ensure that the function is correct. The default value is `None`.                                           |
| need_pos                    | bool                                        | Optional   | Only the default value `False` is supported.                                                                   |
| pooling                     | torchrec.modules.embedding_configs.PoolType | Optional   | Type of the pooling operation. The value can be:<ul><li>`SUM`: Sum. </li><li>`MEAN`: Average. </li><li>`NONE`: Does not perform pooling. </li></ul> The default value is `SUM`.|

**Returns**

- Success: A `HashEmbeddingBagConfig` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from hybrid_torchrec import HashEmbeddingBagConfig, HashEmbeddingBagCollection
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

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `HashEmbeddingBagCollection`

**Description**

Creates a single-node table object with pooling and hash mapping.

**Function Prototype**

```python
class HashEmbeddingBagCollection:
    def __init__(**kwargs):
```

**Parameters**

| Parameter        | Type                                                | Mandatory/Optional| Description                                                                                                                                                                                                                                                                                                     |
|-------------|----------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables      | List[HashEmbeddingBagConfig \| EmbeddingBagConfig] | Mandatory   | List of sparse table configuration objects. The list length must be in the range [1, 10000]. <p>See `HashEmbeddingBagConfig` for parameter details.</p>                                                                                                                                                                                                                                     |
| is_weighted | bool                                               | Optional   | Only the default value `False` is supported.                                                                                                                                                                                                                                                                                          |
| device      | str or `torch.device`                                 | Optional   | Device for the sparse table. The default value is `torch.device("cpu")`.<br>If the type is a string, supported values are:<ul><li>`"npu"`: NPU device. </li><li>`"meta"`: Meta device. </li><li>`"cpu"`: CPU device. The CPU device does not support distributed tables and supports only single-node tables.</li></ul><br>If the type is `torch.device`, supported values are:<ul><li>`torch.device("npu")`: NPU device. </li><li>`torch.device("meta")`: Meta device. </li><li>`torch.device("cpu")`: CPU device. The CPU device does not support distributed tables and supports only single-node tables.</li></ul> |

**Returns**

- Success: A `HashEmbeddingBagCollection` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from hybrid_torchrec import HashEmbeddingBagConfig, HashEmbeddingBagCollection

table_configs: list[HashEmbeddingBagConfig] = xx
ebc = HashEmbeddingBagCollection(device="npu", tables=table_configs)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `EmbCacheEmbeddingBagConfig`

**Description**

Configuration class API for `EmbCacheEmbeddingBagCollection`. It configures table size, `dim`, data type, and other settings.

**Function Prototype**

```python
class EmbCacheEmbeddingBagConfig:
    def __init__(**kwargs):
```

**Parameters**

| Parameter                        | Type                                         | Mandatory/Optional| Description                                                               |
|-----------------------------|---------------------------------------------|-------|-------------------------------------------------------------------|
| num_embeddings              | int                                         | Mandatory   | Number of rows in the sparse table. The value ranges from 1 to 1 billion. When `row_wise` is used, at least one sparse table must have a row count greater than or equal to the number of used devices.                |
| embedding_dim               | int                                         | Mandatory   | Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8.                                 |
| name                        | str                                         | Mandatory   | Name of the sparse table. The value can contain only numbers, letters, and underscores (_).                                            |
| data_type                   | torchrec.types.DataType                     | Optional   | Data type of the sparse table. Only the default value `DataType.FP32` is supported.                                   |
| feature_names               | List[str]                                   | Mandatory   | Names of the features queried by the sparse table. The value can contain only numbers, letters, and underscores (_).                                        |
| weight_init_max             | float                                       | Optional   | Only the default value `None` or 1.0 is supported.                                        |
| weight_init_min             | float                                       | Optional   | Only the default value `None` or 0.0 is supported.                                        |
| num_embeddings_post_pruning | int                                         | Optional   | Only the default value `None` is supported.                                            |
| init_fn                     | Callable                                    | Optional   | A function of the `nn.Parameter` type can be passed. You need to ensure that the function is correct. The default value is `None`.                   |
| need_pos                    | bool                                        | Optional   | Only the default value `False` is supported.                                           |
| pooling                     | torchrec.modules.embedding_configs.PoolType | Optional   | Type of the pooling operation. The value can be:<ul><li>`SUM`: Sum. </li><li>`MEAN`: Average. </li></ul>The default value is `SUM`.|
| weight_init_mean            | float                                       | Optional   | Mean of weight initialization. Use it for the `UNIFORM` initialization type. The default value is 0.0.                                   |
| weight_init_stddev          | float                                       | Optional   | Standard deviation of weight initialization. Use it for the `UNIFORM` initialization type. The default value is 0.05.                                 |
| initializer_type            | InitializerType                             | Optional   | Weight initialization type. Supported values are `LINEAR`, `TRUNCATED_NORMAL`, and `UNIFORM`. The default value is `LINEAR`.             |
| admit_and_evict_config      | AdmitAndEvictConfig                         | Optional   | Feature admission and eviction configuration. Admission and eviction are disabled by default. This parameter is reserved and is not supported currently.                              |
| is_incremental              | bool                                        | Optional   | Enables incremental saving and loading.                                                      |

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
    # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
    initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
    weight_init_mean=0.0,
    weight_init_stddev=0.05,
)
```

## `EmbCacheEmbeddingBagCollection`

**Description**

Creates a single-node table object with pooling, hash mapping, and multi-level cache.

**Function Prototype**

```python
class EmbCacheEmbeddingBagCollection:
    def __init__(**kwargs):
```

**Parameters**

| Parameter                    | Type                                               | Mandatory/Optional| Description                                                                                                                                                            |
|-------------------------|---------------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables                  | List[EmbCacheEmbeddingBagConfig\|EmbeddingConfig] | Mandatory   | List of sparse table configuration files. The list length must be in the range [1, 10000].                                                                                                                                |
| world_size              | int                                               | Mandatory   | `world_size` for distributed training. The value range is [1, 10000].                                                                                                                             |
| batch_size              | int                                               | Mandatory   | Batch size. The value range is [1, 102400].                                                                                                                                         |
| multi_hot_sizes         | List[int]                                         | Mandatory   | List of multi-hot encoding sizes for each feature. The length of this list must match the length of the `tables` list. The list length must be in the range [1, 10000]. The multi-hot encoding size range in the list is [1, 102400].                                                                              |
| is_weighted             | bool                                              | Optional   | Only the default value `False` is supported.                                                                                                                                                  |
| need_accumulate_offset  | bool                                              | Optional   | Indicates whether to accumulate offsets. The default value is `True`.                                                                                                                                            |
| device                  | torch.device                                      | Optional   | Computing device. The default value is `torch.device("cpu")`.                                                                                                                                  |
| embedding_optimizer_cls | Type[torch.optim.Optimizer]                       | Optional   | Type of the embedding optimizer. The default value is `torch.optim.Adagrad`. Value range:<ul><li>`torch.optim.Adagrad`: Adagrad optimizer </li><li>`torch.optim.Adam`: Adam optimizer </li><li>`torch.optim.SGD`: SGD optimizer</li></ul> |

**Returns**

- Success: An `EmbCacheEmbeddingBagCollection` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from typing import List
from torchrec_embcache.distributed import EmbCacheEmbeddingBagCollection

embedding_configs: List[EmbCacheEmbeddingBagConfig] = xx
world_size: int = 2
batch_size: int = 128
table_num = len(embedding_configs)
sparse_ebc: torch.nn.Module = EmbCacheEmbeddingBagCollection(
    embedding_configs,
    world_size,
    batch_size,
    multi_hot_sizes=[1] * table_num,
    device=torch.device("meta"),
)
```

## `EmbCacheEmbeddingConfig`

**Description**

Configuration class API for `EmbCacheEmbeddingCollection`. It configures table size, `dim`, data type, and other settings.

**Function Prototype**

```python
class EmbCacheEmbeddingConfig:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|num_embeddings|int|Mandatory|Number of rows in the sparse table. The value ranges from 1 to 1 billion. When `row_wise` is used, at least one sparse table must have a row count greater than or equal to the number of used devices.|
|embedding_dim|int|Mandatory|Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8.|
|name|str|Mandatory|Name of the sparse table. The value can contain only numbers, letters, and underscores (_).|
|data_type|torchrec.types.DataType|Optional|Data type of the sparse table. Only the default value `DataType.FP32` is supported.|
|feature_names|List[str]|Mandatory|Names of the features queried by the sparse table. The value can contain only numbers, letters, and underscores (_).|
|weight_init_max|float|Optional|Only the default value `None` or 1.0 is supported.|
|weight_init_min|float|Optional|Only the default value `None` or 0.0 is supported.|
|num_embeddings_post_pruning|int|Optional|Only the default value `None` is supported.|
|init_fn|Callable|Optional|A function of the `nn.Parameter` type can be passed. You need to ensure that the function is correct. The default value is `None`.|
|need_pos|bool|Optional|Only the default value `False` is supported.|
|weight_init_mean|float|Optional|Mean of weight initialization. Use it for the `UNIFORM` initialization type. The default value is 0.0.|
|weight_init_stddev|float|Optional|Standard deviation of weight initialization. Use it for the `UNIFORM` initialization type. The default value is 0.05.|
|initializer_type|InitializerType|Optional|Weight initialization type. Supported values are `LINEAR`, `TRUNCATED_NORMAL`, and `UNIFORM`. The default value is `LINEAR`.|
|admit_and_evict_config|AdmitAndEvictConfig|Optional|Feature admission and eviction configuration. Admission and eviction are disabled by default.|
|is_incremental|bool|Optional| Enables incremental saving and loading.                     |

**Returns**

- Success: An `EmbCacheEmbeddingConfig` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec_embcache.distributed import AdmitAndEvictConfig, EmbCacheEmbeddingConfig, InitializerType

emb_config = EmbCacheEmbeddingConfig(
    name="table0",
    embedding_dim=128,
    num_embeddings=1000,
    feature_names=["feat0"],
    # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
    initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
    weight_init_mean=0.0,
    weight_init_stddev=0.05,
    admit_and_evict_config=admit_and_evict_config,  # Pass the admission and eviction configuration parameters.
)
```

## `EmbCacheEmbeddingCollection`

**Description**

Creates a single-node table object with hash mapping and multi-level cache.

**Function Prototype**

```python
class EmbCacheEmbeddingCollection:
    def __init__(**kwargs):
```

**Parameters**

| Parameter                    | Type                                            | Mandatory/Optional| Description                                                                                                                                                           |
|-------------------------|------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tables                  | List[EmbCacheEmbeddingConfig\|EmbeddingConfig] | Mandatory   | List of sparse table configuration objects. The list length must be in the range [1, 10000].                                                                                                                                 |
| world_size              | int                                            | Mandatory   | `world_size` for distributed training. The value range is [1, 10000].                                                                                                                            |
| batch_size              | int                                            | Mandatory   | Batch size. The value range is [1, 102400].                                                                                                                                        |
| multi_hot_sizes         | List[int]                                      | Mandatory   | List of multi-hot encoding sizes for each feature. The length of this list must match the length of the `tables` list. The list length must be in the range [1, 10000]. The multi-hot encoding size range in the list is [1, 102400].                                                                             |
| need_indices            | bool                                           | Optional   | Indicates whether indexes are required. The default value is `False`.                                                                                                                                             |
| need_accumulate_offset  | bool                                           | Optional   | Indicates whether to accumulate offsets. The default value is `True`.                                                                                                                                           |
| device                  | torch.device                                   | Optional   | Computing device. The default value is `torch.device("cpu")`.                                                                                                                                 |
| embedding_optimizer_cls | Type[torch.optim.Optimizer]                    | Optional   | Type of the embedding optimizer. The default value is `torch.optim.Adagrad`. Value range:<ul><li>`torch.optim.Adagrad`: Adagrad optimizer </li><li>`torch.optim.Adam`: Adam optimizer </li><li>`torch.optim.SGD`: SGD optimizer</li></ul> |

**Returns**

- Success: An `EmbCacheEmbeddingCollection` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec_embcache.distributed import EmbCacheEmbeddingCollection

embedding_configs: List[EmbCacheEmbeddingConfig] = xx
sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
    embedding_configs,  # type: ignore
    2,
    128,
    multi_hot_sizes=[1] * 2,
    device=torch.device("meta"),
)
```
