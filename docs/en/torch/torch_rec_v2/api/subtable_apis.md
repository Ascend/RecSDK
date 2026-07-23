# Sharding APIs

## `ShardingEnv` (TorchRec)

>[!NOTE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Stores distributed parameters.

**Function Prototype**

```python
class ShardingEnv:
    def __init__(**kwargs):
def from_process_group(cls, pg: dist.ProcessGroup) -> "ShardingEnv":
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|----|--|
|world_size|int|Mandatory|Number of devices to use. The value range is [1, 8].|
|rank|int|Mandatory|Current device number. The value range is [0, world_size-1].|
|pg|dist.ProcessGroup|Mandatory|Distributed communication link. Only links with the `hccl` or `gloo` backend are supported. <div class="notice"><span class="noticetitle">Note</span><div class="notebody">In PyTorch, the backend name of `hccl` is `custom`.</div></div>|
|output_dtensor|bool|Optional|Only the default value `False` is supported.|

**Example**

```python
import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv
host_gp = dist.new_group(backend="gloo")
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `Topology` (TorchRec)

>[!NOTE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Stores distributed environment parameters.

**Function Prototype**

```python
class Topology:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional| Description                                               |
|--|--|--|---------------------------------------------------|
|world_size|int|Mandatory| Number of devices to use. The value range is [1, 8].                                 |
|compute_device|str|Mandatory| Device name. When you use an NPU device, set this to `"npu"`.                   |
|hbm_cap|int|Optional| When you use an NPU device, only the default value `None` is supported.                            |
|ddr_cap|int|Optional| When you use an NPU device, only the default value `None` is supported.                            |
|local_world_size|int|Optional| When you use an NPU device, only the default value `None` is supported.|
|hbm_mem_bw|float|Optional| When you use an NPU device, only the default value `(897 * 1024 * 1024 * 1024 / 1000)` is supported.|
|ddr_mem_bw|float|Optional| When you use an NPU device, only the default value `(51 * 1024 * 1024 * 1024 / 1000)` is supported.|
|hbm_to_ddr_mem_bw|float|Optional| When you use an NPU device, only the default value `(32 * 1024 * 1024 * 1024 / 1000)` is supported.|
|intra_host_bw|float|Optional| When you use an NPU device, only the default value `(600 * 1024 * 1024 * 1024 / 1000)` is supported.|
|inter_host_bw|float|Optional| When you use an NPU device, only the default value `(12.5 * 1024 * 1024 * 1024 / 1000)` is supported.|
|bwd_compute_multiplier|float|Optional| When you use an NPU device, only the default value `2` is supported.                               |
|custom_topology_data|torchrec.distribute.planner.types.CustomTopologyData|Optional| When you use an NPU device, only the default value `None` is supported.                            |
|weighted_feature_bwd_compute_multiplier|float|Optional| When you use an NPU device, only the default value `1` is supported.                               |
|uneven_sharding_perf_multiplier|float|Optional| When you use an NPU device, only the default value `1` is supported.                               |

**Example**

```python
from torchrec.distributed.planner import Topology,
topo = Topology(world_size=world_size, compute_device="npu")
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbeddingCollectionSharder`

**Description**

An embedding table sharder that inherits from `EmbeddingCollectionSharder` of TorchRec. It works exactly the same as the native class. This API mainly overrides the index deduplication process so it can work with dynamic embedding tables.

**Function Prototype**

```python
class DynamicEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|fused_params|Optional[Dict[str, Any]]|Optional|Parameters used to configure fused embedding operations, such as optimizer state and learning rate. The default value is `None`.|
|qcomm_codecs_registry|Optional[Dict[str, QuantizedCommCodecs]]|Optional|Registry used to register quantized communication codecs. Quantized communication can reduce communication volume. The default value is `None`.|
|use_index_dedup|bool|Optional|Indicates whether to use local unique. The default value is `False`.|

**Example**

```python
from dynamic_emb.distributed.embedding import DynamicEmbeddingCollectionSharder
eb_sharder = DynamicEmbeddingCollectionSharder(
    fused_params=fused_params,
    use_index_dedup=False,
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbParameterConstraints`

**Description**

Constraints for dynamic embedding tables. Inherits from the base `ParameterConstraints` of TorchRec.

**Function Prototype**

```python
class DynamicEmbParameterConstraints(ParameterConstraints):
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                                                                                                                                                                |
|--|--|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|use_dynamicemb|bool|Optional|Indicates whether to enable dynamic sparse tables. When you use an NPU device, only the default value `True` is supported.|
|dynamicemb_options|DynamicEmbTableOptions|Optional|Specific options for dynamic sparse tables. See the value range of `DynamicEmbTableOptions`.|
|sharding_type|List[str]|Optional|Sharding type. This parameter is required when you use an NPU device. The value range is:<ul><li>`"row_wise"`: Shards by row index. </li></ul><div class="note"><span class="notetitle">Note</span><div class="notebody">Mixing different sharding types is not supported.</div></div> |
|compute_kernels|List[str]| Optional   | Computing kernel type. This parameter is required when you use an NPU device. The value range is:<ul><li>`"fused"`: Queries with a fused table. This mode is used only when `sharding_type` is `"row_wise"`.</li></ul>|
|min_partition|List[int]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|pooling_factors|List[float]| Optional   | When you use an NPU device, only the default value `POOLING_FACTOR` is supported.                                                                                                                                                                   |
|num_poolings|List[float]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|batch_sizes|List[int]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|is_weighted|bool| Optional   | When you use an NPU device, only the default value `False` is supported.                                                                                                                                                                            |
|cache_params|torchrec.distributed.types.CacheParams| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|enforce_hbm|bool| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|stochastic_rounding|bool| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|bounds_check_mode|enum.IntEnum| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|feature_names|List[str]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|output_dtype|Enum| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|device_group|str| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|key_value_params|torchrec.distributed.types.KeyValueParams| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |

**Example**

```python
from dynamic_emb.distributed.planner.types import DynamicEmbParameterConstraints
constraints = {
    "table0": DynamicEmbParameterConstraints(
        sharding_types=[ShardingType.ROW_WISE.value],
        compute_kernels=["fused"],
        dynamicemb_options=DynamicEmbTableOptions(),
    ),
}
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbTableOptions`

**Description**

Dynamic embedding table parameter class used to configure parameters for each dynamic embedding table. These parameters serve as input to `DynamicEmbParameterConstraints`.

**Function Prototype**

```python
@dataclass
class DynamicEmbTableOptions(_ContextOptions):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|training|bool|Optional|Flag that indicates whether the dynamic embedding table is in training or evaluation mode. The default value is `True`. In training mode, DynamicEmb stores the embedding and optimizer state together in the underlying key-value table.|
|initializer_args|DynamicEmbInitializerArgs|Optional|Parameters used to initialize dynamic embedding vectors in training mode. The default value is a `DynamicEmbInitializerArgs` instance. See the value range of `DynamicEmbInitializerArgs`.|
|eval_initializer_args|DynamicEmbInitializerArgs|Optional|Parameters used to initialize dynamic embedding vectors in evaluation mode. The default value is a `DynamicEmbInitializerArgs` instance, and only `DynamicEmbInitializerMode.CONSTANT` is supported.|
|caching|bool|Optional|Indicates whether to enable cache mode. Only the default value `False` is supported.|
|init_capacity|Optional[int]|Optional|Initial table capacity on a single NPU. If this parameter is not set, the default value is the sharded `max_capacity`. If you set it, the value is rounded up to the next power of two. The value range is [0,MAX_INT32). Ensure that the memory usage fits your environment.|
|max_load_factor|float|Optional|Maximum load factor that triggers rehash. The default value is 0.5. The value range is (0.0,1.0).|
|score_strategy|DynamicEmbScoreStrategy|Optional|Score assigned to each key for eviction. The default value is `DynamicEmbScoreStrategy.TIMESTAMP`. Currently, `DynamicEmbScoreStrategy.CUSTOMIZED` is not supported.|
|bucket_capacity|int|Optional|Capacity of each bucket in HKV. The default value is 128. If you set this parameter, the value is rounded up to the next power of two. When a bucket is full, the key with the lowest score in the bucket is evicted and its slot is used for the new key. A larger bucket capacity makes score-based eviction more accurate, but it also affects performance. The value range is [16,1024]. Ensure that the memory usage fits your environment.|
|safe_check_mode|DynamicEmbCheckMode|Optional|Indicates whether to enable insertion safety checks. The default value is `DynamicEmbCheckMode.IGNORE`.|
|global_hbm_for_values|int|Optional|Total amount of NPU memory, in bytes, used to store the embedding and optimizer state. When the default value 0 is used, the planner sets it to `byte size of the value type * number of sharded rows (rounded up to the next power of two) * number of sharded columns`.|
|external_storage|Storage|Optional|External storage or parameter server that replaces the default `KeyValueTable`. Only the default value `None` is supported.|
|index_type|torch.dtype|Optional|Index type of sparse features. Only the default value `torch.int64` is supported.|

**Example**

```python
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

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbInitializerMode`

**Description**

An enumeration class that defines the initialization method for each embedding vector in a dynamic embedding table. It supports uniform, normal, and constant initialization, and uses uniform distribution by default.

**Function Prototype**

```python
class DynamicEmbInitializerMode(enum.Enum):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                   |
|--|--|-------|-------------------------------------------------------|
|NORMAL|str|-| Initializes embedding vectors with a normal distribution.|
|UNIFORM|str|-|Initializes embedding vectors with a uniform distribution.|
|CONSTANT|str|-|Initializes the values of all embedding vectors to a given constant.|

**Example**

```python
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
initializer_args=DynamicEmbInitializerArgs(
    mode=DynamicEmbInitializerMode.NORMAL,
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbInitializerArgs`

**Description**

A dataclass for the parameters of each initialization mode in `DynamicEmbInitializerMode`.

**Function Prototype**

```python
@dataclass
class DynamicEmbInitializerArgs:
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                 |
|--|--|-------|-----------------------------------------------------|
|mode|DynamicEmbInitializerMode|Optional|Initialization mode. The value can be `NORMAL`, `UNIFORM`, or `CONSTANT`. The default value is `UNIFORM`.|
|mean|float|Optional|Mean of the normal distribution. The default value is 0.0.|
|std_dev|float|Optional|Standard deviation of the normal distribution. The default value is 1.0.|
|lower|float|Optional|Lower bound of the uniform distribution. The default value is `None`.|
|upper|float|Optional|Upper bound of the uniform distribution. The default value is `None`.|
|value|float|Optional|Fixed value used for constant initialization. The default value is 0.0.|

**Example**

```python
from dynamic_emb.distributed.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
)
initializer_args=DynamicEmbInitializerArgs(
    mode=DynamicEmbInitializerMode.CONSTANT,
    value=0.1,
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbScoreStrategy`

**Description**

An enumeration class for the HKV scoring mechanism, used for customized eviction of sparse features.

**Function Prototype**

```python
class DynamicEmbScoreStrategy(enum.IntEnum):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                            |
|--|--|-------|----------------------------------------------------------------|
|TIMESTAMP|int|-|Uses the global nanosecond timestamp of the device as the score.|
|STEP|int|-|Uses the step counter inside the embedding table as the score.|
|CUSTOMIZED|int|-|Uses a fully customized score.|
|LFU|int|-|Automatically calculates the score based on the access frequency of the embedding item.|

**Example**

```python
table_options = DynamicEmbTableOptions(
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `EmbOptimType`

**Description**

An enumeration class for dynamic embedding table optimizers.

**Function Prototype**

```python
@enum.unique
class EmbOptimType(enum.Enum):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                            |
|--|--|-------|----------------------------------------------------------------|
|ADAM|str|-|Adam optimizer for embedding tables.|
|ADAMW|str|-|AdamW optimizer for embedding tables.|

**Example**

```python
from dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer import
table_options = [
    DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        optimizer_type=EmbOptimType.ADAM,
    )
]
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbeddingEnumerator`

**Description**

Inherits from `EmbeddingEnumerator` of TorchRec and works exactly the same way. When it enumerates a sharding plan, it distinguishes between regular and dynamic embedding tables.

**Function Prototype**

```python
class DynamicEmbeddingEnumerator(EmbeddingEnumerator):
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|topology|Topology|Mandatory|Topology of GPU and host memory, used to optimize distributed or cross-device memory access. See the value range of `Topology`.|
|batch_size|Optional[int]|Optional|Training batch size. The behavior is the same as the parameter with the same name in TorchRec. The default value is 512. The value range is [1,1000000].|
|constraints|Optional[Dict[str, DynamicEmbParameterConstraints]]|Optional|Dictionary of constraint configurations for dynamic embedding table parameters, with the parameter name as the key and the corresponding constraint object as the value. Use this to define initialization, caching policy, capacity, and so on. The default value is `None`. See the value range of `DynamicEmbParameterConstraints`.|
|estimator|Optional[Union[ShardEstimator, List[ShardEstimator]]]|Optional|Estimator used to estimate sharding size. One or more estimators are supported, and the behavior is the same as in TorchRec. Only the default value `None` is supported.|
|use_exact_enumerate_order|Optional[bool]|Optional|Indicates whether to enumerate sharded parameters strictly in the dictionary order of model parameter names (`named_children`). If the value is `True`, the parameter traversal order matches the model structure. Only the default value `False` is supported.|

**Example**

```python
from dynamic_emb.distributed.planner.enumerators import DynamicEmbeddingEnumerator
enumerator=DynamicEmbeddingEnumerator(
    topology=topology,
    constraints=constraints,
)

```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbeddingShardingPlanner`

**Description**

A wrapper around `EmbeddingShardingPlanner` of TorchRec. Unlike `EmbeddingShardingPlanner`, this wrapper also accepts the additional `eb_configs` parameter to plan the capacity configuration of dynamic embedding tables.

**Function Prototype**

```python
class DynamicEmbeddingShardingPlanner:
     def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|eb_configs|List[BaseEmbeddingConfig]|Mandatory|List of all embedding table configurations in the TorchRec model, used for dynamic embedding planning. Currently, only `EmbeddingConfig` is supported. See the value range of `EmbeddingConfig`.|
|constraints|Dict[str, DynamicEmbParameterConstraints]|Mandatory|Dictionary of constraint configurations for each embedding table, with the table name as the key and the constraint object as the value. See the value range of `DynamicEmbParameterConstraints`.|
|topology|Optional[Topology]|Optional|Topology of GPU and host memory. If the value is `None`, the default topology is used. See the value range of `Topology`.|
|batch_size|Optional[int]|Optional|Training batch size. If the value is `None`, the default value 512 is used. The value range is [1,1000000].|
|enumerator|Optional[Enumerator]|Optional|Enumerator used for sharding. If the value is `None`, the default enumerator is used. The default value is `None`.|
|storage_reservation|Optional[StorageReservation]|Optional|Storage reservation information used to reserve part of the memory during planning. Only the default value `None` is supported.|
|proposer|Optional[Union[Proposer, List[Proposer]]]|Optional|Sharding plan proposer. Only the default value `None` is supported.|
|partitioner|Optional[Partitioner]|Optional|Partitioner used to shard embedding tables. Only the default value `None` is supported.|
|performance_model|Optional[PerfModel]|Optional|Performance model used to evaluate the efficiency of different sharding plans. Only the default value `None` is supported.|
|stats|Optional[Union[Stats, List[Stats]]]|Optional|Statistics collector. Only the default value `None` is supported.|
|debug|bool|Optional|Indicates whether to enable debug mode. When debug mode is enabled, more log information is output. Only the default value `True` is supported.|

**Example**

```python
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

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DynamicEmbCheckMode`

**Description**

An enumeration class for the safe-check mode of embedding table index insertion. When a dynamic embedding table has a small capacity, a single feature with a large number of indexes may cause the hash table to fail to insert indexes. After you enable safe checks, you can observe such cases, including the number of failed insertions and the number of indexes that fail each time, so you can determine whether the capacity of the dynamic embedding table is too low.

**Function Prototype**

```python
@enum.unique
class DynamicEmbCheckMode(enum.IntEnum):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|ERROR|int|-|If an index cannot be inserted successfully, a runtime error is raised with the number of failed insertions.|
|WARNING|int|-|If an index cannot be inserted successfully, a warning is output with the number of failed insertions.|
|IGNORE|int|-|Does not check whether insertion succeeds.|

**Example**

```python
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

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
