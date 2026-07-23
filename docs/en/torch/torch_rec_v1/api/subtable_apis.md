# Sharding APIs

## `ShardingEnv` (TorchRec)

>[!NOTICE]
>
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
|--|---|---|--|
|world_size|int|Mandatory|Number of devices to use. The value range is [1, 8].|
|rank|int|Mandatory|Current device number. The value range is [0, world_size-1].|
|pg|dist.ProcessGroup|Mandatory|Distributed communication link. Only links with the `hccl` or `gloo` backend are supported. In PyTorch, the backend name of `hccl` is `custom`.|
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

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Stores network device topology parameters in the distributed environment.

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
|local_world_size|int|Optional| When you use an NPU device, only the default value `None` is supported.                            |
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

## `ParameterConstraints` (TorchRec)

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Specifies the query scope for the sharding plan.

**Function Prototype**

```python
class ParameterConstraints:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                                                                                                                                                                         |
|--|--|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|sharding_type|List[str]| Optional   | Sharding type. This parameter is required when you use an NPU device. The value range is:<ul><li>`"row_wise"`: Shards by row index. </li><li>`"data_parallel"`: Keeps one table replica on each rank. </li></ul><div class="note"><span class="notetitle">Note</span><div class="notebody">Mixing different sharding types is not supported.</div></div> |
|compute_kernels|List[str]| Optional   | Computing kernel type. This parameter is required when you use an NPU device. The value range is:<ul><li>`"fused"`: Queries with a fused table. This mode is used only when `sharding_type` is `"row_wise"`. </li><li>`"dense"`: Queries with sharded tables. This mode is used only when `sharding_type` is `"data_parallel"`.</li></ul>                                      |
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
from torchrec.distributed.planner import ParameterConstraints
constraints = {
    "table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
}
```

## `get_default_hybrid_sharders`

**Description**

Retrieves the sharders.

**Function Prototype**

```python
def get_default_hybrid_sharders(host_env: ShardingEnv) -> List[ModuleSharder[nn.Module]]:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|host_env|ShardingEnv|Mandatory|Communication domain required for the host connection. For details about how to create a host, see [Step 3](../quick_start.md#interface-call-introduction). Only the `"gloo"` backend is supported.|

**Returns**

Success: A list of supported sharders is returned.

Failure: An exception is thrown.

**Example**

```python
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from torchrec.distributed.types import ShardingEnv
import torch.distributed as dist

host_gp = dist.new_group(backend="gloo")
world_size = dist.get_world_size()
rank = dist.get_rank()
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
hybrid_sharders = get_default_hybrid_sharders(host_env=host_env)
```

## `EmbeddingShardingPlanner` (TorchRec)

### Initialization

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Creates a sharding planner to find the most suitable sharding plan.

**Function Prototype**

```python
class EmbeddingShardingPlanner:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                   |
|--|--|-------|-------------------------------------------------------|
|topology|Topology| Optional   | See the value range of `Topology` (TorchRec). When you use an NPU device, this parameter is required.                           |
|constraints|Dict[str, ParameterConstraints]| Optional   | See the value range of `ParameterConstraints` (TorchRec). When you use an NPU device, this parameter is required.|
|batch_size|int| Optional   | The value range is [1, 1000000].                                   |
|enumerator|torchrec.distributed.planner.types.Enumerator| Optional   | When you use an NPU device, only the default value `None` is supported.                                |
|storage_reservation|torchrec.distributed.planner.types.StorageReservation| Optional   | When you use an NPU device, only the default value `None` is supported.                                |
|proposer|torchrec.distributed.planner.types.Proposer| Optional   | When you use an NPU device, only the default value `None` is supported.                                |
|partitioner|torchrec.distributed.planner.types.Partitioner| Optional   | When you use an NPU device, only the default value `None` is supported.                                |
|performance_model|torchrec.distributed.planner.types.PerfModel| Optional   | When you use an NPU device, only the default value `None` is supported.                                |
|stats|torchrec.distributed.planner.types.Stats| Optional   | When you use an NPU device, only the default value `None` is supported.                                |
|debug|bool| Optional   | When you use an NPU device, only the default value `True` is supported.                                |
|callbacks|List[Callable]| Optional   | When you use an NPU device, only the default value `None` is supported.                                |

**Example**

```python
from torchrec.distributed.planner import EmbeddingShardingPlanner
planner = EmbeddingShardingPlanner(
    topology=topology,
    constraints=constraints,
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

### `collective_plan`

>[!NOTICE]
>
>The APIs in this class are TorchRec open-source APIs, not external Rec SDK Torch APIs. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Searches for the most suitable sharding plan.

**Function Prototype**

```python
def collective_plan(
    module: nn.Module,
    sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
    pg: Optional[dist.ProcessGroup] = None,
)
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                 |
|--|--|-------|-----------------------------------------------------|
|module|nn.Module| Optional   | When you use an NPU device, you must pass a list of modules that contains `HashEmbeddingBagCollection`.|
|sharders|List[ModuleSharder[nn.Module]]| Optional   | Sharder list. When you use an NPU device, only the result of `get_default_hybrid_sharders()` is supported.  |
|pg|dist.ProcessGroup| Optional   | When you use an NPU device, pass `dist.GroupMember.WORLD`.                          |

**Example**

```python
from torchrec.distributed.planner import EmbeddingShardingPlanner
planner = EmbeddingShardingPlanner(XXX)
plan = planner.collective_plan(test_model, hybrid_sharders, dist.GroupMember.WORLD)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `DistributedModelParallel` (TorchRec)

### Initialization

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Converts the input `Module` into a distributed `Module` and executes the sharding plan.

**Function Prototype**

```python
class DistributedModelParallel:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                            |
|--|--|-------|----------------------------------------------------------------|
|module|nn.Module| Mandatory   | Model to parallelize. It must contain `HashEmbeddingBagCollection`.                |
|device|torch.device| Optional   | Device. When you use an NPU, set this to `torch.device("npu")`. The default value is `torch.device("cpu")`. |
|plan|ShardingPlan| Optional   | Sharding plan. Ensure that the input is the result returned by `EmbeddingShardingPlanner.collective_plan()`.|
|sharders|List[ModuleSharder[nn.Module]]| Optional   | Sharder list. When you use an NPU, only `get_default_hybrid_sharders()` is supported.           |
|env|ShardingEnv| Optional   | When `device` is `torch.device("npu")`, only the default value `None` is supported.              |
|init_data_parallel|bool| Optional   | When `device` is `torch.device("npu")`, only the default value `True` is supported.                                         |
|init_parameters|bool| Optional   | When `device` is `torch.device("npu")`, only the default value `True` is supported.                                         |
|data_parallel_wrapper|torchrec.distributed.DataParallelWrapper| Optional   | When `device` is `torch.device("npu")`, only the default value `None` is supported.                                         |

**Example**

```python
from torchrec.distributed.model_parallel import DistributedModelParallel
ddp_model = DistributedModelParallel(
    test_model, device=torch.device("npu"), plan=plan, sharders=get_default_hybrid_sharders(host_env=host_env)
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

### `fused_optimizer`

>[!NOTICE]
>
>The APIs in this class are TorchRec open-source APIs, not external Rec SDK Torch APIs. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Returns the fused optimizer for sparse tables.

**Function Prototype**

```python
def fused_optimizer()
```

**Returns**

- Success: The optimizer for sparse tables is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec.distributed.model_parallel import DistributedModelParallel
model = DistributedModelParallel(XXX)
optimizer = model.fused_optimizer
```

## `EmbCacheEmbeddingBagCollectionSharder`

**Description**

Creates an `EmbCacheEmbeddingBagCollectionSharder` sharder to shard `EmbCacheEmbeddingBagCollection` across different devices.

**Function Prototype**

```python
class EmbCacheEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|cpu_device|torch.device|Mandatory|CPU device.|
|cpu_env|ShardingEnv|Mandatory|CPU environment configuration.|
|npu_device|torch.device|Mandatory|NPU device.|
|npu_env|ShardingEnv|Mandatory|NPU environment configuration.|
|fused_params|Dict[str, Any]|Optional|Fused parameters. The default value is `None`, which is the same as `EmbeddingBagCollectionSharder` in TorchRec.|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|Optional|Quantized communication codec registry. The default value is `None`, which is the same as `EmbeddingBagCollectionSharder` in TorchRec.|

**Returns**

- Success: An `EmbCacheEmbeddingBagCollectionSharder` object is returned.
- Failure: An exception is thrown.

## `EmbCacheEmbeddingCollectionSharder`

**Description**

Initializes an `EmbCacheEmbeddingCollectionSharder` sharder to shard `EmbCacheEmbeddingCollection` across different devices.

**Function Prototype**

```python
class EmbCacheEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|cpu_device|torch.device|Mandatory|CPU device.|
|cpu_env|ShardingEnv|Mandatory|CPU environment configuration.|
|npu_device|torch.device|Mandatory|NPU device.|
|npu_env|ShardingEnv|Mandatory|NPU environment configuration.|
|fused_params|Dict[str, Any]|Optional|Fused parameters. The default value is `None`, which is the same as `EmbeddingBagCollectionSharder` in TorchRec.|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|Optional|Quantized communication codec registry. The default value is `None`, which is the same as `EmbeddingBagCollectionSharder` in TorchRec.|

**Returns**

- Success: An `EmbCacheEmbeddingCollectionSharder` object is returned.
- Failure: An exception is thrown.
