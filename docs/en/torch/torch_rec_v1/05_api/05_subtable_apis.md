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
    def __init__(
        self,
        world_size: int,
        rank: int,
        pg: dist.ProcessGroup,
        output_dtensor: bool = False,
    ) -> None:

    @classmethod
    def from_process_group(cls, pg: dist.ProcessGroup) -> "ShardingEnv":
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|world_size|int|Mandatory|Number of devices to use. The value range is [1, 8].|
|rank|int|Mandatory|Current device number. The value range is [0, `world_size` - 1].|
|pg|dist.ProcessGroup|Mandatory|Distributed communication process group. Only process groups with the `hccl` or `gloo` backend are supported. In PyTorch, the backend name of `hccl` is `custom`.|
|output_dtensor|bool|Optional|Only the default value `False` is supported.|

**Example**

```python
import os
import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv

# Simulate the initialization of a single-device torch.distributed environment. In actual use, WORLD_SIZE and RANK may be set by the PyTorch distributed training launcher
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
dist.init_process_group(backend="hccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
host_gp = dist.new_group(backend="gloo")
# Two ways to create a ShardingEnv object. Use either of them
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
host_env_new = ShardingEnv.from_process_group(host_gp)
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

## `Topology` (TorchRec)

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Stores network device topology parameters in the distributed environment.

**Function Prototype**

```python
class Topology:
    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: Optional[int] = None,
        local_world_size: Optional[int] = None,
        hbm_mem_bw: float = HBM_MEM_BW,
        ddr_mem_bw: float = DDR_MEM_BW,
        hbm_to_ddr_mem_bw: float = HBM_TO_DDR_MEM_BW,
        intra_host_bw: float = INTRA_NODE_BANDWIDTH,
        inter_host_bw: float = CROSS_NODE_BANDWIDTH,
        bwd_compute_multiplier: float = BWD_COMPUTE_MULTIPLIER,
        custom_topology_data: Optional[CustomTopologyData] = None,
        weighted_feature_bwd_compute_multiplier: float = WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER,
        uneven_sharding_perf_multiplier: float = 1.0,
        generalized_comms_bandwidths: Optional[GeneralizedCommsBandwidth] = None,
    ) -> None:
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
|custom_topology_data|torchrec.distributed.planner.types.CustomTopologyData|Optional| When you use an NPU device, only the default value `None` is supported.                            |
|weighted_feature_bwd_compute_multiplier|float|Optional| When you use an NPU device, only the default value `1` is supported.                               |
|uneven_sharding_perf_multiplier|float|Optional| When you use an NPU device, only the default value `1.0` is supported.                               |
|generalized_comms_bandwidths|torchrec.distributed.planner.types.GeneralizedCommsBandwidth|Optional| Generalized communication bandwidth configuration. When you use an NPU device, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0. |

**Example**

```python
from torchrec.distributed.planner import Topology

topology = Topology(world_size=8, compute_device="npu")
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

## `ParameterConstraints` (TorchRec)

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Specifies the query scope for the sharding plan.

**Function Prototype**

```python
@dataclass
class ParameterConstraints:
    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    min_partition: Optional[int] = None
    pooling_factors: List[float] = field(
        default_factory=lambda: [POOLING_FACTOR]
    )
    num_poolings: Optional[List[float]] = None
    batch_sizes: Optional[List[int]] = None
    is_weighted: bool = False
    cache_params: Optional[CacheParams] = None
    enforce_hbm: Optional[bool] = None
    stochastic_rounding: Optional[bool] = None
    bounds_check_mode: Optional[BoundsCheckMode] = None
    feature_names: Optional[List[str]] = None
    output_dtype: Optional[DataType] = None
    device_group: Optional[str] = None
    key_value_params: Optional[KeyValueParams] = None
    use_virtual_table: bool = False
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                                                                                                                                                                          |
|--|--|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|sharding_types|List[str]| Optional   | Sharding type. This parameter is required when you use an NPU device. The value range of strings in the list is:<ul><li>`"row_wise"`: Shards by row index. </li><li>`"data_parallel"`: Keeps one table replica on each rank. </li></ul><div class="note"><span class="notetitle">Note:</span><div class="notebody">Mixing different sharding types is not supported. The list can contain only one element, that is, one sharding type.</div></div> |
|compute_kernels|List[str]| Optional   | Computing kernel type. This parameter is required when you use an NPU device. The value range is:<ul><li>`"fused"`: Queries with a fused table. This mode is supported only when the value of `sharding_types` is `"row_wise"`. </li><li>`"dense"`: Queries with sharded tables. This mode is supported only when the value of `sharding_types` is `"data_parallel"`.</li></ul>                                      |
|min_partition|int| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|pooling_factors|List[float]| Optional   | When you use an NPU device, only the default value `[POOLING_FACTOR]` is supported.                                                                                                                                                                   |
|num_poolings|List[float]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|batch_sizes|List[int]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|is_weighted|bool| Optional   | When you use an NPU device, only the default value `False` is supported.                                                                                                                                                                            |
|cache_params|torchrec.distributed.types.CacheParams| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|enforce_hbm|bool| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|stochastic_rounding|bool| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|bounds_check_mode|torchrec.distributed.types.BoundsCheckMode| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|feature_names|List[str]| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|output_dtype|torchrec.modules.embedding_configs.DataType| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|device_group|str| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|key_value_params|torchrec.distributed.types.KeyValueParams| Optional   | When you use an NPU device, only the default value `None` is supported.                                                                                                                                                                             |
|use_virtual_table|bool| Optional   | Indicates whether to enable the virtual table for this table. When you use an NPU device, only the default value `False` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0. |

**Example**

```python
from torchrec.distributed.planner import ParameterConstraints

constraints = {
    "table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
}
```

## `get_default_hybrid_sharders`

**Description**

Retrieves the list of sharders in full-device memory mode.

**Function Prototype**

```python
def get_default_hybrid_sharders(host_env: ShardingEnv) -> List[ModuleSharder[nn.Module]]:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|host_env|ShardingEnv|Mandatory|Communication domain required for the host connection. On the host side, only the `"gloo"` backend is supported for the communication domain. For details about the implementation, see the example of this API.|

**Returns**

Success: A list of supported sharders is returned.

Failure: An exception is thrown.

**Example**

```python
import os

from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from torchrec.distributed.types import ShardingEnv
import torch.distributed as dist

# Simulate the initialization of a single-device torch.distributed environment. In actual use, WORLD_SIZE and RANK may be set by the PyTorch distributed training launcher
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
dist.init_process_group(backend="hccl")
host_gp = dist.new_group(backend="gloo")
world_size = dist.get_world_size()
rank = dist.get_rank()
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
hybrid_sharders = get_default_hybrid_sharders(host_env=host_env)
```

## `EmbCacheEmbeddingBagCollectionSharder`

**Description**

Creates a sharder for the `EmbCacheEmbeddingBagCollection` class in multi-level cache mode, to shard `EmbCacheEmbeddingBagCollection` across different devices.

**Function Prototype**

```python
class EmbCacheEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        cpu_device: torch.device,
        cpu_env: ShardingEnv,
        npu_device: torch.device,
        npu_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|cpu_device|torch.device|Mandatory|CPU device.|
|cpu_env|ShardingEnv|Mandatory|CPU environment configuration.|
|npu_device|torch.device|Mandatory|NPU device.|
|npu_env|ShardingEnv|Mandatory|NPU environment configuration.|
|fused_params|Dict[str, Any]|Optional|Fused parameters. The default value is `None`.|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|Optional|Quantized communication codec registry. The default value is `None`.|

**Returns**

- Success: An `EmbCacheEmbeddingBagCollectionSharder` object is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
from torch import distributed as dist
from torchrec.distributed.types import ShardingEnv
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingBagCollectionSharder

cpu_device = torch.device("cpu")
device = torch.device("npu")
cpu_pg = dist.new_group(backend="gloo")
cpu_env = ShardingEnv.from_process_group(cpu_pg)
embcache_sharder = EmbCacheEmbeddingBagCollectionSharder(
    cpu_device=cpu_device,
    cpu_env=cpu_env,
    npu_device=device,
    npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
)
sharders = [embcache_sharder]
```

## `EmbCacheEmbeddingCollectionSharder`

**Description**

Creates a sharder for the `EmbCacheEmbeddingCollection` class in multi-level cache mode, to shard `EmbCacheEmbeddingCollection` across different devices.

**Function Prototype**

```python
class EmbCacheEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        cpu_device: torch.device,
        cpu_env: ShardingEnv,
        npu_device: torch.device,
        npu_env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|cpu_device|torch.device|Mandatory|CPU device.|
|cpu_env|ShardingEnv|Mandatory|CPU environment configuration.|
|npu_device|torch.device|Mandatory|NPU device.|
|npu_env|ShardingEnv|Mandatory|NPU environment configuration.|
|fused_params|Dict[str, Any]|Optional|Fused parameters. The default value is `None`.|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|Optional|Quantized communication codec registry. The default value is `None`.|

**Returns**

- Success: An `EmbCacheEmbeddingCollectionSharder` object is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
from torch import distributed as dist
from torchrec.distributed.types import ShardingEnv
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder

cpu_device = torch.device("cpu")
device = torch.device("npu")
cpu_pg = dist.new_group(backend="gloo")
cpu_env = ShardingEnv.from_process_group(cpu_pg)
embcache_sharder = EmbCacheEmbeddingCollectionSharder(
    cpu_device=cpu_device,
    cpu_env=cpu_env,
    npu_device=device,
    npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
)
sharders = [embcache_sharder]
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
class EmbeddingShardingPlanner(ShardingPlanner):
    def __init__(
        self,
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
        timeout_seconds: Optional[int] = None,
        plan_loader: Optional[PlanLoader] = None,
    ) -> None:
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                    |
|--|--|-------|-------------------------------------------------------|
|topology|Topology| Optional   | See the value range of [Topology (TorchRec)](#topology-torchrec). When you use an NPU device, this parameter is required.                            |
|batch_size|int| Optional   | The value range is [1, 1000000].                                    |
|enumerator|torchrec.distributed.planner.types.Enumerator| Optional   | When you use an NPU device, only the default value `None` is supported.                                 |
|storage_reservation|torchrec.distributed.planner.types.StorageReservation| Optional   | When you use an NPU device, only the default value `None` is supported.                                 |
|proposer|torchrec.distributed.planner.types.Proposer| Optional   | When you use an NPU device, only the default value `None` is supported.                                 |
|partitioner|torchrec.distributed.planner.types.Partitioner| Optional   | When you use an NPU device, only the default value `None` is supported.                                 |
|performance_model|torchrec.distributed.planner.types.PerfModel| Optional   | When you use an NPU device, only the default value `None` is supported.                                 |
|stats|Union[Stats, List[Stats]]| Optional   | When you use an NPU device, only the default value `None` is supported. The `Stats` type is `torchrec.distributed.planner.types.Stats`.                                 |
|constraints|Dict[str, ParameterConstraints]| Optional   | See the value range of [ParameterConstraints (TorchRec)](#parameterconstraints-torchrec). When you use an NPU device, this parameter is required. |
|debug|bool| Optional   | When you use an NPU device, only the default value `True` is supported.                                 |
|callbacks|List[Callable]| Optional   | When you use an NPU device, only the default value `None` is supported.                                 |
|timeout_seconds|int| Optional   | Timeout duration (in seconds). When you use an NPU device, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.2.0 and v1.5.0. |
|plan_loader|torchrec.distributed.planner.types.PlanLoader| Optional   | Sharding plan loader. When you use an NPU device, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0. |

**Example**

```python
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints, Topology

topology = Topology(world_size=8, compute_device="npu")
constraints = {"table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])}
planner = EmbeddingShardingPlanner(
    topology=topology,
    constraints=constraints,
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

### `collective_plan`

>[!NOTICE]
>
>The APIs in this class are TorchRec open-source APIs, not external Rec SDK Torch APIs. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Searches for the most suitable sharding plan.

**Function Prototype**

```python
def collective_plan(
    self,
    module: nn.Module,
    sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> ShardingPlan:
```

**Parameters**

| Parameter | Type                               | Mandatory/Optional | Description                                                                                                                                                                                                                                                                                            |
| -------- | ------------------------------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| module   | torch.nn.Module                      | Optional  | When you use an NPU device, you must pass a model containing a [HashEmbeddingBagCollection](02_table_creation_apis.md#hashembeddingbagcollection)/[EmbCacheEmbeddingCollection](02_table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](02_table_creation_apis.md#embcacheembeddingbagcollection) instance. |
| sharders | List[ModuleSharder[torch.nn.Module]] | Optional  | Sharder list. When you use an NPU, only the return value of `get_default_hybrid_sharders()`, `List[EmbCacheEmbeddingCollectionSharder]`, or `List[EmbCacheEmbeddingBagCollectionSharder]` is supported.<br>Sharder types correspond one-to-one to the Collection types in the module.                                                                                       |
| pg       | torch.distributed.ProcessGroup       | Optional  | When you use an NPU device, pass `dist.GroupMember.WORLD`.                                                                                                                                                                                                                                                      |

**Example**

Note: For the detailed definition of `test_model` in the code example, see how `test_model` is created in [DistributedModelParallel Example](#section_distributed_model_parallel_code_sample).

```python
import os

import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints, Topology
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders

# Simulate the initialization of a single-device torch.distributed environment. In actual use, WORLD_SIZE and RANK may be set by the PyTorch distributed training launcher
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
dist.init_process_group(backend="hccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
topology = Topology(world_size=world_size, compute_device="npu")
constraints = {"table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])}
planner = EmbeddingShardingPlanner(
    topology=topology,
    constraints=constraints,
)
# test_model is a torch.nn.Module object that contains both sparse and dense models. Its detailed definition is omitted here
test_model = ......
host_gp = dist.new_group(backend="gloo")
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
sharders = get_default_hybrid_sharders(host_env=host_env)
plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

## `DistributedModelParallel` (TorchRec)<a id="TOPIC_0000002338384297"></a>

### Initialization

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Converts the input `Module` into a distributed `Module` and executes the sharding plan.

**Function Prototype**

```python
class DistributedModelParallel(nn.Module, FusedOptimizerModule):
    def __init__(
        self,
        module: nn.Module,
        env: Optional[ShardingEnv] = None,
        device: Optional[torch.device] = None,
        plan: Optional[ShardingPlan] = None,
        sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
        init_data_parallel: bool = True,
        init_parameters: bool = True,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
        model_tracker_configs: Optional[ModelTrackerConfigs] = None,
    ) -> None:
```

**Parameters**

| Parameter            | Type                                   | Mandatory/Optional | Description                                                                                                                                                                                                                                                                                           |
| --------------------- | ---------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| module                | nn.Module                                | Mandatory | Model for distributed parallel training. You must pass a model containing a [HashEmbeddingBagCollection](02_table_creation_apis.md#hashembeddingbagcollection)/[EmbCacheEmbeddingCollection](02_table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](02_table_creation_apis.md#embcacheembeddingbagcollection) instance. |
| env                   | ShardingEnv                              | Optional  | When `device` is `torch.device("npu")`, only the default value `None` is supported.                                                                                                                                                                                                             |
| device                | torch.device                             | Optional  | Device. When you use an NPU, set this to `torch.device("npu")`. The default value is `torch.device("cpu")`.                                                                                                                                                                                                                  |
| plan                  | ShardingPlan                             | Optional  | Sharding plan. Ensure that the input is the result returned by `EmbeddingShardingPlanner.collective_plan()`.                                                                                                                                                                                                             |
| sharders              | List[ModuleSharder[torch.nn.Module]]     | Optional  | Sharder list. When you use an NPU, only the following three sharder types are supported: the return value of `get_default_hybrid_sharders()`, `List[EmbCacheEmbeddingCollectionSharder]`, and `List[EmbCacheEmbeddingBagCollectionSharder]`. They must be paired with the three Collection types in the `module` parameter. |
| init_data_parallel    | bool                                     | Optional  | When `device` is `torch.device("npu")`, only the default value `True` is supported.                                                                                                                                                                                                             |
| init_parameters       | bool                                     | Optional  | When `device` is `torch.device("npu")`, only the default value `True` is supported.                                                                                                                                                                                                             |
| data_parallel_wrapper | torchrec.distributed.DataParallelWrapper | Optional  | When `device` is `torch.device("npu")`, only the default value `None` is supported.                                                                                                                                                                                                             |
| model_tracker_configs | torchrec.distributed.model_parallel.ModelTrackerConfigs | Optional  | Model tracker configuration. When you use an NPU device, only the default value `None` is supported. You cannot customize this parameter. This parameter is supported only in TorchRec v1.5.0. |

**Example<a id="section_distributed_model_parallel_code_sample"></a>**

Note: Because using `DataParallelWrapper` requires creating many preceding parameters, this example provides the complete code for ease of use and understanding.

```python
import os

import torch
import torch_npu
import torchrec
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)

from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from hybrid_torchrec.modules.hash_embeddingbag import HashEmbeddingBagCollection, HashEmbeddingBagConfig


class DenseModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TestModel(torch.nn.Module):
    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        sparse_output = self.sparse_model(batch.sparse_features)
        sparse_output_dict: dict[str, torchrec.JaggedTensor] = sparse_output.wait()  # type: ignore
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output_dict[feat_name].values())
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)
        dense_output: torch.Tensor = self.dense_model(embeddings)
        loss = self.loss_fn(dense_output, batch.labels)
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # forward must return loss and output, with loss first and output second
        return loss, output


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


# Simulate the initialization of a single-device torch.distributed environment. In actual use, WORLD_SIZE and RANK may be set by the PyTorch distributed training launcher
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
rank = int(os.environ.get("LOCAL_RANK", 0))
torch_npu.npu.set_device(rank)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
dist.init_process_group(backend="hccl")

DENSE_OUTPUT_DIM: int = 2
embedding_dims: list[int] = [64, 64, 64]
num_embeddings: list[int] = [400, 4000, 400]
table_num: int = len(num_embeddings)
world_size = dist.get_world_size()
npu_device = torch.device("npu")

# Create sparse tables
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

# Create an integrated model: wrap the sparse tables and the dense part into one model
dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), DENSE_OUTPUT_DIM)
test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

# Define the optimizer for sparse tables
embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.01, "eps": 1e-5}
apply_optimizer_in_backward(
    embedding_optimizer_class,
    test_model.sparse_model.parameters(),
    optimizer_kwargs=optimizer_kwargs,
)

# Shard sparse tables
host_gp = dist.new_group(backend="gloo")
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
sharders = get_default_hybrid_sharders(host_env=host_env)
constraints = {
    table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
    for table_name in table_names
}
planner = EmbeddingShardingPlanner(
    topology=Topology(world_size=world_size, compute_device="npu"),
    constraints=constraints,
)
plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)
ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

### `fused_optimizer`

>[!NOTICE]
>
>The APIs in this class are TorchRec open-source APIs, not external Rec SDK Torch APIs. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Returns the fused optimizer for sparse tables.

**Function Prototype**

```python
@property
def fused_optimizer(self) -> CombinedOptimizer:
```

**Returns**

- Success: The optimizer for sparse tables is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec.distributed.model_parallel import DistributedModelParallel

# ddp_model is a model object created by DistributedModelParallel (TorchRec). Its detailed definition is omitted here
ddp_model = ......
optimizer = ddp_model.fused_optimizer
```
