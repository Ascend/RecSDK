# 分表接口<a name="ZH-CN_TOPIC_0000002336268665"></a>

## ShardingEnv（TorchRec）<a name="ZH-CN_TOPIC_0000002336148941"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

保存分布式相关参数。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|world_size|int|必选|使用的卡数。取值范围：[1，8]|
|rank|int|必选|当前的卡号。取值范围：[0，world_size -1]|
|pg|dist.ProcessGroup|必选|分布式通讯链接。取值范围：只支持backend为hccl和gloo的链接。hccl在PyTorch里面的backend_name为custom。|
|output_dtensor|bool|可选|仅支持默认值为False，不支持用户自定义。|

**使用示例**

```python
import os
import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv

# 模拟初始化单卡torch.distributed环境。实际使用时WORLD_SIZE和RANK可能由PyTorch分布式训练启动器设置。
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
dist.init_process_group(backend="hccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
host_gp = dist.new_group(backend="gloo")
# 两种创建ShardingEnv对象方式，使用其中一种即可。
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
host_env_new = ShardingEnv.from_process_group(host_gp)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## Topology（TorchRec）<a name="ZH-CN_TOPIC_0000002336268737"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

保存分布式环境网络设备拓扑参数。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

|参数名|类型|可选/必选| 说明                                                |
|--|--|--|---------------------------------------------------|
|world_size|int|必选| 使用的卡数。取值范围：[1，8]                                  |
|compute_device|str|必选| 设备名称。当使用NPU设备时取值为"npu"，即npu设备。                    |
|hbm_cap|int|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|ddr_cap|int|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|local_world_size|int|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|hbm_mem_bw|float|可选| 当使用NPU设备时仅支持默认值为(897 \* 1024 \* 1024 \* 1024 / 1000)，不支持用户自定义。 |
|ddr_mem_bw|float|可选| 当使用NPU设备时仅支持默认值为(51 \* 1024 \* 1024 \* 1024 / 1000)，不支持用户自定义。 |
|hbm_to_ddr_mem_bw|float|可选| 当使用NPU设备时仅支持默认值为(32 \* 1024 \* 1024 \* 1024 / 1000)，不支持用户自定义。 |
|intra_host_bw|float|可选| 当使用NPU设备时仅支持默认值为(600 \* 1024 \* 1024 \* 1024 / 1000)，不支持用户自定义。 |
|inter_host_bw|float|可选| 当使用NPU设备时仅支持默认值为(12.5 \* 1024 \* 1024 \* 1024 / 1000)，不支持用户自定义。 |
|bwd_compute_multiplier|float|可选| 当使用NPU设备时仅支持默认值为2，不支持用户自定义。                                |
|custom_topology_data|torchrec.distributed.planner.types.CustomTopologyData|可选| 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                             |
|weighted_feature_bwd_compute_multiplier|float|可选| 当使用NPU设备时仅支持默认值为1，不支持用户自定义。                                |
|uneven_sharding_perf_multiplier|float|可选| 当使用NPU设备时仅支持默认值为1.0，不支持用户自定义。                                |
|generalized_comms_bandwidths|torchrec.distributed.planner.types.GeneralizedCommsBandwidth|可选| 通用通信带宽配置。当使用NPU设备时仅支持默认值为None，不支持用户自定义。仅支持TorchRec v1.5.0版本。 |

**使用示例**

```python
from torchrec.distributed.planner import Topology

topology = Topology(world_size=8, compute_device="npu")
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## ParameterConstraints（TorchRec）<a name="ZH-CN_TOPIC_0000002336148869"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

指定分表计划的查询范围。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

|参数名|类型| 可选/必选 | 说明                                                                                                                                                                                                          |
|--|--|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|sharding_types|List[str]| 可选    | 分表的类型。当使用NPU设备时为必选参数，列表中字符串取值范围：<ul><li>"row_wise"：按照行号进行分表。</li><li>"data_parallel"：每个rank保留一个表副本。</li></ul><div class="note"><span class="notetitle">说明：</span><div class="notebody">不支持混合使用不同的分表类型，列表中只能有一个元素，即一种分表类型。</div></div> |
|compute_kernels|List[str]| 可选    | 计算的kernel类型。当使用NPU设备时为必选参数，取值范围：<ul><li>"fused"：采用合表的方式查询。该方式仅支持sharding_types中的值为"row_wise"时使用。</li><li>"dense"：采用分表的方式查询。该方式仅支持sharding_types中的值为"data_parallel"时使用。</li></ul>                                      |
|min_partition|int| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|pooling_factors|List[float]| 可选    | 当使用NPU设备时仅支持默认值为\[POOLING_FACTOR\]，不支持用户自定义。                                                                                                                                                                    |
|num_poolings|List[float]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|batch_sizes|List[int]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|is_weighted|bool| 可选    | 当使用NPU设备时仅支持默认值为False，不支持用户自定义。                                                                                                                                                                             |
|cache_params|torchrec.distributed.types.CacheParams| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|enforce_hbm|bool| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|stochastic_rounding|bool| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|bounds_check_mode|torchrec.distributed.types.BoundsCheckMode| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|feature_names|List[str]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|output_dtype|torchrec.modules.embedding_configs.DataType| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|device_group|str| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|key_value_params|torchrec.distributed.types.KeyValueParams| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                              |
|use_virtual_table|bool| 可选    | 是否对该表启用虚拟表。当使用NPU设备时仅支持默认值为False，不支持用户自定义。仅支持TorchRec v1.5.0版本。                                                                                                                                                    |

**使用示例**

```python
from torchrec.distributed.planner import ParameterConstraints

constraints = {
    "table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
}
```

## get\_default\_hybrid\_sharders<a name="ZH-CN_TOPIC_0000002338277269"></a>

**功能描述<a name="section634582619155"></a>**

获取纯显存模式下的分表器列表。

**函数原型<a name="section1483104721911"></a>**

```python
def get_default_hybrid_sharders(host_env: ShardingEnv) -> List[ModuleSharder[nn.Module]]:
```

**参数说明<a name="section182631461211"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|host_env|ShardingEnv|必选|传入host连接需要的通讯域。host侧通讯域仅支持backend为"gloo"，详细实现请参考当前API使用示例。|

**返回值<a name="section06646162266"></a>**

成功：返回支持的分表器列表。

失败：抛出异常

**使用示例<a name="section106984023511"></a>**

```python
import os

from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from torchrec.distributed.types import ShardingEnv
import torch.distributed as dist

# 模拟初始化单卡torch.distributed环境。实际使用时WORLD_SIZE和RANK可能由PyTorch分布式训练启动器设置。
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

## EmbCacheEmbeddingBagCollectionSharder<a name="ZH-CN_TOPIC_0000002396403112"></a>

**功能描述<a name="section634582619155"></a>**

创建多级缓存模式的EmbCacheEmbeddingBagCollection类的分表器，用于将EmbCacheEmbeddingBagCollection分片到不同的设备上。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|cpu_device|torch.device|必选|CPU设备。|
|cpu_env|ShardingEnv|必选|CPU环境配置。|
|npu_device|torch.device|必选|NPU设备。|
|npu_env|ShardingEnv|必选|NPU环境配置。|
|fused_params|Dict[str, Any]|可选|融合参数，默认值None。|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|可选|量化通信编解码器注册表，默认值None。|

**返回值说明**

- 成功：返回EmbCacheEmbeddingBagCollectionSharder对象。
- 失败：抛出异常。

**使用示例**

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

## EmbCacheEmbeddingCollectionSharder<a name="ZH-CN_TOPIC_0000002396403120"></a>

**功能描述<a name="section634582619155"></a>**

创建多级缓存模式的EmbCacheEmbeddingCollection类的分表器，用于将EmbCacheEmbeddingCollection分片到不同的设备上。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|cpu_device|torch.device|必选|CPU设备。|
|cpu_env|ShardingEnv|必选|CPU环境配置。|
|npu_device|torch.device|必选|NPU设备。|
|npu_env|ShardingEnv|必选|NPU环境配置。|
|fused_params|Dict[str, Any]|可选|融合参数，默认值None。|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|可选|量化通信编解码器注册表，默认值None。|

**返回值说明**

- 成功：返回EmbCacheEmbeddingCollectionSharder对象。
- 失败：抛出异常。

**使用示例**

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

## EmbeddingShardingPlanner（TorchRec）<a name="ZH-CN_TOPIC_0000002304198202"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002524309357"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

创建分表计划器，用于搜索最合适的分表计划。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

|参数名|类型| 可选/必选 | 说明                                                    |
|--|--|-------|-------------------------------------------------------|
|topology|Topology| 可选    | 参考[Topology（TorchRec）](#topologytorchrec)的取值范围。当使用NPU设备时参数为必选。                            |
|batch_size|int| 可选    | 取值范围：[1, 1000000]。                                    |
|enumerator|torchrec.distributed.planner.types.Enumerator| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|storage_reservation|torchrec.distributed.planner.types.StorageReservation| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|proposer|torchrec.distributed.planner.types.Proposer| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|partitioner|torchrec.distributed.planner.types.Partitioner| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|performance_model|torchrec.distributed.planner.types.PerfModel| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|stats|Union[Stats, List[Stats]]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。Stats类型为torchrec.distributed.planner.types.Stats。                                 |
|constraints|Dict[str, ParameterConstraints]| 可选    | 参考[ParameterConstraints（TorchRec）](#parameterconstraintstorchrec)的取值范围。当使用NPU设备时参数为必选。 |
|debug|bool| 可选    | 当使用NPU设备时仅支持默认值为True，不支持用户自定义。                                 |
|callbacks|List[Callable]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|timeout_seconds|int| 可选    | 超时时间（秒）。当使用NPU设备时仅支持默认值为None，不支持用户自定义。仅支持TorchRec v1.2.0和v1.5.0版本。 |
|plan_loader|torchrec.distributed.planner.types.PlanLoader| 可选    | 分表计划加载器。当使用NPU设备时仅支持默认值为None，不支持用户自定义。仅支持TorchRec v1.5.0版本。 |

**使用示例**

```python
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints, Topology

topology = Topology(world_size=8, compute_device="npu")
constraints = {"table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])}
planner = EmbeddingShardingPlanner(
    topology=topology,
    constraints=constraints,
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

### collective\_plan<a name="ZH-CN_TOPIC_0000002508694909"></a>

>[!NOTICE]
>
>此类下的接口为TorchRec开源接口，非Rec SDK Torch对外接口。本章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

搜索最合适的分表计划。

**函数原型<a name="section1483104721911"></a>**

```python
def collective_plan(
    self,
    module: nn.Module,
    sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> ShardingPlan:
```

**参数说明**

|参数名|类型| 可选/必选 | 说明                                                  |
|--|--|-------|-----------------------------------------------------|
|module|torch.nn.Module| 可选    | 当使用NPU设备时必须传入包含HashEmbeddingBagCollection的module列表。 |
|sharders|List[ModuleSharder[torch.nn.Module]]| 可选    | Sharder的列表。使用NPU时仅支持传入get_default_hybrid_sharders()返回值/List[EmbCacheEmbeddingCollectionSharder]/List[EmbCacheEmbeddingBagCollectionSharder]。   |
|pg|torch.distributed.ProcessGroup| 可选    | 当使用NPU设备时传入dist.GroupMember.WORLD。                           |

**使用示例**

注：代码示例中test_model的详细定义可参考[DistributedModelParallel使用示例](#section_distributed_model_parallel_code_sample)中test_model创建实现。

```python
import os

import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv
from torchrec.distributed.planner import EmbeddingShardingPlanner, ParameterConstraints, Topology
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders

# 模拟初始化单卡torch.distributed环境。实际使用时WORLD_SIZE和RANK可能由PyTorch分布式训练启动器设置。
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
# test_model为包含稀疏模型和稠密模型的torch.nn.Module对象，此处省略详细定义。
test_model = ......
host_gp = dist.new_group(backend="gloo")
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
sharders = get_default_hybrid_sharders(host_env=host_env)
plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## DistributedModelParallel（TorchRec）<a id="TOPIC_0000002338384297"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002492189666"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

将传入的Module变为分布式的Module，并执行分表计划。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明**

| 参数名                | 类型                                     | 可选/必选 | 说明                                                                                                                                                                                                                                                                                            |
| --------------------- | ---------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| module                | nn.Module                                | 必选      | 需要并行的模型。包含[HashEmbeddingBagCollection](table_creation_apis.md#hashembeddingbagcollection)/[EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](./table_creation_apis.md#embcacheembeddingbagcollection)的module列表。 |
| env                   | ShardingEnv                              | 可选      | device为torch.device("npu")时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                                                                             |
| device                | torch.device                             | 可选      | 设备。使用NPU时取值为torch.device("npu")，即npu设备，默认为torch.device("cpu")                                                                                                                                                                                                                  |
| plan                  | ShardingPlan                             | 可选      | 分表计划。用户需保证传入的必须是EmbeddingShardingPlanner.collective_plan返回的结果。                                                                                                                                                                                                            |
| sharders              | List[ModuleSharder[torch.nn.Module]]     | 可选      | Sharder列表。使用NPU时仅支持传入get_default_hybrid_sharders()返回值/List[EmbCacheEmbeddingCollectionSharder]/List[EmbCacheEmbeddingBagCollectionSharder]等三种sharders类型，且需与module参数中的三种Collection类型配对使用。                                                                    |
| init_data_parallel    | bool                                     | 可选      | device为torch.device("npu")时仅支持默认值为True，不支持用户自定义。                                                                                                                                                                                                                             |
| init_parameters       | bool                                     | 可选      | device为torch.device("npu")时仅支持默认值为True，不支持用户自定义。                                                                                                                                                                                                                             |
| data_parallel_wrapper | torchrec.distributed.DataParallelWrapper | 可选      | device为torch.device("npu")时仅支持默认值为None，不支持用户自定义。                                                                                                                                                                                                                             |
| model_tracker_configs | torchrec.distributed.model_parallel.ModelTrackerConfigs | 可选      | 模型追踪器配置。当使用NPU设备时仅支持默认值为None，不支持用户自定义。仅支持TorchRec v1.5.0版本。                                                                                                                                                                                                |

**使用示例<a id="section_distributed_model_parallel_code_sample"></a>**

注：由于DataParallelWrapper使用时需创建较多的前置参数，为便于使用和理解，当前示例中提供完整代码。

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
        # forward必须返回loss和output，且loss在前，output在后
        return loss, output


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


# 模拟初始化单卡torch.distributed环境。实际使用时WORLD_SIZE和RANK可能由PyTorch分布式训练启动器设置。
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

# 创建稀疏表
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

# 创建整合模型：将稀疏表和Dense部分包装到一个模型中
dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), DENSE_OUTPUT_DIM)
test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

# 定义稀疏表优化器
embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.01, "eps": 1e-5}
apply_optimizer_in_backward(
    embedding_optimizer_class,
    test_model.sparse_model.parameters(),
    optimizer_kwargs=optimizer_kwargs,
)

# 稀疏表分表
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

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

### fused\_optimizer<a name="ZH-CN_TOPIC_0000002476574952"></a>

>[!NOTICE]
>
>此类下的接口为TorchRec开源接口，非Rec SDK Torch对外接口。本章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

返回稀疏表的融合优化器。

**函数原型<a name="section1483104721911"></a>**

```python
@property
def fused_optimizer(self) -> CombinedOptimizer:
```

**返回值说明<a name="section1367815197580"></a>**

- 成功：返回稀疏表的优化器。
- 失败：抛出异常。

**使用示例<a name="section1045492782314"></a>**

```python
from torchrec.distributed.model_parallel import DistributedModelParallel

# ddp_model为DistributedModelParallel（TorchRec）创建的模型对象，此处省略详细定义。
ddp_model = ......
optimizer = ddp_model.fused_optimizer
```
