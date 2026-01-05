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


## ParameterConstraints（TorchRec）<a name="ZH-CN_TOPIC_0000002336148869"></a>

>[!NOTICE] 须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

指定分表计划的查询范围。

**函数原型<a name="section1483104721911"></a>**

```cpp
class ParameterConstraints:
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                                                                                                                                                                 |
|--|--|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|sharding_type|List[str]| 可选    | 分表的类型。当使用NPU设备时为必选参数，取值范围：<li>"row_wise"：按照行号进行分表。</li><li>"data_parallel"：每个rank保留一个表副本。</li><div class="note"><span class="notetitle">说明</span><div class="notebody">不支持混合使用不同的分表类型。</div></div> |
|compute_kernels|List[str]| 可选    | 计算的kernel类型。当使用NPU设备时为必选参数，取值范围：<li>"fused"：采用合表的方式查询。该方式仅在sharding_type为"row_wise"时使用。</li><li>"dense"：采用分表的方式查询。该方式仅在sharding_type为"data_parallel"时使用。</li>                                      |
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
from torchrec.distributed.planner import ParameterConstraints
constraints = {
   "table0": ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
}
```


## get\_default\_hybrid\_sharders<a name="ZH-CN_TOPIC_0000002338277269"></a>

**功能描述<a name="section634582619155"></a>**

获取分表器。

**函数原型<a name="section1483104721911"></a>**

```cpp
def get_default_hybrid_sharders(host_env: ShardingEnv) -> List[ModuleSharder[nn.Module]]:
```

**参数说明<a name="section182631461211"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|host_env|ShardingEnv|必选|传入host连接需要的通讯域。参考[步骤3](../quick_start.md#接口调用介绍)的创建方法。仅支持backend为"gloo"。|


**返回值<a name="section06646162266"></a>**

成功：返回默认的分表器。

失败：抛出异常

**使用示例<a name="section106984023511"></a>**

```cpp
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from torchrec.distributed.types import ShardingEnv
import torch.distributed as dist

host_gp = dist.new_group(backend="gloo")
world_size = dist.get_world_size()
rank = dist.get_rank()
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)
```


## EmbeddingShardingPlanner（TorchRec）<a name="ZH-CN_TOPIC_0000002304198202"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002524309357"></a>

>[!NOTICE] 须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

创建分表计划器，用于搜索最合适的分表计划。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbeddingShardingPlanner:
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                    |
|--|--|-------|-------------------------------------------------------|
|topology|Topology| 可选    | 参考Topology（TorchRec）的取值范围。当使用NPU设备时参数为必传。                            |
|constraints|Dict[str, ParameterConstraints]| 可选    | 参考ParameterConstraints（TorchRec）的取值范围。当使用NPU设备时参数为必传。 |
|batch_size|int| 可选    | 取值范围：[1, 1000000]。                                    |
|enumerator|torchrec.distributed.planner.types.Enumerator| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|storage_reservation|torchrec.distributed.planner.types.StorageReservation| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|proposer|torchrec.distributed.planner.types.Proposer| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|partitioner|torchrec.distributed.planner.types.Partitioner| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|performance_model|torchrec.distributed.planner.types.PerfModel| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|stats|torchrec.distributed.planner.types.Stats| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |
|debug|bool| 可选    | 当使用NPU设备时仅支持默认值为True，不支持用户自定义。                                 |
|callbacks|List[Callable]| 可选    | 当使用NPU设备时仅支持默认值为None，不支持用户自定义。                                 |


**使用示例<a name="section193151694205"></a>**

```cpp
from torchrec.distributed.planner import EmbeddingShardingPlanner
planner = EmbeddingShardingPlanner(
 topology=topology,
 constraints=constraints,
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


### collective\_plan<a name="ZH-CN_TOPIC_0000002508694909"></a>

>[!NOTICE] 须知 
>此类下的接口为TorchRec开源接口，非Rec SDK Torch对外接口。本章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

搜索最合适的分表计划。

**函数原型<a name="section1483104721911"></a>**

```cpp
def collective_plan(
 module: nn.Module,
 sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
 pg: Optional[dist.ProcessGroup] = None,
)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                  |
|--|--|-------|-----------------------------------------------------|
|module|nn.Module| 可选    | 当使用NPU设备时必须传入包含HashEmbeddingBagCollection的module列表。 |
|sharders|List[ModuleSharder[nn.Module]]| 可选    | Sharder的列表。当使用NPU设备时仅支持传入get_default_hybrid_sharders()的结果。   |
|pg|dist.ProcessGroup| 可选    | 当使用NPU设备时传入dist.GroupMember.WORLD。                           |


**使用示例<a name="section193151694205"></a>**

```cpp
from torchrec.distributed.planner import EmbeddingShardingPlanner
planner = EmbeddingShardingPlanner(XXX)
plan = planner.collective_plan(test_model, hybrid_sharders, dist.GroupMember.WORLD)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。



## DistributedModelParallel（TorchRec）<a name="ZH-CN_TOPIC_0000002338384297"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002492189666"></a>

>[!NOTICE] 须知  
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

将传入的Module变为分布式的Module，并执行分表计划。

**函数原型<a name="section1483104721911"></a>**

```cpp
class DistributedModelParallel:
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型| 可选/必选 | 说明                                                             |
|--|--|-------|----------------------------------------------------------------|
|module|nn.Module| 必选    | 需要并行的模型。包含HashEmbeddingBagCollection的module列表。                 |
|device|torch.device| 可选    | 设备。使用NPU时取值为torch.device("npu")，即npu设备，默认为torch.device("cpu")  |
|plan|ShardingPlan| 可选    | 分表计划。用户需保证传入的必须是EmbeddingShardingPlanner.collective_plan返回的结果。 |
|sharders|List[ModuleSharder[nn.Module]]| 可选    | Sharder的列表。使用NPU时仅支持传入get_default_hybrid_sharders()            |
|env|ShardingEnv| 可选    | device为torch.device("npu")时仅支持默认值为None，不支持用户自定义。               |
|init_data_parallel|bool| 可选    | device为torch.device("npu")时仅支持默认值为True，不支持用户自定义。                                          |
|init_parameters|bool| 可选    | device为torch.device("npu")时仅支持默认值为True，不支持用户自定义。                                          |
|data_parallel_wrapper|torchrec.distributed.DataParallelWrapper| 可选    | device为torch.device("npu")时仅支持默认值为None，不支持用户自定义。                                          |


**使用示例<a name="section193151694205"></a>**

```cpp
from torchrec.distributed.model_parallel import DistributedModelParallel
ddp_model = DistributedModelParallel(
 test_model, device=torch.device("npu"), plan=plan, sharders=get_default_hybrid_sharders(host_env=host_env)
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


### fused\_optimizer<a name="ZH-CN_TOPIC_0000002476574952"></a>

>[!NOTICE] 须知 
>此类下的接口为TorchRec开源接口，非Rec SDK Torch对外接口。本章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

返回稀疏表的融合优化器。

**函数原型<a name="section1483104721911"></a>**

```cpp
def fused_optimizer()
```

**返回值说明<a name="section1367815197580"></a>**

-   成功：返回系数表的优化器。
-   失败：抛出异常。

**使用示例<a name="section1045492782314"></a>**

```cpp
from torchrec.distributed.model_parallel import DistributedModelParallel
model = DistributedModelParallel(XXX)
optimizer = model.fused_optimizer
```



## EmbCacheEmbeddingBagCollectionSharder<a name="ZH-CN_TOPIC_0000002396403112"></a>

**功能描述<a name="section634582619155"></a>**

创建EmbCacheEmbeddingBagCollectionSharder分表器，用于将EmbCacheEmbeddingBagCollection分片到不同的设备上。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|cpu_device|torch.device|必选|CPU设备|
|cpu_env|ShardingEnv|必选|CPU环境配置|
|npu_device|torch.device|必选|NPU设备|
|npu_env|ShardingEnv|必选|NPU环境配置|
|fused_params|Dict[str, Any]|可选|融合参数，默认值None, 和torchrec的EmbeddingBagCollectionSharder一致|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|可选|量化通信编解码器注册表，默认值None, 和torchrec的EmbeddingBagCollectionSharder一致|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheEmbeddingBagCollectionSharder对象。
-   失败：抛出异常。


## EmbCacheEmbeddingCollectionSharder<a name="ZH-CN_TOPIC_0000002396403120"></a>

**功能描述<a name="section634582619155"></a>**

初始化EmbCacheEmbeddingCollectionSharder分表器，用于将EmbCacheEmbeddingCollection分片到不同的设备上。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheEmbeddingCollectionSharder(EmbeddingCollectionSharder):
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|cpu_device|torch.device|必选|CPU设备|
|cpu_env|ShardingEnv|必选|CPU环境配置|
|npu_device|torch.device|必选|NPU设备|
|npu_env|ShardingEnv|必选|NPU环境配置|
|fused_params|Dict[str, Any]|可选|融合参数，默认值None, 和torchrec的EmbeddingBagCollectionSharder一致|
|qcomm_codecs_registry|Dict[str, QuantizedCommCodecs]|可选|量化通信编解码器注册表，默认值None, 和torchrec的EmbeddingBagCollectionSharder一致|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheEmbeddingCollectionSharder对象。
-   失败：抛出异常。


