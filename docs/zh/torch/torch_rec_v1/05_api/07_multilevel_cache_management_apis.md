# 多级缓存管理接口<a name="ZH-CN_TOPIC_0000002430082789"></a>

## InitializerType<a name="ZH-CN_TOPIC_0000002430202769"></a>

**功能描述<a name="section634582619155"></a>**

定义多级缓存模式下稀疏表权重的初始化方式的枚举类。

**函数原型<a name="section1483104721911"></a>**

```python
class InitializerType(Enum):
    LINEAR = "linear"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"
```

**参数说明**

|枚举值|说明|
|--|--|
|LINEAR|线性初始化。|
|TRUNCATED_NORMAL|截断正态分布初始化。|
|UNIFORM|均匀分布初始化。|

**说明**

该枚举类无需初始化，使用时直接引用对应枚举值即可。

**使用示例**

```python
from torchrec_embcache.distributed.configs import InitializerType

init_type = InitializerType.LINEAR
```

## Saver<a name="ZH-CN_TOPIC_0000002420844874"></a>

**功能描述<a name="section634582619155"></a>**

多级缓存稀疏表保存加载功能类，提供多级缓存稀疏表数据（稀疏表Embedding，Embedding对应的优化器参数等）的保存、加载接口。

**函数原型<a name="section1483104721911"></a>**

```python
class Saver:
    def __init__(self, rank: int = None):

    def save(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:

    def load(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:
```

**使用约束<a name="section72467171850"></a>**

1. 保存/加载接口仅支持多级缓存保存/加载稀疏表相关数据（稀疏表Embedding，Embedding对应的优化器参数等）。
2. 不支持保存/加载Dense数据（需自行调用Torch原生接口）。
3. 不支持纯显存模式下稀疏表保存/加载。
4. 保存/加载接口仅支持保存/加载本地文件系统。
5. 增量保存/加载功能不支持准入淘汰功能，同时开启将触发配置校验错误。
6. 增量保存/加载功能，仅适用于pipeline训练模式下生成的增量数据。
7. 差异卡加载功能不支持准入淘汰，同时开启将触发配置校验错误。
8. pipeline模式下，如果需要在训练过程中（即Dataset未迭代到末尾）执行保存，需要在保存前手动触发wait_pipeline_compute_swapinfo。
    详细参考[保存加载用例](../../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py)。

**参数说明**

|**参数名**|**类型**|**可选/必选**|**说明**|
|--|--|--|--|
|rank|int|可选|当前进程在整个world_size中的rank。当torch分布式环境已初始化时，该参数为可选，此时将使用torch.distributed.get_rank()获取rank；否则该参数为必选。|
|module|torch.nn.Module|必选|模型对象实例。模型（或子模型）需包含类型为EmbCacheShardedEmbeddingBagCollection/EmbCacheShardedEmbeddingCollection的模型实例，且深度不能超过500。使用多级缓存支持的创表接口/分表接口进行模型创建和模型分片时即满足要求。|
|path|str|必选|保存/加载路径，长度取值范围：[1,1024]。<br>保存/加载的路径中不能包含软链接和敏感字符（Key、password、privatekey），不能使用特殊路径（如/usr下的路径），且路径的权限不能高于750。|
|incremental|bool|可选|是否开启增量保存/加载功能。默认为False，表示不开启。增量保存/加载功能适用于pipeline训练模式下生成的增量数据，且需要创建表时在[EmbCacheEmbeddingBagConfig](02_table_creation_apis.md#embcacheembeddingbagconfig)/[EmbCacheEmbeddingConfig](02_table_creation_apis.md#embcacheembeddingconfig)配置is_incremental参数为True。|

>[!NOTE]
>
>1. 调用save/load接口时，若incremental为True，表示以增量的方式进行保存/加载稀疏表数据。
>2. 增量加载时，一般先加载全量保存的base数据；再对多份增量数据，逐个调用load接口并设置incremental为True进行增量加载。
>3. 全量/增量的保存/加载操作不强制要求保存/加载的数量/顺序，仅在实现逻辑上有所差异。

**返回值说明**

- 成功：接口调用无报错，保存落盘/加载稀疏表数据。
- 失败：抛出异常。

**使用示例**

注：样例中ddp_model为调用[DistributedModelParallel（TorchRec）](05_subtable_apis.md#TOPIC_0000002338384297)接口并传入多级缓存模式支持的稀疏表（[EmbCacheEmbeddingCollection](02_table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](02_table_creation_apis.md#embcacheembeddingbagcollection)）创建的模型对象，当前示例中使用EmbCacheEmbeddingCollection。

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

from torchrec_embcache.distributed.configs import EmbCacheEmbeddingConfig
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder
from torchrec_embcache.saver import Saver
from torchrec_embcache.utils import safe_makedirs


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
batch_size = 100
embedding_dims: list[int] = [64, 64, 64]
num_embeddings: list[int] = [400, 4000, 400]
table_num: int = len(num_embeddings)
world_size = dist.get_world_size()
npu_device = torch.device("npu")

# 创建模型
embedding_configs: list[EmbCacheEmbeddingConfig] = []
table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
for i in range(table_num):
    emb_config = EmbCacheEmbeddingConfig(
        name=table_names[i],
        embedding_dim=embedding_dims[i],
        num_embeddings=num_embeddings[i],
        feature_names=[feat_names[i]],
    )
    embedding_configs.append(emb_config)
sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
    embedding_configs,  # type: ignore
    world_size,
    batch_size,
    multi_hot_sizes=[1] * table_num,
    device=torch.device("meta"),
)
dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), DENSE_OUTPUT_DIM)
test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

# 定义优化器
embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.01, "eps": 0.1}
apply_optimizer_in_backward(
    embedding_optimizer_class,
    test_model.sparse_model.parameters(),
    optimizer_kwargs=optimizer_kwargs,
)

# 稀疏表分表
cpu_pg = dist.new_group(backend="gloo")
cpu_env = ShardingEnv.from_process_group(cpu_pg)  # pyright: ignore[reportArgumentType]
cpu_device = torch.device("cpu")
sharders: list[EmbCacheEmbeddingCollectionSharder] = [
    EmbCacheEmbeddingCollectionSharder(
        cpu_device=cpu_device,
        cpu_env=cpu_env,
        npu_device=npu_device,
        npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),  # type: ignore
    ),
]
constraints: dict[str, ParameterConstraints] = {
    table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
    for table_name in table_names
}
planner: EmbeddingShardingPlanner = EmbeddingShardingPlanner(
    topology=Topology(world_size=world_size, compute_device="npu"),
    constraints=constraints,
)
plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)  # type: ignore
ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)  # type: ignore

saver = Saver(rank=rank)
save_dir = os.path.abspath("save_dir")
sparse_save_dir = os.path.join(save_dir, "sparse")
if not os.path.exists(save_dir):
    safe_makedirs(save_dir)
saver.save(ddp_model, sparse_save_dir)  # 保存
saver.load(ddp_model, sparse_save_dir)  # 加载
```
