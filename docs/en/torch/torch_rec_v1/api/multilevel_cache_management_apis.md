# Multi-Level Cache Management APIs

## `InitializerType`

**Description**

An enumeration class that defines how to initialize sparse table weights in multilevel cache mode.

**Function Prototype**

```python
class InitializerType(Enum):
    LINEAR = "linear"
    TRUNCATED_NORMAL = "truncated_normal"
    UNIFORM = "uniform"
```

**Enumeration Values**

|Enumeration Value|Description|
|--|--|
|LINEAR|Linear initialization|
|TRUNCATED_NORMAL|Truncated normal initialization|
|UNIFORM|Uniform initialization|

**Notes**

This enumeration class does not require initialization. When using it, directly reference the corresponding enumeration value.

**Example**

```python
from torchrec_embcache.distributed.configs import InitializerType

init_type = InitializerType.LINEAR
```

## `Saver`

**Description**

A utility class for saving and loading multilevel cache sparse tables. It provides save and load APIs for multilevel cache sparse table data, such as sparse table embeddings and optimizer parameters for those embeddings.

**Function Prototype**

```python
class Saver:
    def __init__(self, rank: int = None):

    def save(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:

    def load(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:
```

**Constraints**

1. The saving and loading APIs support only multilevel cache saving and loading of sparse table data, such as sparse table embeddings and optimizer parameters for the embeddings.
2. Dense data cannot be saved or loaded. You must call the native Torch APIs.
3. Saving and loading sparse tables in full-device memory mode is not supported.
4. The saving and loading APIs support only the local file system.
5. The incremental saving and loading feature does not support admission and eviction. If you enable both at the same time, configuration validation fails.
6. The incremental saving and loading feature applies only to incremental data generated in pipeline training mode.
7. Differential device loading does not support admission and eviction. If you enable both at the same time, configuration validation fails.
8. In pipeline mode, if you need to save during training, that is, before the `Dataset` completes an iteration, you must manually trigger `wait_pipeline_compute_swapinfo` before saving.
    For details, see [saving and loading example](../../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py).

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|rank|int|Optional|Rank of the current process in the full `world_size`. When the PyTorch distributed environment is already initialized, this parameter is optional. In that case, `torch.distributed.get_rank()` gets the rank. Otherwise, this parameter is required.|
|module|torch.nn.Module|Mandatory|Model object instance. The model, or one of its submodules, must include an instance of `EmbCacheShardedEmbeddingBagCollection` or `EmbCacheShardedEmbeddingCollection`, and the nesting depth must not exceed 500. This requirement is met when you create and shard the model with the table creation and sharding APIs supported by multi-level cache.|
|path|str|Mandatory|Path for saving and loading. The value length range is [1,1024].<br>The path for saving and loading cannot contain symbolic links or sensitive strings (`Key`, `password`, `privatekey`). Do not use special paths, such as paths in `/usr`, and keep the permissions on the path no higher than 750.|
|incremental|bool|Optional|Indicates whether to enable incremental saving and loading. The default value is `False`, which means this feature is disabled. Incremental saving and loading applies to incremental data generated in pipeline training mode. To use this feature, set the `is_incremental` parameter to `True` in [EmbCacheEmbeddingBagConfig](table_creation_apis.md#embcacheembeddingbagconfig) or [EmbCacheEmbeddingConfig](table_creation_apis.md#embcacheembeddingconfig) when you create the table.|

>[!NOTE]
>
>1. When you call the `save` or `load` APIs, if `incremental` is `True`, the sparse table data is saved or loaded incrementally.
>2. For incremental loading, load the full base data first, and then call the `load` API for each piece of incremental data with `incremental` set to `True`.
>3. Full and incremental saving and loading operations do not enforce a specific quantity or order of saves and loads. They differ only in implementation logic.

**Returns**

- Success: The API call completes without errors and saves or loads the sparse table data.
- Failure: An exception is thrown.

**Example**

Note: In this example, `ddp_model` is a model object created by calling the [DistributedModelParallel (TorchRec)](subtable_apis.md#TOPIC_0000002338384297) API and passing in sparse tables supported by multilevel cache mode, namely [EmbCacheEmbeddingCollection](table_creation_apis.md#embcacheembeddingcollection) or [EmbCacheEmbeddingBagCollection](table_creation_apis.md#embcacheembeddingbagcollection). This example uses `EmbCacheEmbeddingCollection`.

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
        # forward must return loss and output, with loss first and output second
        return loss, output


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


# Simulate initialization of a single-device torch.distributed environment. In actual use, WORLD_SIZE and RANK may be set by the PyTorch distributed training launcher
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

# Create the model
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

# Define the optimizer
embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.01, "eps": 0.1}
apply_optimizer_in_backward(
    embedding_optimizer_class,
    test_model.sparse_model.parameters(),
    optimizer_kwargs=optimizer_kwargs,
)

# Shard the sparse tables
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
saver.save(ddp_model, sparse_save_dir)  # Save
saver.load(ddp_model, sparse_save_dir)  # Load
```
