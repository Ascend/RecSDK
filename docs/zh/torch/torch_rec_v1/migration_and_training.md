# 迁移与训练<a name="ZH-CN_TOPIC_0000002302389292"></a>

## 训练场景介绍<a name="ZH-CN_TOPIC_0000002302229612"></a>

**基于Rec SDK Torch搭建网络<a name="section16627105015515"></a>**

用户可按[快速入门](./quick_start.md)的步骤搭建模型并进行训练。

**基于开源TorchRec进行迁移<a name="section9248145363514"></a>**

如果用户已经在TorchRec上搭建了网络，则按照接口对应关系进行替换，如[表1](#table16435142101913)所示。

**表 1**  接口对应关系
<a id="table16435142101913"></a>

|TorchRec接口|Rec SDK Torch接口|接口功能描述|
|--|--|--|
|EmbeddingBagConfig|HashEmbeddingBagConfig|稀疏表配置|
|EmbeddingBagCollection|HashEmbeddingBagCollection|创建稀疏表|
|get_default_sharders|get_default_hybrid_sharders|获取分表器|
|TrainPipelineSparseDist|HybridTrainPipelineSparseDist|查询稀疏表|

接口示例：

- TorchRec示例：

    ```python
    import torch.distributed as dist
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import get_default_sharders
    class TestModel(torch.nn.Module):
        def __init__(self, *):
            # Rec SDK Torch 使用的接口为HashEmbeddingBagCollection
            self.sparse_model = EmbeddingBagCollection(xx)
            self.dense_model = xx
        def forward(self, batch: Batch):
            # sparse(self.ebc)前向、dense前向调用
            # 注意：模型前向返回值需loss在前，output在后，对齐TorchRec原生TrainPipelineSparseDist使用方式
            return loss, output
    def invoke_main():
        dist.init_process_group(backend="hccl")    
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("npu")
    
        dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
        data_loader = DataLoader(
            dataset,
        )
        test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS, NUM_EMBEDS)
        ...
        sharder创建
        ...
        #  Rec SDK Torch 使用的接口为get_default_hybrid_sharders
        hybrid_sharder = get_default_sharders()
        ...
        优化器创建
        ...
        # Rec SDK Torch 使用的接口为HybridTrainPipelineSparseDist
        pipeline = TrainPipelineSparseDist()
        for i in range(20):
            output = pipeline.progress(batched_iterator)
    ```

- Rec SDK Torch示例：

    ```python
    import torch.distributed as dist
    from hybrid_torchrec import HashEmbeddingBagCollection
    from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
    from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
    ...
    class TestModel(torch.nn.Module):
        def __init__(self, *):
            # 原生TorchRec使用的接口为EmbeddingBagCollection
            self.sparse_model = HashEmbeddingBagCollection(xx)
            self.dense_model = xx
        def forward(self, batch: Batch):
            # sparse前向、dense前向调用
            # 注意：模型前向返回值需loss在前，output在后，对齐TorchRec原生TrainPipelineSparseDist使用方式
            return loss, output
    def invoke_main():
        dist.init_process_group(backend="hccl")    
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("npu")
        
        # Rec SDK Torch创建host连接
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
        data_loader = DataLoader(
            dataset,
        )
        test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS, NUM_EMBEDS)
        ...
        sharder创建
        ...
        # 原生TorchRec使用的接口为get_default_sharders
        hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)
        ...
        优化器创建
        ...
        # 原生TorchRec使用的接口为TrainPipelineSparseDist
        pipeline = HybridTrainPipelineSparseDist()
        for i in range(20):
            output = pipeline.progress(batched_iterator)
    ```

## Rec SDK Torch迁移样例<a name="ZH-CN_TOPIC_0000002336268713"></a>

Rec SDK Torch支持Torch开源推荐模型迁移适配，迁移样例可以参考如下：

[Rec SDK Torch DCNv2迁移样例](https://gitcode.com/Ascend/RecSDK/blob/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/dlrm/README.md)。

## 功能特性介绍

Rec SDK Torch 支持纯显存模式和多级缓存模式两种训练模式。

纯显存模式指在训练时，所有稀疏表Embedding数据都存储在Device内存中。

多级缓存模式指在训练时，会以Device Memory + DDR（Host Memory）结合的方式存储Embedding。

本章节将介绍纯显存模式和多级缓存模式的相关功能特性和代码使用示例。

> [!NOTE]
> 
> 本文档中的**约束**章节内容仅介绍主要使用场景下的约束信息，详细信息请参见对应的API文档。

### 基于纯显存模式训练

**纯显存模式对比原生TorchRec框架的API差异**

Rec SDK Torch纯显存模式在配置稀疏表、创建稀疏表、稀疏表分表、创建pipeline时使用的API接口对比原生TorchRec框架有差异。

|场景/API|原生TorchRec框架|Rec SDK Torch纯显存模式|
|--|--|--|
|配置稀疏表|EmbeddingBagConfig|HashEmbeddingBagConfig|
|创建稀疏表|EmbeddingBagCollection|HashEmbeddingBagCollection|
|稀疏表分表器|get_default_sharders()|get_default_hybrid_sharders()|
|创建pipeline|TrainPipelineSparseDist|HybridTrainPipelineSparseDist|

**约束**

- 仅支持EBC模式。
- 支持pipeline模式和非pipeline模式（直接调用分片后的稀疏表前向）进行训练。
- 不支持保存/加载稀疏表数据。

**纯显存模式测试用例**

纯显存模式的完整测试用例请参见[README](../../../../training/torch_rec_v1/hybrid_torchrec/test/st/README.md)。

#### 基础使用<a id="basic_usage_device_memory"></a>

训练流程上和原生TorchRec框架pipeline模式相同，仅在创建稀疏表、稀疏表分表、创建pipeline时使用的API有差异。

**代码示例**

```python
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchrec
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from hybrid_torchrec import HashEmbeddingBagConfig, HashEmbeddingBagCollection
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders

from dataset import RandomRecDataset, Batch

logging.getLogger().setLevel(logging.INFO)

BATCH_NUM: int = 20
DENSE_OUTPUT_DIM: int = 2


class DenseModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 定义dense layer
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 定义dense层实现
        return self.linear(x)


class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        # sparse前向
        sparse_output: EmbeddingBagCollectionAwaitable = self.sparse_model(batch.sparse_features)
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output[feat_name])  # type: ignore
        # 合并所有稀疏特征的embedding
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # dense前向
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # 计算loss
        loss: torch.Tensor = self.loss_fn(dense_output, batch.labels)

        # 自行定义output输出内容
        output: dict[str, torch.Tensor] = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output

        # 返回前向输出
        # 注意：forward必须返回loss和output两个值，且loss在前，output在后；该用法为TorchRec原生TrainPipelineSparseDist用法
        return loss, output


def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


def set_distribute_env():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(rank)  # type: ignore
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(backend="hccl")


def train():
    # 0.设置分布式环境
    set_distribute_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    npu_device = torch.device("npu")

    embedding_dims: list[int] = [64, 16, 32]
    num_embeddings: list[int] = [400, 4000, 400]
    table_num: int = len(num_embeddings)

    # 1.创建数据集

    batch_size: int = 128
    dataset = RandomRecDataset(BATCH_NUM, batch_size, num_embeddings, table_num)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    # 2.创建稀疏表
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

    # 3.创建模型：将稀疏表和Dense部分包装到一个模型中
    dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), DENSE_OUTPUT_DIM)
    test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

    # 4.定义稀疏表优化器
    embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
    optimizer_kwargs = {"lr": 0.01, "eps": 0.1}
    apply_optimizer_in_backward(
        embedding_optimizer_class,
        test_model.sparse_model.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    # 5.稀疏表分表
    host_gp = dist.new_group(backend="gloo")
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    sharders = get_default_hybrid_sharders(host_env=host_env)
    constraints = {
        # 分表方式为row_wise，compute_kernels使用fused
        table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
        for table_name in table_names
    }
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)
    #   此处分表会根据sharders参数匹配对应class类型：EmbeddingBagCollection/HashEmbeddingBagCollection，
    #   只会对稀疏表参数进行分表，不会对dense参数分表
    ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)

    # 6.整合优化器
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])

    # 7.创建pipeline
    pipeline = HybridTrainPipelineSparseDist(
        ddp_model, optimizer, npu_device, return_loss=True, execute_all_batches=True
    )

    # 8.使用pipeline进行训练
    dataset_iterator = iter(data_loader)
    for step in range(BATCH_NUM):
        # 前面创建pipeline时，设置了return_loss=True，所以pipeline.progress()会返回output和loss两个值
        # 若创建pipeline时设置return_loss=False，则仅返回output
        output, loss = pipeline.progress(dataset_iterator)
        logging.info("rank: %d, step: %s, loss: %s, sparse output: %s", rank, step, loss, output["sparse"])


if __name__ == "__main__":
    train()
    # 脚本使用方式：
    #   1.将脚本写入到main.py文件中，并拷贝RecSDK/training/torch_rec_v1/hybrid_torchrec/test/st/dataset.py到py文件同一目录
    #   2.启动单卡： WORLD_SIZE=1 RANK=0 python3 main.py
    #   3.启动多卡（2卡）： torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=2 main.py
```

更多demo样例请参见[Rec SDK Torch Little Demo样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_examples/little_demo)。

#### 稀疏表数据并行（DP）

**约束**

- DP模式仅支持单个稀疏表。

**对比非DP模式**

DP模式对比[非DP模式（基础使用）](#basic_usage_device_memory)，主要区别在于训练时稀疏表参数存储方式不同。

DP模式下，稀疏表将不再切分到不同的Device，而是每个Device都持有完整的稀疏表参数。在反向传播时会聚合所有Device的梯度再统一更新。

**代码示例**

DP模式完整代码示例请参见[DP模式测试用例](../../../../training/torch_rec_v1/hybrid_torchrec/test/st/test_hybrid_hash_embeddingbag_dp.py)。

简化代码示例（仅展示和基础使用模式差异部分）：

```python
    # 差异点：DP模式仅支持单表，不支持多表模式。
    embedding_dims = [64]
    num_embeddings = [400]

    ...

    # 5.稀疏表分表
    host_gp = dist.new_group(backend="gloo")
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    sharders = get_default_hybrid_sharders(host_env=host_env)
    constraints = {
        # 差异点：分表方式为data_parallel，compute_kernels使用dense
        table_name: ParameterConstraints(sharding_types=["data_parallel"], compute_kernels=["dense"])
        for table_name in table_names
    }
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)
    dmp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)

    ...
```

### 基于多级缓存模式训练

**多级缓存模式和纯显存模式功能差异**

- 多级缓存模式稀疏表存储方式：Device Memory作为Cache，存储训练时所需的Embedding参数；DDR为完整的稀疏表，存储和管理全量的Embedding参数。
- 支持EC模式和EBC模式。
- 支持保存/加载稀疏表参数。
- 仅支持稀疏表的row-wise分表方式。
- 仅支持通过pipeline模式进行训练。
  - pipeline模式训练时，会在调用`sparse_model.forward()`前执行换入（H2D）换出（D2H）操作，保证当前批次的训练数据均在Device Memory中。

**多级缓存模式和纯显存模式使用API差异**

多级缓存模式在配置稀疏表、创建稀疏表、稀疏表分表、创建pipeline时使用的API接口和纯显存模式有差异。

|场景/API|纯显存模式|多级缓存EC模式|多级缓存EBC模式|
|--|--|--|--|
|配置稀疏表|HashEmbeddingBagConfig|EmbCacheEmbeddingConfig|EmbCacheEmbeddingBagConfig|
|创建稀疏表|HashEmbeddingBagCollection|EmbCacheEmbeddingCollection|EmbCacheEmbeddingBagCollection|
|稀疏表分表器|get_default_hybrid_sharders()|[EmbCacheEmbeddingCollectionSharder()]|[EmbCacheEmbeddingBagCollectionSharder()]|
|创建pipeline|HybridTrainPipelineSparseDist|EmbCacheTrainPipelineSparseDist|EmbCacheTrainPipelineSparseDist|

**多级缓存模式测试用例**

多级缓存模式的完整测试用例请参见[README](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/README.md)。

#### 基础使用<a id="basic_usage_embcache"></a>

##### 多级缓存EC模式

**代码示例**

```python
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchrec
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.embedding import EmbeddingCollectionAwaitable
from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
from torchrec_embcache.distributed.configs import AdmitAndEvictConfig, EmbCacheEmbeddingConfig, InitializerType
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder

from dataset import RandomRecDataset, Batch

logging.getLogger().setLevel(logging.INFO)

OUTPUT_SIZE = 2
BATCH_NUM: int = 50


class DenseModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 定义dense layer
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 定义dense层实现
        return self.linear(x)


class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        # sparse前向
        sparse_output: EmbeddingCollectionAwaitable = self.sparse_model(batch.sparse_features)
        sparse_output_dict: dict[str, torchrec.JaggedTensor] = sparse_output.wait()  # type: ignore
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output_dict[feat_name].values())
        # 拼接稀疏表的embedding
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # dense前向
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # 计算loss
        loss = self.loss_fn(dense_output, batch.labels)
        
        # 自行定义output输出内容
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # 返回前向输出
        # 注意：forward必须返回loss和output两个值，且loss在前，output在后；该用法为TorchRec原生TrainPipelineSparseDist用法
        return loss, output


def set_distribute_env():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(rank)  # type: ignore
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(backend="hccl")


def train():
    # 0.设置分布式环境
    set_distribute_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    npu_device = torch.device("npu")

    embedding_dims: list[int] = [64, 64, 64]  # TorchRec EC模式约束多个表必须使用相同的dim
    num_embeddings: list[int] = [400, 4000, 400]
    table_num: int = len(num_embeddings)

    # 1.创建数据集
    batch_size: int = 128
    dataset: RandomRecDataset = RandomRecDataset(BATCH_NUM, batch_size, num_embeddings, table_num)
    data_loader: DataLoader[Batch] = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    # 2.创建稀疏表
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        emb_config = EmbCacheEmbeddingConfig(
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # 注意：多级缓存模式，不支持自定义初始化函数
            initializer_type=InitializerType.UNIFORM,  # embedding初始化方式通过initializer_type参数设置
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
        )
        embedding_configs.append(emb_config)
    sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
        embedding_configs,  # type: ignore
        world_size,
        batch_size,
        multi_hot_sizes=[1] * table_num,
        device=torch.device("meta"),
    )

    # 3.创建模型，将稀疏表和Dense部分包装到一个模型中
    dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), OUTPUT_SIZE)
    test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

    # 4.定义稀疏表优化器
    embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
    optimizer_kwargs = {"lr": 0.01, "eps": 0.1}
    apply_optimizer_in_backward(
        embedding_optimizer_class,
        test_model.sparse_model.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    # 5.稀疏表分表
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
        # 分表方式为row_wise，compute_kernels使用fused
        table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
        for table_name in table_names
    }
    planner: EmbeddingShardingPlanner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan: EmbeddingShardingPlan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)  # type: ignore
    #   此处分表会根据sharders参数匹配对应class类型：EmbCacheEmbeddingCollection，
    #   只会对稀疏表参数进行分表，不会对dense参数分表
    ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)  # type: ignore

    # 6.整合优化器
    dense_optimizer: KeyedOptimizerWrapper = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])

    # 7.创建pipeline
    pipeline: EmbCacheTrainPipelineSparseDist = EmbCacheTrainPipelineSparseDist(
        ddp_model,
        optimizer,
        cpu_device=cpu_device,
        npu_device=npu_device,
        return_loss=True,
    )

    # 8.使用pipeline进行训练
    dataset_iterator = iter(data_loader)
    for step in range(BATCH_NUM):
        # 前面创建pipeline时，设置了return_loss=True，所以pipeline.progress()会返回output和loss两个值
        output, loss = pipeline.progress(dataset_iterator)
        logging.info("rank: %d, step: %s, loss: %s, sparse output: %s", rank, step, loss, output["sparse"])


if __name__ == "__main__":
    train()
    # 脚本使用方式：
    #   1.将脚本写入到main.py文件中，并拷贝RecSDK/training/torch_rec_v1/torchrec_embcache/tests/acc_test/dataset.py到py文件同一目录
    #   2.启动单卡： WORLD_SIZE=1 RANK=0 python3 main.py
    #   3.启动多卡（2卡）： torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=2 main.py
```

更多代码样例请参见[多级缓存EC测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_pipeline.py)。

##### 多级缓存EBC模式

**约束**

- 多级缓存EBC模式仅支持多个稀疏表使用相同的Embedding Dim。

**对比多级缓存EC模式差异**

EBC模式对比EC模式，使用的创建稀疏表和分表器的API存在差异；以及整合模型的`forward()`中对稀疏表查表结果的处理方式存在差异。

**代码示例**

完整代码示例请参见[多级缓存EBC测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_cache_pipeline.py)。

简化代码示例（仅展示和多级缓存EC模式的差异部分）：

```python
...

# 差异：导入多级缓存EBC模式使用的API
from torchrec_embcache.distributed.configs import EmbCacheEmbeddingBagConfig, InitializerType
from torchrec_embcache.distributed.embedding_bag import EmbCacheEmbeddingBagCollection
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingBagCollectionSharder

...

class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        # sparse前向
        sparse_output: EmbeddingBagCollectionAwaitable = self.sparse_model(batch.sparse_features)
        # 差异：解析embedding前向查表结果方式和多级缓存EC模式有差异
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output[feat_name])  # type: ignore
        # 合并所有稀疏特征的embedding
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # dense前向
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # 计算loss
        loss: torch.Tensor = self.loss_fn(dense_output, batch.labels)

        # 自行定义output输出内容
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # 返回前向输出
        # 注意：forward必须返回loss和output两个值，且loss在前，output在后；该用法为TorchRec原生TrainPipelineSparseDist用法
        return loss, output

...

def train():
    ...

    # 2.创建稀疏表
    embedding_configs: list[EmbCacheEmbeddingBagConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        emb_config = EmbCacheEmbeddingBagConfig(  # 差异：EBC模式创建表的config、collection API和EC模式有差异
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # 注意：多级缓存模式，不支持自定义初始化函数
            initializer_type=InitializerType.UNIFORM,  # embedding初始化方式通过initializer_type参数设置
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
        )
        embedding_configs.append(emb_config)
    sparse_ebc: torch.nn.Module = EmbCacheEmbeddingBagCollection(
        embedding_configs,  # type: ignore
        world_size,
        batch_size,
        multi_hot_sizes=[1] * table_num,
        device=torch.device("meta"),
    )

    ...

    # 5.稀疏表分表
    cpu_pg = dist.new_group(backend="gloo")
    cpu_env = ShardingEnv.from_process_group(cpu_pg)  # pyright: ignore[reportArgumentType]
    cpu_device = torch.device("cpu")
    sharders: list[EmbCacheEmbeddingBagCollectionSharder] = [  # 差异：EBC模式分表器和EC模式有差异
        EmbCacheEmbeddingBagCollectionSharder(
            cpu_device=cpu_device,
            cpu_env=cpu_env,
            npu_device=npu_device,
            npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),  # type: ignore
        ),
    ]
    constraints: dict[str, ParameterConstraints] = {
        # 分表方式为row_wise，compute_kernels使用fused
        table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
        for table_name in table_names
    }
    planner: EmbeddingShardingPlanner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan: EmbeddingShardingPlan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)  # type: ignore
    #   此处分表会根据sharders参数匹配对应class类型：EmbCacheEmbeddingBagCollection，
    #   只会对稀疏表参数进行分表，不会对dense参数分表
    ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)  # type: ignore

    ...

```

#### 准入淘汰

多级缓存模式下，支持**基于时间和计数**、**基于展示点击和分数**两种准入淘汰策略。

> [!NOTICE] 须知
> 
> **仅支持单独使用其中一种准入淘汰策略，不支持两种策略混合使用。**

##### 准入淘汰（基于时间和计数）

**说明**

- 准入
  - 统计特征ID出现次数：超过准入策略参数阈值的特征ID会被准入，未超过阈值的特征ID不会被准入。未准入的特征ID的Embedding查表结果为全0.0向量（值可在准入策略中配置）。
  - 特征ID的出现次数会进行AllToAll通信，确保所有进程都能获取到所有特征ID的出现次数。
- 淘汰
  - 统计特征ID的时间戳，和特征ID所在Embedding表中最新的时间戳（该稀疏表中所有特征ID的时间戳取max）进行计算得到一个差值，差值超过淘汰策略参数阈值的特征ID和Embedding会被淘汰。
  - 特征ID的时间戳未进行AllToAll通信，每个进程仅统计当前进程读取到的特征ID的时间戳数据。

**约束**

- 仅支持多级缓存EC模式。
- 准入与淘汰可单独开启或同时开启。

**对比多级缓存EC模式的代码差异**

- 准入场景
  - 创建EmbCacheEmbeddingConfig时，需指定admit_and_evict_config参数，并配置其中的准入策略参数。
- 淘汰场景
  - 构造的Dataset中的数据，需包含和特征ID对应的时间戳数据。
  - 创建EmbCacheEmbeddingConfig时，需指定admit_and_evict_config参数，并配置其中的淘汰策略参数。
  - 创建pipeline时，需指定evict_step_interval参数，用于配置触发淘汰操作的步数间隔。

**代码示例**

完整代码示例请参见[准入淘汰（基于时间和计数）测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_feature_filter.py)。

简化代码示例（仅展示和多级缓存EC模式的差异部分）：

```python
    ...

    # 1.创建数据集
    batch_size: int = 128
    # 差异：淘汰场景：需创建特征ID时间戳的数据集，详细实现请参见dataset.py中RandomRecDataset定义
    dataset: RandomRecDataset = RandomRecDataset(
        BATCH_NUM, batch_size, num_embeddings, table_num, is_evict_enabled=True
    )
    data_loader: DataLoader[Batch] = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    # 2.创建稀疏表
    evict_step_interval = 20  # 淘汰步长，每20个batch淘汰一次
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        # 差异：设置准入淘汰配置参数并传递给emb_config
        admit_and_evict_config = AdmitAndEvictConfig(
            admit_threshold=2,
            not_admitted_default_value=0.999,
            evict_threshold=2000_0000,
            evict_step_interval=evict_step_interval,
        )
        emb_config = EmbCacheEmbeddingConfig(
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # 注意：多级缓存模式，不支持自定义初始化函数
            initializer_type=InitializerType.UNIFORM,  # embedding初始化方式通过initializer_type参数设置
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
            admit_and_evict_config=admit_and_evict_config,  # 差异：传递准入淘汰配置参数
        )
        embedding_configs.append(emb_config)
    sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
        embedding_configs,  # type: ignore
        world_size,
        batch_size,
        multi_hot_sizes=[1] * table_num,
        device=torch.device("meta"),
    )

    ...

    # 7.创建pipeline
    pipeline: EmbCacheTrainPipelineSparseDist = EmbCacheTrainPipelineSparseDist(
        ddp_model,
        optimizer,
        cpu_device=cpu_device,
        npu_device=npu_device,
        return_loss=True,
        evict_step_interval=evict_step_interval,  # 差异：淘汰场景，将淘汰步长参数传递给pipeline
    )

    ...

```

##### 准入淘汰（基于展示点击和分数）

**说明**

- 准入
  - 使用展示次数与点击次数按权重合成准入分数：`score = alpha × 展示 + beta × 点击（参数见 ShowClickParams）`。
  - 当 `showclick_params.admit_threshold > 0` 时表示开启准入：分数低于准入阈值的特征视为未准入。特征ID的Embedding查表结果为全0.0向量（值可在准入策略中配置）。
  - 特征ID的出现次数会进行AllToAll通信，确保所有进程都能获取到所有特征ID的出现次数。
- 淘汰
  - 维护淘汰分数：每步按 `new_score = (oldScore + alpha × 展示 + beta × 点击) × score_decay` 更新（score_decay 为 1 表示不衰减）。
  - 当 `evict_percentage > 0` 时表示开启淘汰：按分数升序取约 evict_percentage 比例的Embedding进行淘汰。

**约束**

- 仅支持多级缓存EC模式。
- 准入与淘汰可单独开启或同时开启。

**代码示例**

完整代码示例请参见[准入淘汰（基于时间和计数）测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_show_click.py)。

简化代码示例（仅展示和多级缓存EC模式的差异部分）：

```python
# 差异：导入ShowClickBatch
from dataset import RandomRecDataset, Batch, ShowClickBatch

...

class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: ShowClickBatch):  # 差异: typing类型为ShowClickBatch
        # sparse前向
        sparse_output: EmbeddingCollectionAwaitable = self.sparse_model(batch.sparse_features)
        sparse_output_dict: dict[str, torchrec.JaggedTensor] = sparse_output.wait()  # type: ignore
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output_dict[feat_name].values())
        # 拼接稀疏表的embedding
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # dense前向
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # 计算loss
        loss = self.loss_fn(dense_output, batch.click_labels)  # 差异: ShowClickBatch中labels字段名为click_labels

        # 自行定义output输出内容
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # 返回前向输出
        # 注意：forward必须返回loss和output两个值，且loss在前，output在后；该用法为TorchRec原生TrainPipelineSparseDist用法
        return loss, output


def train():
    ...

    # 1.创建数据集
    batch_size: int = 128
    # 差异：创建包含click_labels的数据集，详细实现请参见dataset.py中RandomRecDataset定义
    dataset: RandomRecDataset = RandomRecDataset(
        BATCH_NUM, batch_size, num_embeddings, table_num, is_enable_score=True
    )
    data_loader: DataLoader[Batch] = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        pin_memory_device="npu",
        num_workers=1,
    )

    # 2.创建稀疏表
    evict_step_interval = 20  # 淘汰步长，每20个batch淘汰一次
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    showclick_params = ShowClickParams(alpha=1, beta=1, admit_threshold=0.1, evict_percentage=0.1, score_decay=0.9)
    for i in range(table_num):
        # 差异：设置show click准入淘汰配置参数并传递给emb_config
        admit_and_evict_config = AdmitAndEvictConfig(
            showclick_params=showclick_params,
            not_admitted_default_value=0.999,
            evict_step_interval=evict_step_interval,
            policy_type=AdmitAndEvictPolicyType.POLICY_SHOWCLICK,  # type: ignore
        )
        emb_config = EmbCacheEmbeddingConfig(
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # 注意：多级缓存模式，不支持自定义初始化函数
            initializer_type=InitializerType.UNIFORM,  # embedding初始化方式通过initializer_type参数设置
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
            admit_and_evict_config=admit_and_evict_config,  # 差异：传递准入淘汰配置参数
        )
        embedding_configs.append(emb_config)
    sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
        embedding_configs,  # type: ignore
        world_size,
        batch_size,
        multi_hot_sizes=[1] * table_num,
        device=torch.device("meta"),
    )

    ...

    # 7.创建pipeline
    pipeline: EmbCacheTrainPipelineSparseDist = EmbCacheTrainPipelineSparseDist(
        ddp_model,
        optimizer,
        cpu_device=cpu_device,
        npu_device=npu_device,
        return_loss=True,
        evict_step_interval=evict_step_interval,  # 差异：淘汰场景，将淘汰步长参数传递给pipeline
    )

    ...
```

#### 梯度累积

**说明**

梯度累积的核心思想是将原本需要一次性计算的大批量（batch）数据拆分成多个小批量（micro-batch）进行训练，每次训练时计算梯度并累加。当累加到一定步数时再用累积后的梯度进行一次参数更新。

梯度累积的等效batch size：`micro-batch_size * 累积步数`。

在梯度累积场景，由于调小了batch_size，因此训练时占用的Device Memory更少，适合Device Memory有限但需要大batch_size训练的场景。

**约束**

- 需使用Rec SDK Torch提供的适配梯度累积功能的优化器，才支持梯度累积功能。支持的优化器请参见[optimizer_class参数说明](./api/optimizers_apis.md#section888634319218)。
- 仅支持EC模式。

**对比多级缓存EC模式的API差异**

梯度累积场景对比多级缓存EC模式的主要差异在于需要**使用支持梯度累积的优化器**，并在优化器初始化参数中**设置累积步数参数**。

**代码示例**

完整代码示例请参见[梯度累积测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_aggregation.py)。

简化代码示例（仅展示和多级缓存EC模式的差异部分）：

```python
...

# 差异：导入支持梯度累积的优化器
from torchrec.optim import AccumulateAdagrad, AccumulateAdam, AccumulateSGD

...

def train():
    ...

    # 4.定义稀疏表优化器
    # 差异：使用支持梯度累积的optimizer_class
    embedding_optimizer_class: type[torch.optim.Optimizer] = AccumulateAdagrad
    # 差异：设置梯度累积参数。use_accumulate为True表示开启梯度累积，accumulate_step为累积步数
    optimizer_kwargs = {"lr": 0.01, "eps": 0.1, "use_accumulate": True, "accumulate_step": 10}
    apply_optimizer_in_backward(
        embedding_optimizer_class,
        test_model.sparse_model.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    ...
```

#### 保存/加载

**说明**

- 支持对多级缓存模式下的稀疏表数据（Embedding、稀疏表优化器参数）进行保存/加载。
- 全量保存/加载：
  - 支持对所有稀疏表数据进行全量保存/加载。
- 增量保存/加载：
  - 支持对增量（基于上一次全量/增量保存）稀疏表数据进行增量保存。
  - 支持基于增量保存的稀疏表数据进行加载。
- 全量、增量保存/加载均支持差异卡加载功能（比如使用8卡训练并保存稀疏表数据，后续使用4卡加载稀疏表数据）。
- 支持EC模式和EBC模式。

**约束**

- 仅支持保存稀疏表相关数据，不支持保存/加载模型Dense部分，Dense部分可通过PyTorch原生API处理。
- 仅支持基于本地文件系统进行保存/加载。
- 准入淘汰（基于时间和计数）场景：
  - 支持对准入淘汰相关数据的全量保存/加载，不支持增量保存/加载。
- 准入淘汰（基于展示点击和分数）场景：
  - 不支持对准入淘汰相关数据的全量、增量保存/加载。

##### 全量保存/加载

**代码示例**

完整代码示例请参见[保存/加载测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py)。

简化代码示例（仅展示和多级缓存EC模式的差异部分）：

```python
# 差异：导入Saver类和safe_makedirs函数
import os
import shutil

from torchrec_embcache.saver import Saver
from torchrec_embcache.utils import safe_makedirs

...

def train():
    ...

    # 7.创建pipeline
    ...

    is_train = True

    # 差异：8.[Pre-operation]: 加载稀疏表数据。Dense部分数据加载需自行处理。
    save_dir = os.path.abspath("save_dir")
    sparse_save_dir = os.path.join(save_dir, "sparse")
    if not os.path.exists(save_dir):
        safe_makedirs(save_dir)
    if not is_train:
        # 加载稀疏表数据
        saver: Saver = Saver(rank=rank)
        saver.load(ddp_model, sparse_save_dir)

    # 8.使用pipeline进行训练
    ...

    # 差异：9.训练后保存稀疏表数据。Dense部分数据保存需自行处理。
    if is_train:
        if os.path.exists(sparse_save_dir):
            shutil.rmtree(sparse_save_dir, ignore_errors=True)
        saver: Saver = Saver(rank=rank)
        # 若需要在训练过程中（即Dataset未迭代到末尾）进行保存，则需手动触发wait_pipeline_compute_swapinfo()方法
        pipeline.wait_pipeline_compute_swapinfo()
        # 调用save接口保存稀疏表数据
        saver.save(ddp_model, sparse_save_dir)
    
    ...
```

##### 增量保存/加载

**说明**

增量保存时，仅保存上一次全量/增量保存后，新增的特征ID和对应稀疏表数据。

增量场景下，加载数据需先加载base数据（全量保存的结果），再加载delta数据（增量保存的结果）。

- 加载base数据时会清空当前的稀疏表数据。
- 增量数据保存/加载可执行多次（传入不同的delta数据保存/加载路径）。

**增量保存/加载对比全量保存/加载接口参数差异**

- 创建embedding_config时，需传入`is_incremental=True`参数，开启增量保存/加载功能。
- 调用`Saver.save()`接口执行增量保存时，需传入`incremental=True`参数。
- 调用`Saver.load()`接口执行增量加载时，需传入`incremental=True`参数。

**代码示例**

完整代码示例请参见[保存/加载测试用例](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py)。

简化代码示例（仅展示和多级缓存EC模式的差异部分）：

```python
...
# 差异：导入Saver类和safe_makedirs函数
from torchrec_embcache.saver import Saver
from torchrec_embcache.utils import safe_makedirs

...

def train():
    ...

    # 2.创建稀疏表
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        emb_config = EmbCacheEmbeddingConfig(
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # 注意：多级缓存模式，不支持自定义初始化函数
            initializer_type=InitializerType.UNIFORM,  # embedding初始化方式通过initializer_type参数设置
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
            is_incremental=True,  # 差异：配置稀疏表支持增量保存
        )
        embedding_configs.append(emb_config)
    sparse_ebc: torch.nn.Module = EmbCacheEmbeddingCollection(
        embedding_configs,  # type: ignore
        world_size,
        batch_size,
        multi_hot_sizes=[1] * table_num,
        device=torch.device("meta"),
    )

    ...
    
    # 设置True则进行全量和增量保存，False则全量和增量加载。
    is_train = True

    # 差异：8.[Pre-operation]: 加载稀疏表数据
    saver: Saver = Saver(rank=rank)
    save_dir = os.path.abspath("save_dir")
    sparse_save_dir = os.path.join(save_dir, "sparse")
    sparse_save_dir_base = os.path.join(sparse_save_dir, "base")
    sparse_save_dir_delta = os.path.join(sparse_save_dir, "delta")
    if not os.path.exists(save_dir):
        safe_makedirs(save_dir)
    if not is_train:
        # 加载稀疏表数据
        saver.load(ddp_model, sparse_save_dir_base)  # 加载全量数据
        saver.load(ddp_model, sparse_save_dir_delta, incremental=True)  # 加载增量数据

    # 8.使用pipeline进行训练
    dataset_iterator = iter(data_loader)
    for step in range(BATCH_NUM):
        # 对稀疏表数据做全量保存
        if is_train and step == 20:
            if os.path.exists(sparse_save_dir):
                shutil.rmtree(sparse_save_dir, ignore_errors=True)
            # 若需要在训练过程中（即Dataset未迭代到末尾）进行保存，则需手动触发wait_pipeline_compute_swapinfo()方法
            pipeline.wait_pipeline_compute_swapinfo()
            # 调用save接口保存稀疏表数据
            saver.save(ddp_model, sparse_save_dir_base)

        # 前面创建pipeline时，设置了return_loss=True，所以pipeline.progress()会返回output和loss两个值
        output, loss = pipeline.progress(dataset_iterator)
        logging.info("rank: %d, step: %s, loss: %s, sparse output: %s", rank, step, loss, output["sparse"])

    # 差异：9.对稀疏表数据做增量保存
    if is_train:
        # 若需要在训练过程中（即Dataset未迭代到末尾）进行保存，则需手动触发wait_pipeline_compute_swapinfo()方法
        pipeline.wait_pipeline_compute_swapinfo()
        # 调用save接口保存稀疏表数据
        saver.save(ddp_model, sparse_save_dir_delta, incremental=True)

```

注：上述代码示例仅为展示使用方式，可能和实际使用场景存在差异。
