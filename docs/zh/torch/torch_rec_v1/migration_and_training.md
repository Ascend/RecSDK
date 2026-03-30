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

> [!NOTE]说明 
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
