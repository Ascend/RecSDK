# Migration and Training

## Training Scenarios

**Building a Network with Rec SDK Torch**

Follow the steps in [Quick Start](./quick_start.md) to build and train a model.

**Migrating from Open-Source TorchRec**

Based on a TorchRec model network, replace the APIs according to the mapping in [Table 1](#table16435142101913).

**Table 1** API mapping
<a id="table16435142101913"></a>

|TorchRec API|Rec SDK Torch API|Description|
|--|--|--|
|EmbeddingBagConfig|HashEmbeddingBagConfig|Configure sparse tables.|
|EmbeddingBagCollection|HashEmbeddingBagCollection|Create sparse tables.|
|get_default_sharders|get_default_hybrid_sharders|Get table sharders.|
|TrainPipelineSparseDist|HybridTrainPipelineSparseDist|Create pipelines.|

API example:

- TorchRec example

    ```python
    import torch.distributed as dist
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import get_default_sharders
    class TestModel(torch.nn.Module):
        def __init__(self, *):
            # The API used by Rec SDK Torch is HashEmbeddingBagCollection.
            self.sparse_model = EmbeddingBagCollection(xx)
            self.dense_model = xx
        def forward(self, batch: Batch):
            # Sparse (self.ebc) forward call and dense forward call
            # Note: the model forward return value must place loss first and output second, to match the native TorchRec TrainPipelineSparseDist usage.
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
        Sharder creation
        ...
        #  The API used by Rec SDK Torch is get_default_hybrid_sharders.
        hybrid_sharder = get_default_sharders()
        ...
        Optimizer creation
        ...
        # The API used by Rec SDK Torch is HybridTrainPipelineSparseDist.
        pipeline = TrainPipelineSparseDist()
        for i in range(20):
            output = pipeline.progress(batched_iterator)
    ```

- Rec SDK Torch example

    ```python
    import torch.distributed as dist
    from hybrid_torchrec import HashEmbeddingBagCollection
    from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
    from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
    ...
    class TestModel(torch.nn.Module):
        def __init__(self, *):
            # The API used by native TorchRec is EmbeddingBagCollection.
            self.sparse_model = HashEmbeddingBagCollection(xx)
            self.dense_model = xx
        def forward(self, batch: Batch):
            # Sparse forward call and dense forward call
            # Note: the model forward return value must place loss first and output second, to match the native TorchRec TrainPipelineSparseDist usage.
            return loss, output
    def invoke_main():
        dist.init_process_group(backend="hccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("npu")

        # Create host connection for Rec SDK Torch.
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
        data_loader = DataLoader(
            dataset,
        )
        test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS, NUM_EMBEDS)
        ...
        Sharder creation
        ...
        # The API used by native TorchRec is get_default_sharders.
        hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)
        ...
        Optimizer creation
        ...
        # The API used by native TorchRec is TrainPipelineSparseDist.
        pipeline = HybridTrainPipelineSparseDist()
        for i in range(20):
            output = pipeline.progress(batched_iterator)
    ```

## Rec SDK Torch Migration Examples

Rec SDK Torch supports adaptation for open-source Torch recommendation models. This section describes the main changes needed to migrate the open-source DLRM model (DCNv2) to the Rec SDK Torch framework. For complete migration changes, see [README](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/dlrm/README.md) and review the code after applying the patch file.

The main migration change is to replace the native TorchRec APIs used by the open-source model, such as sparse table configuration and training pipelines, with the corresponding APIs in the Rec SDK Torch framework.

Before migrating, download the open-source model code and switch to the specified commit:

```bash
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99
```

The main changes are as follows:

1. Modify the distributed backend.

    Replace the code in lines 555 to 562 of `dlrm/torchrec_dlrm/dlrm_main.py` with the following:

    ```python
        if torch.cuda.is_available():
            device: torch.device = torch.device(f"cuda:{rank}")
            backend = "nccl"
            torch.cuda.set_device(device)
        elif torch_npu.npu.is_available():
            device: torch.device = torch.device(f"npu:{rank}")
            backend = "hccl"
            torch_npu.npu.set_device(device)
        else:
            device: torch.device = torch.device("cpu")
            backend = "gloo"
    ```

2. Modify sparse table configuration.

    Replace the code in lines 589 to 601 of `dlrm/torchrec_dlrm/dlrm_main.py` with the following:

    ```python
        if with_embcache:
            if use_ec:
                from torchrec_embcache.distributed.configs import EmbCacheEmbeddingConfig, InitializerType
                ec_configs = [
                    EmbCacheEmbeddingConfig(
                        name=f"t_{feature_name}",
                        embedding_dim=args.embedding_dim,
                        num_embeddings=(
                            none_throws(args.num_embeddings_per_feature)[feature_idx]
                            if args.num_embeddings is None
                            else args.num_embeddings
                        ),
                        feature_names=[feature_name],
                        # Initialize the weight limit to zero tensor.
                        weight_init_mean=0.0,
                        weight_init_stddev=0.01,
                        initializer_type=InitializerType.TRUNCATED_NORMAL
                    )
                    for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
                ]
            else:
                from torchrec_embcache.distributed.configs import EmbCacheEmbeddingBagConfig, InitializerType
                eb_configs = [
                    EmbCacheEmbeddingBagConfig(
                        name=f"t_{feature_name}",
                        embedding_dim=args.embedding_dim,
                        num_embeddings=(
                            none_throws(args.num_embeddings_per_feature)[feature_idx]
                            if args.num_embeddings is None
                            else args.num_embeddings
                        ),
                        feature_names=[feature_name],
                        # Initialize the weight limit to zero tensor.
                        weight_init_mean=0.0,
                        weight_init_stddev=0.01,
                        initializer_type=InitializerType.TRUNCATED_NORMAL
                    )
                    for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
                ]
        else:
            eb_configs = [
                EmbeddingBagConfig(
                    name=f"t_{feature_name}",
                    embedding_dim=args.embedding_dim,
                    num_embeddings=(
                        none_throws(args.num_embeddings_per_feature)[feature_idx]
                        if args.num_embeddings is None
                        else args.num_embeddings
                    ),
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
            ]
    ```

3. Modify the embedding collection implementation.

    Replace the code in lines 606 to 644 of `dlrm/torchrec_dlrm/dlrm_main.py` with the following:

    ```python
        if args.interaction_type == InteractionType.ORIGINAL:
            from hybrid_torchrec import HashEmbeddingBagCollection
            dlrm_model = DLRM(
                embedding_bag_collection=HashEmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                dense_device=device,
            )
        elif args.interaction_type == InteractionType.DCN:
            if with_hybrid_torchrec:
                from hybrid_torchrec import HashEmbeddingBagCollection
                dlrm_model = DLRM_DCN(
                    embedding_bag_collection=HashEmbeddingBagCollection(
                        tables=eb_configs, device=torch.device("meta")
                    ),
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
            elif with_embcache:
                if use_ec:
                    from torchrec_embcache.distributed.embedding import EmbCacheEmbeddingCollection
                    dlrm_model = DLRM_DCN_EC(
                        embedding_collection=EmbCacheEmbeddingCollection(
                            tables=ec_configs,
                            batch_size=args.batch_size,
                            multi_hot_sizes=args.multi_hot_sizes,
                            world_size=dist.get_world_size(),
                            device=torch.device("meta"),
                        ),
                        multi_hot_sizes=args.multi_hot_sizes,
                        dense_in_features=len(DEFAULT_INT_NAMES),
                        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                        over_arch_layer_sizes=args.over_arch_layer_sizes,
                        dcn_num_layers=args.dcn_num_layers,
                        dcn_low_rank_dim=args.dcn_low_rank_dim,
                        dense_device=device,
                    )
                else:
                    from torchrec_embcache.distributed.embedding_bag import EmbCacheEmbeddingBagCollection
                    dlrm_model = DLRM_DCN(
                        embedding_bag_collection=EmbCacheEmbeddingBagCollection(
                            tables=eb_configs,
                            batch_size=args.batch_size,
                            multi_hot_sizes=args.multi_hot_sizes,
                            world_size=dist.get_world_size(),
                            device=torch.device("meta"),
                        ),
                        dense_in_features=len(DEFAULT_INT_NAMES),
                        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                        over_arch_layer_sizes=args.over_arch_layer_sizes,
                        dcn_num_layers=args.dcn_num_layers,
                        dcn_low_rank_dim=args.dcn_low_rank_dim,
                        dense_device=device,
                    )
            else:
                dlrm_model = DLRM_DCN(
                    embedding_bag_collection=EmbeddingBagCollection(
                        tables=eb_configs, device=torch.device("meta")
                    ),
                    dense_in_features=len(DEFAULT_INT_NAMES),
                    dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                    over_arch_layer_sizes=args.over_arch_layer_sizes,
                    dcn_num_layers=args.dcn_num_layers,
                    dcn_low_rank_dim=args.dcn_low_rank_dim,
                    dense_device=device,
                )
        elif args.interaction_type == InteractionType.PROJECTION:
            from hybrid_torchrec import HashEmbeddingBagCollection
            dlrm_model = DLRM_Projection(
                embedding_bag_collection=HashEmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
                interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
                dense_device=device,
            )
        else:
            raise ValueError(
                "Unknown interaction option set. Should be original, dcn, or projection."
            )
    ```

    In the preceding code, `DLRM_DCN_EC` is the new EC-version model definition. The model is defined in the new file `dlrm/torchrec_dlrm/ec_dcnv2.py`, and you can see the detailed code in that file after applying the patch.

4. Modify the sharding plan and create the distributed model.

    Replace the code in lines 622 to 675 of `dlrm/torchrec_dlrm/dlrm_main.py` with the following:

    ```python
        constraints = {
            f"t_{feature_name}": ParameterConstraints(
                sharding_types=["row_wise"],
                compute_kernels=["fused"]
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=dist.get_world_size(),
                compute_device=device.type,
            ),
            batch_size=args.batch_size,
            constraints=constraints,
        )

        if with_hybrid_torchrec:
            from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
            host_gp = dist.new_group(backend='gloo')
            host_env = ShardingEnv(world_size=dist.get_world_size(), rank=rank, pg=host_gp)
            sharders = get_default_hybrid_sharders(host_env)
        elif with_embcache:
            if use_ec:
                from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingCollectionSharder
                cpu_device = torch.device("cpu")
                cpu_pg = dist.new_group(backend="gloo")
                cpu_env = ShardingEnv.from_process_group(cpu_pg)
                embcache_sharder = EmbCacheEmbeddingCollectionSharder(
                    cpu_device=cpu_device,
                    cpu_env=cpu_env,
                    npu_device=device,
                    npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
                )
                sharders = [embcache_sharder]
            else:
                from torchrec_embcache.distributed.sharding.embedding_sharder import EmbCacheEmbeddingBagCollectionSharder
                cpu_device = torch.device("cpu")
                cpu_pg = dist.new_group(backend="gloo")
                cpu_env = ShardingEnv.from_process_group(cpu_pg)
                embcache_sharder = EmbCacheEmbeddingBagCollectionSharder(
                    cpu_device=cpu_device,
                    cpu_env=cpu_env,
                    npu_device=device,
                    npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),
                )
                sharders = [embcache_sharder]
        else:
            sharders = get_default_sharders()


        plan = planner.collective_plan(
            train_model, sharders, dist.GroupMember.WORLD
        )
        if rank == 0:
            logging.info("plan:%s", plan)
        model = DistributedModelParallel(
            module=train_model,
            sharders=sharders,
            device=device,
            plan=plan,
        )
    ```

5. Modify the pipeline creation method.

    Replace the code in lines 477 to 480 of `dlrm/torchrec_dlrm/dlrm_main.py` with the following:

    ```python
        if with_hybrid_torchrec:
            from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
            pipeline = HybridTrainPipelineSparseDist(
                model, optimizer, device, execute_all_batches=True
            )
        elif with_embcache:
            from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
            cpu_device: torch.device = torch.device("cpu")
            pipeline = EmbCacheTrainPipelineSparseDist(
                model, optimizer, cpu_device=cpu_device, npu_device=device, execute_all_batches=True
            )
        else:
            # Native TorchRec
            pipeline = TrainPipelineSparseDist(
                model, optimizer, device, execute_all_batches=True
            )
    ```

## Functions and Features

Rec SDK Torch supports two training modes: full-device memory mode and multi-level cache mode.

In full-device memory mode, all sparse table embedding data is stored in device memory during training.

In multi-level cache mode, embeddings are stored using a combination of device memory and DDR (host memory) during training.

This section describes the related features and code examples for full-device memory mode and multi-level cache mode.

> [!NOTE]
>
> The **Constraints** sections in this document only describe constraints for the main usage scenarios. For details, see the corresponding API documentation.

### Training in Full-Device Memory Mode

**API differences between full-device memory mode and the native TorchRec framework**

The APIs used by Rec SDK Torch full-device memory mode differ from those used by the native TorchRec framework for configuring sparse tables, creating sparse tables, sharding sparse tables, and creating pipelines.

|Scenario/API|Native TorchRec Framework|Rec SDK Torch Full-Device Memory Mode|
|--|--|--|
|Sparse table configuration|EmbeddingBagConfig|HashEmbeddingBagConfig|
|Sparse table creation|EmbeddingBagCollection|HashEmbeddingBagCollection|
|Sparse table sharder|get_default_sharders()|get_default_hybrid_sharders()|
|Pipeline creation|TrainPipelineSparseDist|HybridTrainPipelineSparseDist|

**Constraints**

- Only the EBC mode is supported.
- Both pipeline mode and non-pipeline mode (directly calls the sharded sparse table forward pass) are supported for training.
- Sparse table data cannot be saved or loaded.

**Full-device memory mode test case**

For the complete full-device memory mode test case, see [README](../../../../training/torch_rec_v1/hybrid_torchrec/test/st/README.md).

#### Basic Usage

The training process is the same as the native TorchRec pipeline mode. The only differences are the APIs used to create sparse tables, shard sparse tables, and create the pipeline.

**Code example**

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
        # Define the dense layer.
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Define the dense layer implementation.
        return self.linear(x)


class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        # Sparse forward pass
        sparse_output: EmbeddingBagCollectionAwaitable = self.sparse_model(batch.sparse_features)
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output[feat_name])  # type: ignore
        # Merge the embeddings of all sparse features.
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # Dense forward pass
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # Calculate loss.
        loss: torch.Tensor = self.loss_fn(dense_output, batch.labels)

        # Define the output content as required.
        output: dict[str, torch.Tensor] = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output

        # Return the forward output.
        # Note: The forward pass must return the loss and output values, with loss first and output second. This usage matches the native TorchRec TrainPipelineSparseDist usage.
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
    # 0. Set up the distributed environment.
    set_distribute_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    npu_device = torch.device("npu")

    embedding_dims: list[int] = [64, 16, 32]
    num_embeddings: list[int] = [400, 4000, 400]
    table_num: int = len(num_embeddings)

    # 1. Create the dataset.

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

    # 2. Create the sparse table.
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

    # 3. Create the model: Wrap the sparse table and dense part into one model.
    dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), DENSE_OUTPUT_DIM)
    test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

    # 4. Define the sparse table optimizer.
    embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
    optimizer_kwargs = {"lr": 0.01, "eps": 0.1}
    apply_optimizer_in_backward(
        embedding_optimizer_class,
        test_model.sparse_model.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    # 5. Shard the sparse table.
    host_gp = dist.new_group(backend="gloo")
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    sharders = get_default_hybrid_sharders(host_env=host_env)
    constraints = {
        # Use the row_wise sharding type and fused for compute_kernels.
        table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
        for table_name in table_names
    }
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)
    #   Here, sharding matches the corresponding class type based on the sharders argument: EmbeddingBagCollection or HashEmbeddingBagCollection.
    #   It shards only sparse table parameters and does not shard dense parameters
    ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)

    # 6. Combine the optimizers.
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])

    # 7. Create the pipeline.
    pipeline = HybridTrainPipelineSparseDist(
        ddp_model, optimizer, npu_device, return_loss=True, execute_all_batches=True
    )

    # 8. Train with the pipeline.
    dataset_iterator = iter(data_loader)
    for step in range(BATCH_NUM):
        # When the pipeline is created, return_loss is set to True, so pipeline.progress() returns two values: output and loss.
        # If return_loss is set to False when the pipeline is created, only output is returned.
        output, loss = pipeline.progress(dataset_iterator)
        logging.info("rank: %d, step: %s, loss: %s, sparse output: %s", rank, step, loss, output["sparse"])


if __name__ == "__main__":
    train()
    # How to use the script:
    #   1. Write the script into main.py and copy RecSDK/training/torch_rec_v1/hybrid_torchrec/test/st/dataset.py to the same directory as the .py file.
    #   2. Start single-card training: WORLD_SIZE=1 RANK=0 python3 main.py
    #   3. Start multi-card training (two cards): torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=2 main.py
```

For more demo examples, see [Rec SDK Torch Little Demo examples](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_examples/little_demo).

#### Sparse Table Data Parallelism (DP)

**Constraints**

- The DP mode supports only a single sparse table.

**Comparison with the non-DP mode**

Compared with [the non-DP mode (basic usage)](#basic-usage), the DP mode mainly differs in how it stores sparse table parameters during training.

In the DP mode, the sparse table is no longer sharded across different devices. Instead, each device holds a complete copy of the sparse table parameters. During backpropagation, the gradients from all devices are aggregated before a unified update.

**Code example**

For the complete DP mode code example, see [DP mode test case](../../../../training/torch_rec_v1/hybrid_torchrec/test/st/test_hybrid_hash_embeddingbag_dp.py).

Simplified code example, showing only the differences from basic usage:

```python
    # Difference: The DP mode supports only a single table and does not support multi-table mode.
    embedding_dims = [64]
    num_embeddings = [400]

    ...

    # 5. Shard the sparse table.
    host_gp = dist.new_group(backend="gloo")
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    sharders = get_default_hybrid_sharders(host_env=host_env)
    constraints = {
        # Difference: The sharding type is data_parallel, and dense is used for compute_kernels.
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

### Training in Multi-Level Cache Mode

**Functional differences between the multi-level cache mode and full-device memory mode**

- Multi-level cache mode stores sparse tables with device memory as a cache for the embedding parameters needed during training, while DDR stores and manages the full set of embedding parameters.
- The EC mode and EBC mode are supported.
- Sparse table parameters can be saved and loaded.
- Only row-wise sharding for sparse tables is supported.
- Only pipeline mode training is supported.
  - During pipeline mode training, data is swapped in (H2D) and out (D2H) before `sparse_model.forward()` is called, so the training data for the current batch stays in device memory.

**API differences between the multi-level cache mode and full-device memory mode**

The APIs used by multi-level cache mode differ from those used by full-device memory mode for configuring sparse tables, creating sparse tables, sharding sparse tables, and creating pipelines.

|Scenario/API|Full-Device Memory Mode|Multi-Level Cache EC Mode|Multi-Level Cache EBC Mode|
|--|--|--|--|
|Sparse table configuration|HashEmbeddingBagConfig|EmbCacheEmbeddingConfig|EmbCacheEmbeddingBagConfig|
|Sparse table creation|HashEmbeddingBagCollection|EmbCacheEmbeddingCollection|EmbCacheEmbeddingBagCollection|
|Sparse table sharder|get_default_hybrid_sharders()|[EmbCacheEmbeddingCollectionSharder()]|[EmbCacheEmbeddingBagCollectionSharder()]|
|Pipeline creation|HybridTrainPipelineSparseDist|EmbCacheTrainPipelineSparseDist|EmbCacheTrainPipelineSparseDist|

**Multi-level cache mode test case**

For the complete multi-level cache mode test case, see [README](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/README.md).

#### Basic Usage

##### Multi-Level Cache EC Mode

**Code example**

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
        # Define the dense layer.
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Define the dense layer implementation.
        return self.linear(x)


class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        # Sparse forward pass
        sparse_output: EmbeddingCollectionAwaitable = self.sparse_model(batch.sparse_features)
        sparse_output_dict: dict[str, torchrec.JaggedTensor] = sparse_output.wait()  # type: ignore
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output_dict[feat_name].values())
        # Concatenate the embeddings of the sparse tables.
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # Dense forward pass
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # Calculate loss.
        loss = self.loss_fn(dense_output, batch.labels)

        # Define the output content as required.
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # Return the forward output.
        # Note: The forward pass must return the loss and output values, with loss first and output second. This usage matches the native TorchRec TrainPipelineSparseDist usage.
        return loss, output


def set_distribute_env():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(rank)  # type: ignore
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(backend="hccl")


def train():
    # 0. Set up the distributed environment.
    set_distribute_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    npu_device = torch.device("npu")

    embedding_dims: list[int] = [64, 64, 64]  # TorchRec EC mode requires all tables to use the same embedding dim.
    num_embeddings: list[int] = [400, 4000, 400]
    table_num: int = len(num_embeddings)

    # 1. Create the dataset.
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

    # 2. Create the sparse table.
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        emb_config = EmbCacheEmbeddingConfig(
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
            initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
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

    # 3. Create the model: Wrap the sparse table and dense part into one model.
    dense_model: torch.nn.Module = DenseModel(sum(embedding_dims), OUTPUT_SIZE)
    test_model: torch.nn.Module = TestModel(sparse_ebc, dense_model, feat_names)

    # 4. Define the sparse table optimizer.
    embedding_optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adagrad
    optimizer_kwargs = {"lr": 0.01, "eps": 0.1}
    apply_optimizer_in_backward(
        embedding_optimizer_class,
        test_model.sparse_model.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    # 5. Shard the sparse table.
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
        # Use the row_wise sharding type and fused for compute_kernels.
        table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
        for table_name in table_names
    }
    planner: EmbeddingShardingPlanner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan: EmbeddingShardingPlan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)  # type: ignore
    #   Here, sharding matches the corresponding class type based on the sharders argument: EmbCacheEmbeddingCollection.
    #   It shards only sparse table parameters and does not shard dense parameters
    ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)  # type: ignore

    # 6. Combine the optimizers.
    dense_optimizer: KeyedOptimizerWrapper = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])

    # 7. Create the pipeline.
    pipeline: EmbCacheTrainPipelineSparseDist = EmbCacheTrainPipelineSparseDist(
        ddp_model,
        optimizer,
        cpu_device=cpu_device,
        npu_device=npu_device,
        return_loss=True,
    )

    # 8. Train with the pipeline.
    dataset_iterator = iter(data_loader)
    for step in range(BATCH_NUM):
        # When the pipeline is created, return_loss is set to True, so pipeline.progress() returns two values: output and loss.
        output, loss = pipeline.progress(dataset_iterator)
        logging.info("rank: %d, step: %s, loss: %s, sparse output: %s", rank, step, loss, output["sparse"])


if __name__ == "__main__":
    train()
    # How to use the script:
    #   1. Write the script into main.py and copy RecSDK/training/torch_rec_v1/torchrec_embcache/tests/acc_test/dataset.py to the same directory as the .py file.
    #   2. Start single-card training: WORLD_SIZE=1 RANK=0 python3 main.py
    #   3. Start multi-card training (two cards): torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=2 main.py
```

For more code examples, see [Multi-Level Cache EC Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_pipeline.py).

##### Multi-Level Cache EBC Mode

**Constraints**

- The Multi-level cache EBC mode supports only multiple sparse tables that use the same embedding dim.

**Differences from the multi-level cache EC mode**

Compared with the EC mode, the EBC mode uses different APIs to create sparse tables and sharders. It also differs in how the integrated model handles sparse table lookup results in `forward()`.

**Code example**

For the complete code example, see [Multi-Level Cache EBC Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_cache_pipeline.py).

Simplified code example, showing only the differences from the multi-level cache EC mode:

```python
...

# Difference: Import the APIs used by multi-level cache EBC mode.
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
        # Sparse forward pass
        sparse_output: EmbeddingBagCollectionAwaitable = self.sparse_model(batch.sparse_features)
        # Difference: The way embedding forward lookup results are parsed differs from the multi-level cache EC mode.
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output[feat_name])  # type: ignore
        # Merge the embeddings of all sparse features.
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # Dense forward pass
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # Calculate loss.
        loss: torch.Tensor = self.loss_fn(dense_output, batch.labels)

        # Define the output content as required.
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # Return the forward output.
        # Note: The forward pass must return the loss and output values, with loss first and output second. This usage matches the native TorchRec TrainPipelineSparseDist usage.
        return loss, output

...

def train():
    ...

    # 2. Create the sparse table.
    embedding_configs: list[EmbCacheEmbeddingBagConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        emb_config = EmbCacheEmbeddingBagConfig(  # Difference: The table config and collection API in the EBC mode differ from the EC mode.
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
            initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
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

    # 5. Shard the sparse table.
    cpu_pg = dist.new_group(backend="gloo")
    cpu_env = ShardingEnv.from_process_group(cpu_pg)  # pyright: ignore[reportArgumentType]
    cpu_device = torch.device("cpu")
    sharders: list[EmbCacheEmbeddingBagCollectionSharder] = [  # Difference: The sharder in the EBC mode differs from that in the EC mode.
        EmbCacheEmbeddingBagCollectionSharder(
            cpu_device=cpu_device,
            cpu_env=cpu_env,
            npu_device=npu_device,
            npu_env=ShardingEnv.from_process_group(dist.GroupMember.WORLD),  # type: ignore
        ),
    ]
    constraints: dict[str, ParameterConstraints] = {
        # Use the row_wise sharding type and fused for compute_kernels.
        table_name: ParameterConstraints(sharding_types=["row_wise"], compute_kernels=["fused"])
        for table_name in table_names
    }
    planner: EmbeddingShardingPlanner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )
    plan: EmbeddingShardingPlan = planner.collective_plan(test_model, sharders, dist.GroupMember.WORLD)  # type: ignore
    #   Here, sharding matches the corresponding class type based on the sharders argument: EmbCacheEmbeddingBagCollection.
    #   It shards only sparse table parameters and does not shard dense parameters
    ddp_model = DistributedModelParallel(test_model, device=npu_device, plan=plan, sharders=sharders)  # type: ignore

    ...

```

#### Admission and Eviction

The multi-level cache mode supports two admission and eviction policies: **based on time and count** and **based on impressions, clicks, and score**.

> [!NOTE]
>
> **Only one of the admission and eviction policies can be used at a time. Mixing both policies is not supported.**

##### Admission and Eviction Based on Time and Count

**Description**

- Admission
  - Feature ID occurrences are counted. Feature IDs whose counts exceed the admission threshold are admitted, and feature IDs whose counts do not exceed the threshold are not admitted. The embedding lookup result for a non-admitted feature ID is an all-0.0 vector. You can configure the value in the admission policy.
  - AllToAll communication is performed for the occurrence counts of feature IDs to ensure that all processes can obtain the counts for every feature ID.
- Eviction
  - The difference between the timestamp of a feature ID and the latest timestamp in the embedding table containing that feature ID is calculated. The latest timestamp is the maximum timestamp across all feature IDs in the sparse table. Feature IDs and embeddings whose difference exceeds the eviction threshold are evicted.
  - AllToAll communication is not performed on the feature ID timestamps. Each process tracks only the timestamps of the feature IDs it reads.

**Constraints**

- Only the multi-level cache EC mode is supported.
- Admission and eviction can be enabled separately or simultaneously.

**Code differences from the multi-level cache EC mode**

- Admission scenario
  - When creating `EmbCacheEmbeddingConfig`, specify the `admit_and_evict_config` parameter and configure the admission policy parameters in it.
- Eviction scenario
  - The data in the constructed `Dataset` must include the timestamp data that corresponds to each feature ID.
  - When creating `EmbCacheEmbeddingConfig`, specify the `admit_and_evict_config` parameter and configure the eviction policy parameters in it.
  - When creating the pipeline, specify the `evict_step_interval` parameter to configure the step interval that triggers eviction.

**Code example**

For the complete code example, see [Admission and Eviction Based on Time and Count Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_feature_filter.py).

Simplified code example, showing only the differences from the multi-level cache EC mode:

```python
    ...

    # 1. Create the dataset.
    batch_size: int = 128
    # Difference: In the eviction scenario, create a dataset with feature ID timestamps. For the detailed implementation, see the RandomRecDataset definition in dataset.py.
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

    # 2. Create the sparse table.
    evict_step_interval = 20  # Eviction step. Evict once every 20 batches
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        # Difference: Set the admission and eviction configuration parameters and pass them to emb_config.
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
            # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
            initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
            admit_and_evict_config=admit_and_evict_config,  # Difference: Pass the admission and eviction configuration parameters.
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

    # 7. Create the pipeline.
    pipeline: EmbCacheTrainPipelineSparseDist = EmbCacheTrainPipelineSparseDist(
        ddp_model,
        optimizer,
        cpu_device=cpu_device,
        npu_device=npu_device,
        return_loss=True,
        evict_step_interval=evict_step_interval,  # Difference: In the eviction scenario, pass the eviction step interval parameter to the pipeline.
    )

    ...

```

##### Admission and Eviction Based on Impressions, Clicks, and Score

**Description**

- Admission
  - The admission score from impressions and clicks is calculated with weights: `score = alpha * impressions + beta * clicks` as configured by `ShowClickParams`.
  - When `showclick_params.admit_threshold > 0`, admission is enabled. Feature IDs whose scores are below the admission threshold are treated as not admitted. The embedding lookup result for a feature ID is an all-0.0 vector. You can configure the value in the admission policy.
  - AllToAll communication is performed for the occurrence counts of feature IDs to ensure that all processes can obtain the counts for every feature ID.
- Eviction
  - Maintain the eviction score. At each step, update it with `new_score = (oldScore + alpha * impressions + beta * clicks) × score_decay`, where `score_decay = 1` means no decay.
  - When the `evict_percentage` value is greater than zero, eviction is enabled. Embeddings are evicted by taking roughly the `evict_percentage` proportion with the lowest scores.

**Constraints**

- Only the multi-level cache EC mode is supported.
- Admission and eviction can be enabled separately or simultaneously.

**Code example**

For the complete code example, see [Admission and Eviction Based on Time and Count Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_show_click.py).

Simplified code example, showing only the differences from the multi-level cache EC mode:

```python
# Difference: Import ShowClickBatch.
from dataset import RandomRecDataset, Batch, ShowClickBatch

...

class TestModel(torch.nn.Module):

    def __init__(self, sparse_model: torch.nn.Module, dense_model: torch.nn.Module, feat_names: list[str]):
        super().__init__()
        self.sparse_model: torch.nn.Module = sparse_model
        self.dense_model: torch.nn.Module = dense_model
        self.feat_names: list[str] = feat_names
        self.loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch: ShowClickBatch):  # Difference: The typing type is ShowClickBatch.
        # Sparse forward pass
        sparse_output: EmbeddingCollectionAwaitable = self.sparse_model(batch.sparse_features)
        sparse_output_dict: dict[str, torchrec.JaggedTensor] = sparse_output.wait()  # type: ignore
        feat_embeddings: list[torch.Tensor] = list()
        for feat_name in self.feat_names:
            feat_embeddings.append(sparse_output_dict[feat_name].values())
        # Concatenate the embeddings of the sparse tables.
        embeddings: torch.Tensor = torch.concat(feat_embeddings, dim=-1)

        # Dense forward pass
        dense_output: torch.Tensor = self.dense_model(embeddings)

        # Calculate loss.
        loss = self.loss_fn(dense_output, batch.click_labels)  # Difference: In ShowClickBatch, the labels field is named click_labels.

        # Define the output content as required.
        output = dict()
        output["sparse"] = embeddings
        output["dense"] = dense_output
        # Return the forward output.
        # Note: The forward pass must return the loss and output values, with loss first and output second. This usage matches the native TorchRec TrainPipelineSparseDist usage.
        return loss, output


def train():
    ...

    # 1. Create the dataset.
    batch_size: int = 128
    # Difference: Create a dataset that includes click_labels. For the detailed implementation, see the RandomRecDataset definition in dataset.py.
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

    # 2. Create the sparse table.
    evict_step_interval = 20  # Eviction step. Evict once every 20 batches
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    showclick_params = ShowClickParams(alpha=1, beta=1, admit_threshold=0.1, evict_percentage=0.1, score_decay=0.9)
    for i in range(table_num):
        # Difference: Set the admission and eviction configuration parameters based on show and click events and pass them to emb_config.
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
            # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
            initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
            admit_and_evict_config=admit_and_evict_config,  # Difference: Pass the admission and eviction configuration parameters.
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

    # 7. Create the pipeline.
    pipeline: EmbCacheTrainPipelineSparseDist = EmbCacheTrainPipelineSparseDist(
        ddp_model,
        optimizer,
        cpu_device=cpu_device,
        npu_device=npu_device,
        return_loss=True,
        evict_step_interval=evict_step_interval,  # Difference: In the eviction scenario, pass the eviction step interval parameter to the pipeline.
    )

    ...
```

#### Gradient Accumulation

**Description**

The core idea of gradient accumulation is to split a large batch that would otherwise be calculated all at once into multiple micro-batches for training, compute gradients for each training step, and accumulate them. Once the accumulation reaches a certain number of steps, use the accumulated gradients to update the parameters once.

The equivalent batch size for gradient accumulation is `micro-batch_size * accumulation_steps`.

In a gradient accumulation scenario, the batch size is reduced, so training uses less device memory. This is suitable for cases where device memory is limited but large batch size training is required.

**Constraints**

- You must use an optimizer from Rec SDK Torch that supports gradient accumulation. See [optimizer_class parameter description](./api/optimizers_apis.md#section888634319218) for supported optimizers.
- Only the EC mode is supported.

**API differences from the multi-level cache EC mode**

The main difference from multi-level cache EC mode is that you must **use an optimizer that supports gradient accumulation** and **set the accumulation step parameter** in the optimizer initialization arguments.

**Code example**

For the complete code example, see [Gradient Accumulation Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_aggregation.py).

Simplified code example, showing only the differences from the multi-level cache EC mode:

```python
...

# Difference: Import optimizers that support gradient accumulation.
from torchrec.optim import AccumulateAdagrad, AccumulateAdam, AccumulateSGD

...

def train():
    ...

    # 4. Define the sparse table optimizer.
    # Difference: Use an optimizer_class that supports gradient accumulation.
    embedding_optimizer_class: type[torch.optim.Optimizer] = AccumulateAdagrad
    # Difference: Set the gradient accumulation parameters. use_accumulate=True enables gradient accumulation, and accumulate_step is the accumulation step count.
    optimizer_kwargs = {"lr": 0.01, "eps": 0.1, "use_accumulate": True, "accumulate_step": 10}
    apply_optimizer_in_backward(
        embedding_optimizer_class,
        test_model.sparse_model.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    ...
```

#### Saving and Loading

**Description**

- Sparse table data (embedding and sparse table optimizer parameters) can be saved and loaded in the multi-level cache mode.
- Full saving/loading:
  - All sparse table data can be saved or loaded.
- Incremental saving/loading:
  - Incremental sparse table data (based on the last full or incremental saving) can be saved.
  - Sparse table data that is incrementally saved can be loaded.
- Both full and incremental saving/loading support loading with a different number of cards. For example, you can train and save sparse table data on eight cards and later load it on four cards.
- The EC mode and EBC mode are supported.

**Constraints**

- Only sparse table-related data is supported for saving and loading. Saving and loading the dense part of the model is not supported. You can handle the dense part with native PyTorch APIs.
- Only saving and loading based on the local file system is supported.
- Admission and eviction based on time and count:
  - Full saving/loading of admission and eviction-related data is supported. Incremental saving/loading is not supported.
- Admission and eviction based on impressions, clicks, and score:
  - Full or incremental saving/loading of admission and eviction-related data is not supported.

##### Full Saving and Loading

**Code example**

For the complete code example, see [Saving/Loading Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py).

Simplified code example, showing only the differences from the multi-level cache EC mode:

```python
# Difference: Import the Saver class and the safe_makedirs function.
import os
import shutil

from torchrec_embcache.saver import Saver
from torchrec_embcache.utils import safe_makedirs

...

def train():
    ...

    # 7. Create the pipeline.
    ...

    is_train = True

    # Difference: 8. [Pre-operation]: Load sparse table data. Load the dense part separately.
    save_dir = os.path.abspath("save_dir")
    sparse_save_dir = os.path.join(save_dir, "sparse")
    if not os.path.exists(save_dir):
        safe_makedirs(save_dir)
    if not is_train:
        # Load sparse table data.
        saver: Saver = Saver(rank=rank)
        saver.load(ddp_model, sparse_save_dir)

    # 8. Train with the pipeline.
    ...

    # Difference: 9. Save sparse table data after training. Save the dense part separately.
    if is_train:
        if os.path.exists(sparse_save_dir):
            shutil.rmtree(sparse_save_dir, ignore_errors=True)
        saver: Saver = Saver(rank=rank)
        # If you need to save during training, that is, before the Dataset reaches the end, you must manually call wait_pipeline_compute_swapinfo().
        pipeline.wait_pipeline_compute_swapinfo()
        # Call the save API to save sparse table data.
        saver.save(ddp_model, sparse_save_dir)

    ...
```

##### Incremental Saving and Loading

**Description**

During incremental save, only the newly added feature IDs and their corresponding sparse table data after the previous full or incremental saving are saved.

In the incremental scenario, you must load the base data first, which is the result of the full saving, and then load the delta data, which is the result of the incremental saving.

- Loading the base data clears the current sparse table data.
- You can save and load incremental data multiple times by passing different delta saving or loading paths.

**Parameter differences between incremental saving/loading and full saving/loading**

- When creating `embedding_config`, pass `is_incremental=True` to enable incremental saving/loading.
- When calling `Saver.save()` for incremental saving, pass `incremental=True`.
- When calling `Saver.load()` for incremental loading, pass `incremental=True`.

**Code example**

For the complete code example, see [Saving/Loading Test Case](../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py).

Simplified code example, showing only the differences from the multi-level cache EC mode:

```python
...
# Difference: Import the Saver class and the safe_makedirs function.
from torchrec_embcache.saver import Saver
from torchrec_embcache.utils import safe_makedirs

...

def train():
    ...

    # 2. Create the sparse table.
    embedding_configs: list[EmbCacheEmbeddingConfig] = []
    table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
    feat_names: list[str] = [f"feat{i}" for i in range(len(num_embeddings))]
    for i in range(table_num):
        emb_config = EmbCacheEmbeddingConfig(
            name=table_names[i],
            embedding_dim=embedding_dims[i],
            num_embeddings=num_embeddings[i],
            feature_names=[feat_names[i]],
            # init_fn=weight_init,  # Note: The multi-level cache mode does not support custom initialization functions.
            initializer_type=InitializerType.UNIFORM,  # Set the embedding initialization method with the initializer_type parameter.
            weight_init_mean=0.0,
            weight_init_stddev=0.05,
            is_incremental=True,  # Difference: Configure the sparse table to support incremental saving.
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

    # Set this to True to perform full and incremental saving. Set it to False to perform full and incremental loading.
    is_train = True

    # Difference: 8. [Pre-operation]: Load sparse table data.
    saver: Saver = Saver(rank=rank)
    save_dir = os.path.abspath("save_dir")
    sparse_save_dir = os.path.join(save_dir, "sparse")
    sparse_save_dir_base = os.path.join(sparse_save_dir, "base")
    sparse_save_dir_delta = os.path.join(sparse_save_dir, "delta")
    if not os.path.exists(save_dir):
        safe_makedirs(save_dir)
    if not is_train:
        # Load sparse table data.
        saver.load(ddp_model, sparse_save_dir_base)  # Load full data.
        saver.load(ddp_model, sparse_save_dir_delta, incremental=True)  # Load incremental data.

    # 8. Train with the pipeline.
    dataset_iterator = iter(data_loader)
    for step in range(BATCH_NUM):
        # Perform a full saving of the sparse table data.
        if is_train and step == 20:
            if os.path.exists(sparse_save_dir):
                shutil.rmtree(sparse_save_dir, ignore_errors=True)
            # If you need to save during training, that is, before the Dataset reaches the end, you must manually call wait_pipeline_compute_swapinfo().
            pipeline.wait_pipeline_compute_swapinfo()
            # Call the save API to save sparse table data.
            saver.save(ddp_model, sparse_save_dir_base)

        # When the pipeline is created, return_loss is set to True, so pipeline.progress() returns two values: output and loss.
        output, loss = pipeline.progress(dataset_iterator)
        logging.info("rank: %d, step: %s, loss: %s, sparse output: %s", rank, step, loss, output["sparse"])

    # Difference: 9. Perform incremental saving of sparse table data.
    if is_train:
        # If you need to save during training, that is, before the Dataset reaches the end, you must manually call wait_pipeline_compute_swapinfo().
        pipeline.wait_pipeline_compute_swapinfo()
        # Call the save API to save sparse table data.
        saver.save(ddp_model, sparse_save_dir_delta, incremental=True)

```

Note: The code examples are for reference only and may differ from actual usage scenarios.
