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
|EmbeddingCollectionSharder|DynamicEmbeddingCollectionSharder|稀疏表分片器|
|EmbeddingShardingPlanner|DynamicEmbeddingShardingPlanner|分表计划生成器|
|EmbeddingEnumerator|DynamicEmbeddingEnumerator|分片选项枚举|
|ParameterConstraints|DynamicEmbParameterConstraints|分表约束|
|PyTorch保存/加载|DynamicEmbDump/DynamicEmbLoad|动态稀疏表保存与加载|

接口示例：

- TorchRec示例：

    ```bash
    from torchrec.distributed.embedding import EmbeddingCollection
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
    from torchrec.distributed.planner import EmbeddingShardingPlanner
    from torchrec.distributed.model_parallel import get_default_sharders

    class TestModel(torch.nn.Module):
        def __init__(self, *):
            self.ec = EmbeddingCollection(···)

        def forward(self, batch):
            pass

    def invoke_main():
        rank, world_size = get_distribute_env()
        device = torch.device("npu")
        dist.init_process_group(backend="hccl")

        dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
        data_loader = DataLoader(
            dataset,
        )
        test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS ,NUM_EMBEDS)
        ...

        sharder = get_default_sharders()
        planner = EmbeddingShardingPlanner(···)

        sharding_plan = planner.collective_plan(···)
        ...
        优化器创建
        ...
        pipeline = TrainPipelineSparseDist(test_model, optimizer, device)
        for i in range(20):
            pipeline.progress(batched_iterator)
    ```

- Rec SDK Torch示例：

    ```bash
    from torchrec.distributed.planner.types import Topology, ShardingType
    from dynamic_emb import (
        DynamicEmbeddingCollectionSharder,
        DynamicEmbeddingShardingPlanner,
        DynamicEmbTableOptions,
        DynamicEmbParameterConstraints,
        DynamicEmbDump,
        DynamicEmbLoad,
        EmbOptimType,
    )
    ...

    class TestModel(torch.nn.Module):
        def __init__(self, *):
            self.ec = EmbeddingCollection()
        def forward(self, *):
            pass
    def invoke_main():
        rank, world_size = get_distribute_env()
        device = torch.device("npu")
        dist.init_process_group(backend="hccl")

        dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
        data_loader = DataLoader(
            dataset,
        )
        model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS ,NUM_EMBEDS)

        table_options = DynamicEmbTableOptions(
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            evict_strategy=DynamicEmbEvictStrategy.LRU,
            optimizer_type=EmbOptimType.ADAM,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
                value=0.0,
            ),
        )

        eb_configs = [
            EmbeddingConfig(
                name=name,
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[feature_names],
                data_type=DataType.FP32,
            ),
            ···
        ]
        # 设置分表约束
        constraints = {}
        for eb_config in eb_configs:
            constraints[eb_config.name] = DynamicEmbParameterConstraints(···)

        # 创建分片器和规划器
        topology = Topology(···)
        sharder = DynamicEmbeddingCollectionSharder(
            fused_params=fused_params,
            use_index_dedup=True,
        )
        planner = DynamicEmbeddingShardingPlanner(
            eb_configs=eb_configs,
            topology=topology,
            constraints=constraints,
            batch_size=batch_size,
        )

        plan = planner.collective_plan(···)
        ...
        优化器创建
        ...
        # 模型保存
        torch.save(model.state_dict(), "model_dense.pt")
        # 动态稀疏表保存
        DynamicEmbDump(save_dir, model, optim=True)
    ```

## Rec SDK Torch迁移样例<a name="ZH-CN_TOPIC_0000002336268713"></a>

Rec SDK Torch支持Torch开源推荐模型迁移适配，迁移步骤可以参考如下：

[Rec SDK Torch recsys-gr样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/gr)
