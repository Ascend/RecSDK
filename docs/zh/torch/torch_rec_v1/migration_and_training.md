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

-   TorchRec示例：

    ```bash
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import get_default_sharders
    class TestModel(torch.nn.Module):
        def __init__(self, *):
            # Rec SDK Torch 使用的接口为HashEmbeddingBagCollection
            self.ebc = EmbeddingBagCollection()
        def forward(self, batch: Batch):
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
            pipeline.progress(batched_iterator)
    ```

-   Rec SDK Torch示例：

    ```bash
    from hybrid_torchrec import HashEmbeddingBagCollection
    from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
    from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
    ...
    class TestModel(torch.nn.Module):
        def __init__(self, *):
            # 原生TorchRec使用的接口为EmbeddingBagCollection
            self.ebc = HashEmbeddingBagCollection()
        def forward(self, batch: Batch):
            pass
    def invoke_main():
        rank, world_size = get_distribute_env()
        device = torch.device("npu")
        dist.init_process_group(backend="hccl")
        # Rec SDK Torch创建host连接
        host_gp = dist.new_group(backend="gloo")
        host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
        dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
        data_loader = DataLoader(
            dataset,
        )
        test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS ,NUM_EMBEDS)
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
            pipeline.progress(batched_iterator)
    ```


## Rec SDK Torch迁移样例<a name="ZH-CN_TOPIC_0000002336268713"></a>

Rec SDK Torch支持Torch开源推荐模型迁移适配，迁移步骤可以参考如下：

[Rec SDK Torch Dcnv2迁移样例](https://gitcode.com/Ascend/RecSDK/blob/develop_torch_benchmark/torch2.6.0_examples_benchmark/develop/dlrm/README.md)。


