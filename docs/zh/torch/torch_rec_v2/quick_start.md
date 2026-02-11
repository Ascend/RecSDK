# 快速入门<a name="ZH-CN_TOPIC_0000002302229552"></a>

## 使用前说明<a name="ZH-CN_TOPIC_0000002336268801"></a>

本章节提供一个little-demo用例指导用户基于动态稀疏表Rec SDK Torch搭建模型。little-demo存放路径为：[Rec SDK Torch Little Demo样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo)。

**表 1**  little-demo文件说明

|文件名|说明|
|--|--|
|main.py|模型训练入口|
|dataset.py|数据集解析与KeyedJaggedTensor数据构建|
|model.py|模型文件|
|run.sh|启动脚本|
|logger.py|日志模块定义|
|README.md|数据集下载及demo模型运行说明|



## 接口调用介绍<a name="ZH-CN_TOPIC_0000002336148917"></a>

以下步骤省略了具体实现，如需完整代码，请参考[Rec SDK Torch Little Demo样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo)。关键步骤如下：

1.  <a id="li524311501994"></a>定义数据集与数据转换

    自定义Dataset读取原始数据，并通过collate_fn将稀疏特征（Sparse Features）转换为TorchRec所需的KeyedJaggedTensor(KJT)格式。

    ```cpp
    # dataset.py
    def collate_fn(batch):
        ...
        kjt = KeyedJaggedTensor(
            keys=list(values.keys()),
            values=torch.cat(list(values.values())),
            lengths=torch.cat(list(lengths.values())),
        )
        return kjt, torch.tensor(labels, dtype=torch.float)

    # main.py
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        ...
    )
    ```

2.  初始化分布式环境

    ```cpp
    # main.py
    backend = "hccl"
    dist.init_process_group(backend=backend)
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    ```

3. 定义模型结构

    构建包含EmbeddingCollection的模型。

    ```cpp
    # model.py
    class MovieLensModel(nn.Module):
        def __init__(self, embedding_module, ...):
            super().__init__()
            ...
            self.embedding_module = embedding_module
            self.over_arch = nn.Sequential(...)

        def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
            ...
            embeddings = self.embedding_module(kjt)
            ...
            return prediction
    ```

4. 配置DynamicEmbeddingSharder
    
    创建 DynamicEmbeddingCollectionSharder，并配置融合优化器参数。
    
    ```cpp
    # model.py
    def get_sharder(args, optimizer_type):
        ...
        optimizer_kwargs = {
            "optimizer": optimizer_type, 
            "learning_rate": learning_rate, 
            ...
        }
        ...
        return DynamicEmbeddingCollectionSharder(
            fused_params=fused_params,
            use_index_dedup=True,
        )
    ```

4. 自动化分表计划（Planner）

    通过DynamicEmbeddingShardingPlanner结合网络拓扑结构和表配置，生成分布式切分计划。

    ```cpp
    # model.py
    def get_planner(device, eb_configs, batch_size, training):
        ···
        return DynamicEmbeddingShardingPlanner(
            eb_configs=eb_configs,
            topology=topology,
            constraints=dict_const,
            batch_size=batch_size,
            enumerator=enumerator,
        )
    ```

5. 构建分布式模型（DMP）

    通过 DistributedModelParallel(DMP)包装原始模型。

    ```cpp
    # model.py
    def apply_dmp(model, args, training, device):
        ···
        plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)
        return DistributedModelParallel(
            module=model,
            device=device,
            sharders=[sharder],
            plan=plan,
        )
    ```

6. 模型状态保存与加载

    保存模型权重（Dense 部分）以及动态 Embedding 数据。
    
    稀疏权重和优化器状态需使用专用接口
    - 保存：```DynamicEmbDump(save_dir, model, optim=True)```
    - 加载：```DynamicEmbLoad(save_dir, model, optim=True)```
    

## 启动模型训练<a name="ZH-CN_TOPIC_0000002302229704"></a>

1. 准备数据
    
    下载 MovieLens-1M 数据集并解压到当前目录的 ml-1m 文件夹。

2. 直接运行run.sh。
    
    该脚本会自动设置PYTHONPATH并启动训练。
    
    ```bash
    bash run.sh
    ```