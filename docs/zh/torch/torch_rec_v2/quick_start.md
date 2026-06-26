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

## 模型搭建主要步骤介绍<a name="ZH-CN_TOPIC_0000002336148917"></a>

以下出现的相关接口的参数含义及约束详细可见[接口说明](./api/README.md)

以下步骤省略了具体实现，如需完整代码，请参考[Rec SDK Torch Little Demo样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo)。关键步骤如下：

1. 定义数据集与数据转换

    自定义Dataset读取原始数据，并通过collate_fn将稀疏特征 (Sparse Features) 转换为TorchRec所需的KeyedJaggedTensor (KJT) 格式。

    ```python
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

2. 初始化分布式环境

    ```python
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

    ```python
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

    ```python
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

5. 自动化分表计划（Planner）

    通过DynamicEmbeddingShardingPlanner结合网络拓扑结构和表配置，生成分布式切分计划。

    ```python
    # model.py
    def get_planner(device, eb_configs, batch_size, training):
        ...
        return DynamicEmbeddingShardingPlanner(
            eb_configs=eb_configs,
            topology=topology,
            constraints=dict_const,
            batch_size=batch_size,
            enumerator=enumerator,
        )
    ```

6. 构建分布式模型（DMP）

    通过 DistributedModelParallel(DMP)包装原始模型。

    ```python
    # model.py
    def apply_dmp(model, args, training, device):
        ...
        plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)
        return DistributedModelParallel(
            module=model,
            device=device,
            sharders=[sharder],
            plan=plan,
        )
    ```

7. 模型状态保存与加载

    保存模型权重（Dense 部分）以及动态 Embedding 数据。
    稀疏权重和优化器状态需使用专用接口。

    ```python
    # 接口调用
    from dynamic_emb.distributed.dump_load import DynamicEmbDump
    from dynamic_emb.distributed.dump_load import DynamicEmbLoad
    ```

    - 保存：```DynamicEmbDump(save_dir, model, optim=True)```
    - 加载：```DynamicEmbLoad(save_dir, model, optim=True)```

## 启动模型训练<a name="ZH-CN_TOPIC_0000002302229704"></a>

模型运行依赖容器环境。用户可基于已经制作好的容器镜像，快速启动容器并运行模型训练。也可以手动进行容器镜像环境的依赖安装和软件部署。

### 方案1：基于已有容器镜像启动模型训练

1. 获取已有运行镜像并启动容器
请参见[OVERVIEW](../../../../docker/OVERVIEW.zh.md)基于已经制作好的容器镜像，启动容器并进入容器内。

    上述镜像中的软件配套版本如下：

    |软件名称|PyTorch|torch_npu|TorchRec|fbgemm_gpu|dynamic_emb|
    |--|--|--|--|--|--|
    |配套版本|2.7.1|2.7.1|1.2.0+npu|1.2.0|25.09|

2. 配置容器内环境

    执行如下指令配置容器内环境

    ```bash
    # 设置CANN版本
    source /usr/local/set_cann_env.sh a5 # 切换并生效 Ascend 950 配套Toolkit及相关环境变量
    # 激活python虚拟环境
    source /opt/buildtools/torch_v2_pt2.7.1/bin/activate
    ```

3. 启动模型训练

    进入样例代码目录：

    ```bash
    cd /RecSDK/torch_rec_v2_examples/little_demo
    ```

    下载 MovieLens-1M 数据集并解压到当前目录的 ml-1m 文件夹。
    数据集链接：[MovieLens-1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip) 。

    下载后的文件名为ml-1m.zip，将其解压。样例模型默认使用./ml-1m为数据路径，若需更改路径，可在run.sh脚本中通过--data_path参数指定，如`--data_path ./dataset/ml-1m`

    执行模型运行脚本：

    ```bash
    bash run.sh
    ```

    run.sh脚本会自动设置PYTHONPATH并启动训练。
    命令行参数 --train 参数指定模型训练， --dump --load 指定模型进行端到端验证，执行训练->保存->加载->推理全过程。

    ```bash
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=1 main.py --train "$@"
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=1 main.py --load --dump "$@"
    ```

    模型运行完成显示`Demo done.`

### 方案2：手动准备运行环境并启动模型训练

1. 请参见[安装Rec SDK Torch](./recsdk_torch_installation_guide.md#section182972951211)章节进行容器环境准备，启动并进入容器。
2. 下载[Rec SDK Torch Little Demo样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/little_demo)代码。
3. 启动模型训练同方案1。
