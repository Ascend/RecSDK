# little demo模型迁移样例
## 前提条件
需要用户构建容器基础环境，参考RecSDK/torchrec/docker/README.md。


## 模型介绍
本样例以一个最基础的模型来介绍RecSDK-Torch的使用样例。搭建一个基于RecSDK-Torch的训练任务步骤如下：
1. 定义Batch

将本次训练需要的所有特征整合为一个Batch类。并实现to()、pin_memory()、record_stream()方法。完整代码参考dataset.py文件。
```python
@dataclass
class Batch(Pipelineable):
    ...
```
2. 定义Dataset

实现一个返回Batch的Dataset。完整代码参考dataset.py文件。

```python
class RandomRecDataset(IterableDataset[Batch]):
    ...
```
3. 初始化分布式变量

创建host和device侧的链接。完整代码参考main.py文件。
```python
......
dist.init_process_group(backend="hccl")
host_gp = dist.new_group(backend="gloo")
host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
```
4. 定义模型

将稀疏表部分和Dense部分整合为一个Module。该module的输入必须要上述环节中定义的Batch类。返回为模型的loss和输出。完整代码参考model.py。
```python
class TestModel(torch.nn.Module):
    def __init__(self, table_names, feat_names, embed_dims, num_embeds):
        self.ebc = HashEmbeddingBagCollection(...)

    def forward(self, batch: Batch):
        ...
        return loss, result
```
5. 定义稀疏表的优化器

指定sparse侧的优化器
```python
test_model = TestModel(...)

# Optimizer
embedding_optimizer = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.001, "eps": 0.1}
apply_optimizer_in_backward(
    embedding_optimizer,
    test_model.ebc.parameters(),
    optimizer_kwargs=optimizer_kwargs,
)
```
6. 对稀疏表做分表

创建sharder，并使用EmbeddingShardingPlanner创建分表计划，将模型、分表计划和sharder传入DistributedModelParallel中获得分布式模型。注意当前支持row-wise和fused模式。完整代码参考main.py。
```python
    hybrid_sharder = get_default_hybrid_sharders(host_env)
    constraints = {...}
    planner = EmbeddingShardingPlanner(...)
    plan = planner.collective_plan(...)
    logging.info(plan)
    ddp_model = DistributedModelParallel(
        test_model, device=torch.device("npu"), plan=plan, sharders=hybrid_sharder
    )
```

7. 整合优化器

分离dense和sparse的参数，并组合成一个新的优化器。完整代码参考main.py。
```python
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
```
8. 创建pipeline

完整代码参考main.py
```python
pipeline = HybridTrainPipelineSparseDist(
    ddp_model, optimizer, device, execute_all_batches=True
)
```
9. 使用pipeline进行训练

完整代码参考main.py
```python
batched_iterator = iter(data_loader)
for i in range(10):
    pipeline.progress(batched_iterator)
```


## 运行脚本

### 单机运行
```bash
WORLD_SIZE=1 RANK=0 python main.py
```

### 多机运行
使用`torchx`运行分布式程序，缺少`torchx`需要执行下面命令安装:
```bash
pip install torchx
```
运行脚本启动训练：
```bash
bash bash.sh
```
成功后出现demo done字样。
