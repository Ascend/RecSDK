# 快速入门<a name="ZH-CN_TOPIC_0000002302229552"></a>

## 使用前说明<a name="ZH-CN_TOPIC_0000002336268801"></a>

本章节提供一个基础推荐模型示例指导用户基于Rec SDK Torch快速搭建推荐模型。

## 环境准备

### 基础镜像准备

请参见[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5)中“镜像下载”页签，**根据环境架构**获取已经制作好的**最新运行镜像**：26.0.0_openeuler2203-arm或26.0.0_debian12-x86。

上述镜像中的软件配套版本如下：

| 软件名称  | PyTorch | torch_npu | torchrec  | fbgemm_gpu | hybrid_torchrec | torchrec_embcache |
|-------|---------|-----------|-----------|------------|-----------------|-------------------|
| 配套版本 | 2.6.0   | 2.6.0     | 1.1.0+npu | 1.1.0      | 1.1.0           | 1.1.0             |

### 启动容器

创建启动脚本run_docker.sh，内容如下（以下启动命令仅作参考，按需挂载目录）：

```shell
#!/bin/bash
container_name=$1
image_name=$2
free_devices=$(npu-smi info | grep 'No running processes found in NPU' | grep -o '[0-9]\+' | paste -sd ',' -)

if [ -z "${free_devices}" ]; then
    echo "No free devices! Stop docker running."
    exit 1
fi

docker run \
-it \
--name "${container_name}" \
-e ASCEND_VISIBLE_DEVICES="${free_devices}" \
--shm-size="300g" \
-m 300g \
-v /etc/localtime:/etc/localtime:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
"${image_name}" \
/bin/bash
```

部分参数说明：

- free_devices：检测当前空闲NPU卡号。
- -m 300g：设置容器内使用内存大小，可根据实际情况进行配置。
- -e ASCEND_VISIBLE_DEVICES="${free_devices}"：将服务器上空闲的NPU设备挂载到容器内，可根据实际情况进行配置。

执行如下命令新建容器：

```shell
bash run_docker.sh 容器名 镜像名称:镜像版本
```

### 环境可用性验证

执行如下命令，若正常回显NPU卡信息则说明环境可用：

```shell
npu-smi info
```

>[!NOTE]
>
> 前面章节中启动容器指令使用的非特权模式启动，如果有其他容器也挂载了相同NPU卡会导致当前容器NPU内卡不能使用，执行`npu-smi info`时会报错："dcmi model initialized failed, because the device is used. ret is -8020"。
>
> 此时请停止其他容器，确保环境可用。或者**修改容器启动指令**中的ASCEND_VISIBLE_DEVICES参数值，仅挂载可用的NPU卡。ASCEND_VISIBLE_DEVICES参数设置方式参考：`ASCEND_VISIBLE_DEVICES=0,1,2,3`，`ASCEND_VISIBLE_DEVICES=0-1`。

## 搭建模型

**图 1**  模型创建流程<a name="fig55046491373"></a>
![](../../figures/torch_rec_v1/接口调用流程.png "接口调用流程")

创建main.py脚本，添加如下内容：

```python
import logging
import itertools
import os
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
import torch_npu
from hybrid_torchrec import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from hybrid_torchrec.distributed.hybrid_train_pipeline import (
    HybridTrainPipelineSparseDist,
)
from hybrid_torchrec.distributed.sharding_plan import get_default_hybrid_sharders
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchrec import KeyedJaggedTensor, JaggedTensor
from torchrec import PoolingType
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    Topology,
    ParameterConstraints,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.streamable import Pipelineable


logging.getLogger().setLevel(logging.INFO)
FEAT_NAMES = [["phone", "clothes"], ["user"]]
TABLE_NAMES = ["product", "user"]
EMBEDDING_DIMS = [1024, 1024]
NUM_EMBEDDINGS = [10240, 10240]
ID_RANGES = [1024, 1024, 1024]
BATCH_SIZE = 32
BATCH_NUM = 20

# 1. 定义Batch
@dataclass
class Batch(Pipelineable):
    sparse_features: KeyedJaggedTensor
    labels: torch.Tensor

    def __init__(self, sparse_features, labels) -> None:
        self.sparse_features = sparse_features
        self.labels = labels

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch_npu.npu.streams.Stream) -> None:
        self.sparse_features.record_stream(stream)
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
        )


# 2. 定义Dataset
class RandomRecDataset(IterableDataset[Batch]):
    def __init__(self, batch_size, batch_num, feat_names, id_ranges):
        super().__init__()
        self.index = 0
        self.names = list(itertools.chain.from_iterable(feat_names))
        self.id_ranges = id_ranges
        self.data = [self.generate_one_batch(batch_size) for i in range(batch_num)]

    def generate_one_batch(self, batch_size) -> Batch:
        torch.manual_seed(1)
        input_dict = {}
        for name, id_range in zip(self.names, self.id_ranges):
            ids = torch.randint(0, id_range, (batch_size,))
            lengths = torch.ones(batch_size).long()
            input_dict[name] = JaggedTensor(values=ids, lengths=lengths)
        kjt_tensor = KeyedJaggedTensor.from_jt_dict(input_dict)
        label = torch.randint(0, 2, (batch_size,))
        return Batch(kjt_tensor, label)

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


# 3. 初始化分布式变量
def set_distribute_env():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6000"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(backend="hccl")


# 4. 定义模型
class TestModel(torch.nn.Module):
    def __init__(self, table_names, feat_names, embed_dims, num_embeds):
        super().__init__()
        table_configs = []

        for table_name, feat_name, dim, num_embed in zip(
            table_names, feat_names, embed_dims, num_embeds
        ):
            config = HashEmbeddingBagConfig(
                name=table_name,
                embedding_dim=dim,
                num_embeddings=num_embed,
                feature_names=feat_name,
                pooling=PoolingType.SUM,
            )
            table_configs.append(config)

        self.ebc = HashEmbeddingBagCollection(device="npu", tables=table_configs)
        self.input_dim = sum([len(f) * d for f, d in zip(feat_names, embed_dims)])
        self.linear_net = torch.nn.Linear(self.input_dim, self.input_dim)

    def forward(self, batch: Batch):
        result = self.ebc(batch.sparse_features)
        result: torch.Tensor = result.values()
        result = self.linear_net(result)
        loss = result.mean() + result.sum() + result.max() + result.min()
        return loss, result


def create_ddp(test_model):
    host_gp = dist.new_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    host_env = ShardingEnv(world_size=world_size, rank=rank, pg=host_gp)
    hybrid_sharder = get_default_hybrid_sharders(host_env=host_env)
    constraints = {
        table_name: ParameterConstraints(
            sharding_types=["row_wise"], compute_kernels=["fused"]
        )
        for table_name in TABLE_NAMES
    }

    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device="npu"),
        constraints=constraints,
    )

    plan = planner.collective_plan(test_model, hybrid_sharder, dist.GroupMember.WORLD)
    logging.info(plan)
    ddp_model = DistributedModelParallel(
        test_model, device=torch.device("npu"), plan=plan, sharders=hybrid_sharder
    )
    return ddp_model


def invoke_main():
    set_distribute_env()
    device = torch.device("npu")
    # 创建数据集
    dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
    data_loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        prefetch_factor=32,
        pin_memory_device="npu",
        num_workers=4,
    )
    # 创建模型
    test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBEDDING_DIMS, NUM_EMBEDDINGS)

    # 5. 定义稀疏表的优化器
    embedding_optimizer = torch.optim.Adagrad
    optimizer_kwargs = {"lr": 0.001, "eps": 0.1}
    apply_optimizer_in_backward(
        embedding_optimizer,
        test_model.ebc.parameters(),
        optimizer_kwargs=optimizer_kwargs,
    )

    # 6. 对稀疏表做分表
    ddp_model = create_ddp(test_model)

    # 7. 整合优化器
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=0.1),
    )
    optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])

    # 8. 创建pipeline
    pipeline = HybridTrainPipelineSparseDist(
        ddp_model, optimizer, device, execute_all_batches=True
    )

    # 9. 使用pipeline进行训练
    batched_iterator = iter(data_loader)
    for i in range(BATCH_NUM):
        logging.info("step %s done", i)
        pipeline.progress(batched_iterator)
    logging.info("demo done")


if __name__ == "__main__":
    invoke_main()

```

## 启动模型训练

启动单卡训练：

```shell
WORLD_SIZE=1 RANK=0 python3 main.py
```

启动多卡（2卡）训练：

```shell
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=2 main.py
```

预期输出：

训练结束后出现`demo done`字样，说明模型训练完成。

>[!NOTE]
> 模型训练时会占用6000端口号，若提示端口已被占用，请修改main.py脚本/启动指令（启动多卡时）中的端口号为其他未被占用的端口。

## 进阶开发

Rec SDK Torch支持Torch开源推荐模型迁移适配，如需了解开源DLRM（DCNv2）模型迁移Rec SDK Torch可参考[DLRM（DCNv2）模型迁移样例](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/dlrm/README.md)。
