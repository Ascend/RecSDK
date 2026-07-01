# 迁移与训练<a name="ZH-CN_TOPIC_0000002302389292"></a>

## 训练场景介绍<a name="ZH-CN_TOPIC_0000002302229612"></a>

**基于Rec SDK Torch搭建网络<a name="section16627105015515"></a>**

用户可按[快速入门](./quick_start.md)的步骤搭建模型并进行训练。

以下出现的相关接口的参数含义及约束详细可见[接口说明](./api/README.md)。

迁移与训练中使用的软件配套版本如下所示：

|软件名称|PyTorch|torch_npu|torchrec|fbgemm_gpu|dynamic_emb|
|--|--|--|--|--|--|
|配套版本|2.7.1|2.7.1|1.2.0+npu|1.2.0|25.09|

**基于开源TorchRec进行迁移<a name="section9248145363514"></a>**

如果用户已经在TorchRec上搭建了网络，则按照接口对应关系进行替换，如[表1](#table16435142101913)所示。

**表 1**  接口对应关系
<a id="table16435142101913"></a>

|TorchRec接口|Rec SDK Torch接口|接口功能描述|
|--|--|--|
|EmbeddingCollectionSharder|DynamicEmbeddingCollectionSharder|稀疏表分片器|
|EmbeddingShardingPlanner|DynamicEmbeddingShardingPlanner|分片计划生成器|
|EmbeddingEnumerator|DynamicEmbeddingEnumerator|分片选项枚举|
|ParameterConstraints|DynamicEmbParameterConstraints|分表约束|
|PyTorch保存/加载|DynamicEmbDump/DynamicEmbLoad|动态稀疏表保存与加载|

接口示例：

- TorchRec示例：

    ```python
    from torchrec.distributed.embedding import EmbeddingCollection
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
    from torchrec.distributed.planner import EmbeddingShardingPlanner
    from torchrec.distributed.model_parallel import get_default_sharders

    class TestModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            self.ec = EmbeddingCollection(...)

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
        test_model = TestModel(TABLE_NAMES, FEAT_NAMES, EMBED_DIMS, NUM_EMBEDS)
        ...

        sharder = get_default_sharders()
        planner = EmbeddingShardingPlanner(...)

        sharding_plan = planner.collective_plan(...)
        ...
        优化器创建
        ...
        pipeline = TrainPipelineSparseDist(test_model, optimizer, device)
        for i in range(20):
            pipeline.progress(batched_iterator)
    ```

- Rec SDK Torch示例：

    ```python
    import torch
    import torch.nn
    import torch.distributed as dist
    from torchrec.distributed.embedding import EmbeddingCollection
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
        def __init__(self, *args, **kwargs):
            self.ec = EmbeddingCollection(...)
        def forward(self, *args, **kwargs):
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
            ...
        ]
        # 设置分表约束
        constraints = {}
        for eb_config in eb_configs:
            constraints[eb_config.name] = DynamicEmbParameterConstraints(...)

        # 创建分片器和规划器
        topology = Topology(...)
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

        plan = planner.collective_plan(...)
        ...
        优化器创建
        ...
        # 模型保存
        torch.save(model.state_dict(), "model_dense.pt")
        # 动态稀疏表保存
        DynamicEmbDump(save_dir, model, optim=True)
        ...
        # 动态稀疏表加载
        DynamicEmbLoad(save_dir, model, optim=True)
    ```

## Rec SDK Torch迁移样例<a name="ZH-CN_TOPIC_0000002336268713"></a>

### Recsys-GR模型适配

Rec SDK Torch支持Torch开源推荐模型迁移适配，本章节介绍将开源模型Recsys-GR迁移至Rec SDK Torch框架的主要修改及基于Recsys-GR的性能调优实例。完整的代码修改适配及配套版本可参见以下链接：

[Rec SDK Torch Recsys-GR样例](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/gr)

#### 迁移主要修改

##### 1. 设备管理API修改

命名空间变化，修改CUDA接口至NPU接口：`torch.cuda`->`torch_npu.npu`,如：

| CUDA API| NPU API | 位置|
| --- | --- | --- |
|torch.cuda|torch_npu.npu|全文件|
|torch.cuda.device_count() | torch_npu.npu.device_count() | distributed_utils.py|
|torch.cuda.current_device() | torch_npu.npu.current_device() | training.py, hstu_layer.py, pretrain_gr_ranking.py|
|torch.cuda.mem_get_info() | torch_npu.npu.mem_get_info()| pretrain_gr_ranking.py|
|torch.cuda.stream()|torch_npu.npu.stream()|embedding.py|

##### 2. GIN配置参数扩展

扩展 NetworkArgs 类，新增 NPU 适配相关参数，扩展kernel_backend支持。

```python
# 添加NPU后端支持
layer_type: str = "fused"
...
assert self.kernel_backend.lower() in ["cutlass", "triton", "pytorch", "npu_fused"]
assert self.layer_type.lower() in ["fused", "native"]
...
elif network_args.kernel_backend == "npu_fused":
    kernel_backend = KernelBackend.NPU_FUSED
...
```

##### 3. dynamic_emb接口替换

配置NPU dynamic_emb接口替换GPU实现。

```python
# 稀疏表分表相关接口
from dynamic_emb import (
    DynamicEmbeddingEnumerator,
    DynamicEmbParameterConstraints,
    DynamicEmbTableOptions,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbeddingCollectionSharder,
)
...
# 稀疏表配置项接口
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbEvictStrategy
from dynamic_emb import DynamicEmbCheckMode, DynamicEmbInitializerArgs, DynamicEmbInitializerMode
...
```

##### 4. 算子适配

将CUDA上Triton/Cutlass实现的算子转化为Pytorch原生实现/RecSDK推荐业务算子。算子实现位于`example/hstu/ops`文件夹下。

- jagged_to_padded_dense / dense_to_jagged CUDA版本依赖FBGEMM算子，NPU替换为RecSDK业务算子实现
- concat_2D_jagged / split_2D_jagged Triton算子替换为Pytorch实现
- Position Encoding算子 Triton算子替换为Pytorch实现

##### 5. HSTU层适配

- 取消对于`megatron`中`TEColumnParallelLinear`和`TERowParallelLinear`的引用，相关代码替换为`torch.nn`的方法

```python
self._output_layernorm_weight = torch.nn.Parameter(
    torch.ones(self._num_heads * self._linear_dim_per_head, device=device)
)
self._output_layernorm_bias = torch.nn.Parameter(
    torch.zeros(self._num_heads * self._linear_dim_per_head, device=device)
)
self._linear_uvqk = torch.nn.Linear(
    self._embedding_dim,
    sum(self._split_arg_list),
    bias=True,
).apply(init_mlp_weights_optional_bias)
```

- 新增`NpuFusedHSTUAttention`适配`torch.ops.mxrec.hstu_jagged`算子

```python
class NpuFusedHSTUAttention(HSTUAttention):
    def forward(self,
                tq: torch.Tensor,  # (T, d)
                tk: torch.Tensor,  # (T, d)
                tv: torch.Tensor,  # (T, d)
                offsets: torch.Tensor,  # (batch_size, 1)
                max_seqlen: int,
                target_group_size: int = 1,  # target == candidates
                num_candidates: Optional[torch.Tensor] = None,
                num_contextuals: Optional[Union[int, torch.Tensor]] = None,
                ) -> torch.Tensor:  # T, d

        return torch.ops.mxrec.hstu_jagged(tq.view(-1, self.num_heads, self.attention_dim),
                                          tk.view(-1, self.num_heads, self.attention_dim),
                                          tv.view(-1, self.num_heads, self.attention_dim),
                                          None,
                                          None,
                                          0,  # 0默认为下三角，跟TorchHSTUAttention的mask实现不一致
                                          max_seqlen,
                                          1.0 / max_seqlen,
                                          offsets.long(),
                                          num_contextuals,
                                          num_candidates,
                                          target_group_size,
                                          1.0 / (self.attention_dim**0.5),
                                          ).view(-1, self.num_heads * self.attention_dim)
```

#### Recsys-GR 性能调优实例

##### 背景

以非FSDP2模式为例，Recsys-GR 模型在不同架构（x86/ARM）及主频的CPU环境下，端到端训练性能存在显著差距。对比两类硬件环境，device侧设备规格参数相同，核心区别在于CPU架构规格与主频差异，低主频ARM环境性能劣化严重，初步判定存在Host Bound瓶颈。

##### 分析

###### 1. 设备侧（Device）算子耗时验证

设置环境变量 `NPU_PROFILE=1`，使用torch.profiler采集模型性能数据。

```python
if PROFILE_ENABLE:
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
        torch_npu.profiler.ExportType.Text,
        ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    prof = torch_npu.profiler.profile(
        activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=10, warmup=0, active=1, repeat=1, skip_first=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_modules=True,
        with_flops=False,
        experimental_config=experimental_config)

    prof.start()
```

对比NPU侧Kernel耗时，可得两种环境性能基本无差异，可排除Device侧导致的性能劣化。

###### 2. 整体计算耗时分析

利用获取的profiling文件，可使用[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)可视化工具进行时间线（Timeline）可视化分析。

x86与ARM环境下模型的Free时间占比均超70%，NPU实际计算占比极低。

**核心推论**：性能瓶颈不在 NPU 算力，而在Host侧CPU操作（数据搬运、算子下发、调度等待）。

##### 调优方案

###### 1. 多线程数据加载优化

模型采用DataLoader单线程加载（完全运行在CPU上），主进程串行读数据，CPU成为瓶颈，NPU长期空闲。单线程模式大幅放大了CPU主频差异带来的性能差距。

可启用多核多进程加载，支持预加载、进程常驻、数据分区、并行打乱。例如对数据加载函数进行如下优化：

```python
def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
    num_workers: int = 8,
    prefetch_factor: int = 2,
) -> DataLoader:
    def worker_init_fn(worker_id: int) -> None:
        """
        Worker初始化函数，设置每个worker的数据分区
        """
        if hasattr(dataset, 'set_worker_id'):
            dataset.set_worker_id(worker_id, num_workers)
            # 重新shuffle该worker的数据，使用不同的随机种子
            if hasattr(dataset, '_shuffle_batch'):
                dataset._shuffle_batch(worker_id=worker_id)

    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return loader
```

同步修改`recsys-example/examples/hstu/dataset/sequence.py`文件，设置每个worker的数据分区，避免数据重复：

```python
def set_worker_id(self, worker_id: int, num_workers: int) -> None:
    """
    设置当前worker的ID和总worker数，用于多进程数据分区。
    每个worker只处理分配给它的数据子集。

    Args:
        worker_id: 当前worker的ID (0 to num_workers-1)
        num_workers: 总worker数量
    """
    if num_workers <= 1:
        return

    total_samples = len(self._sample_ids)
    samples_per_worker = total_samples // num_workers

    # 计算当前worker负责的数据范围
    start_idx = worker_id * samples_per_worker
    end_idx = start_idx + samples_per_worker if worker_id < num_workers - 1 else total_samples

    # 截取该worker负责的样本ID
    self._sample_ids = self._sample_ids[start_idx:end_idx]
    self._num_samples = len(self._sample_ids)

    # 重新计算batch数量
    self._num_batches = math.ceil(self._num_samples / self._global_batch_size)
```

开启多线程加载后，数据加载时间大幅减小，且对模型训练精度无显著影响。

###### 2. 开启大页内存池

Linux 默认4KB内存页会产生大量TLB Miss和缺页中断，大页内存可大幅降低该开销。详见[开启大页内存池](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0070.html)

在模型run.sh脚本中使能OS开启透明大页内存：

```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

###### 3. jemalloc 内存分配器优化

使用 CANN 优化版 jemalloc 提升内存分配效率。使用模型run.sh脚本进行加载：

```bash
export LD_PRELOAD=${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/lib64/libjemalloc.so
```

###### 4. 细粒度绑核、NUMA 内存绑定

原模型脚本粗粒度绑核效果差，需要针对Python主进程、acl_thread做专属绑核，并配置内存亲和性。

开启算子队列优化:

```bash
export TASK_QUEUE_ENABLE=2
```

单卡 NUMA 内存绑定：

使用NUMA绑定前需分别安装系统级依赖以及python库依赖

```bash
# 安装系统库依赖
# CentOS/RHEL
yum install numactl numactl-devel
# Ubuntu/Debian
apt install numactl libnuma-dev

# 安装python库
pip install py-libnuma
```

```python
# 绑核
import numa
def bind_memory_to_numa0():
    numa.memory.set_membind_nodes(0)
    return True
```

###### 5. 其他调优方向

- 业务框架层数据并行策略优化，如采用流水并行等。
- 编译优化，详见：[编译优化技术介绍](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0062.html)
- BIOS参数调优，如开启超频、高性能模式等。

### DLRM (DCNv2) 模型适配

本章节介绍将开源DLRM（DCNv2）模型迁移到Rec SDK Torch框架时的主要迁移修改。完整的迁移修改请参见[README](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/dlrm/README.md)，查看应用patch文件后的代码。

迁移时的主要修改内容为将开源模型中使用到的TorchRec原生的稀疏表配置等API替换为Rec SDK Torch框架中的API，训练流水线使用TorchRec原生实现。

迁移前请先下载开源模型代码并切换到指定commit版本：

```bash
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99
```

主要修改内容如下：

1. 修改分布式后端

    增加HCCL分布式后端选项：

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

2. 修改稀疏表配置

    使用EC模式的EmbeddingConfig配置稀疏表

    ```python
    from torchrec import EmbeddingConfig
    eb_configs = [
        EmbeddingConfig(
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

3. 配置EmbeddingCollection

    新增EC版本的模型定义`DLRM_DCN_EC`对原模型进行替换，新增模型代码在:`dlrm/torchrec_dlrm/ec_dcnv2.py`下

    ```python
    from torchrec.distributed.embedding import EmbeddingCollection
    ...
    elif args.interaction_type == InteractionType.DCN:
        dlrm_model = DLRM_DCN_EC(
            embedding_collection=EmbeddingCollection(
                tables=eb_configs,
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
    ```

4. 修改分表计划和创建分布式模型

    使用DynamicEmb相关接口替换TorchRec原生接口

    ```python
    from torchrec.distributed.planner import Topology
    from torchrec.distributed.planner.types import ShardingType
    from fbgemm_gpu.split_embedding_configs import SparseType
    from dynamic_emb_extensions import OptimizerType # 动态稀疏表自定义算子库
    from dynamic_emb import (
        DynamicEmbeddingCollectionSharder,
        DynamicEmbeddingShardingPlanner,
        DynamicEmbTableOptions,
        DynamicEmbParameterConstraints,
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
    )
    ...
    constraints = {
        f"t_{feature_name}": DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=["fused"],
            dynamicemb_options=DynamicEmbTableOptions(
                training=True,
                optimizer_type=OptimizerType.AdaGrad if args.adagrad else OptimizerType.SGD,
                embedding_dtype=torch.float32,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.TRUNCATED_NORMAL,
                    mean=0.0,
                    std_dev=0.01,
                ),
            ),
        )
    for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    }

    planner = DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=Topology(
            world_size=dist.get_world_size(),
            compute_device="npu",
        ),
        batch_size=args.batch_size,
        constraints=constraints,
    )

    sharders = [
        DynamicEmbeddingCollectionSharder(
            fused_params={
                "output_dtype": SparseType.FP32,
                "optimizer": "adagrad" if args.adagrad else "sgd",
                "learning_rate":args.learning_rate
            },
            use_index_dedup=False,
        )
    ]

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
