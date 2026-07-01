# Migration and Training

## Training Scenarios

**Building a Network with Rec SDK Torch**

Follow the steps in [Quick Start](./quick_start.md) to build and train a model.

**Migrating from Open-Source TorchRec**

If you have already built a network with TorchRec, replace the APIs according to the mapping in [Table 1](#table16435142101913).

**Table 1** API mapping
<a id="table16435142101913"></a>

|TorchRec API|Rec SDK Torch API|Description|
|--|--|--|
|EmbeddingCollectionSharder|DynamicEmbeddingCollectionSharder|Sparse table sharder|
|EmbeddingShardingPlanner|DynamicEmbeddingShardingPlanner|Sharding plan generator|
|EmbeddingEnumerator|DynamicEmbeddingEnumerator|Sharding option enumerator|
|ParameterConstraints|DynamicEmbParameterConstraints|Sharding constraints|
|PyTorch saving/loading|DynamicEmbDump/DynamicEmbLoad|Dynamic sparse table saving and loading|

API example:

- TorchRec example

    ```python
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
        Optimizer creation
        ...
        pipeline = TrainPipelineSparseDist(test_model, optimizer, device)
        for i in range(20):
            pipeline.progress(batched_iterator)
    ```

- Rec SDK Torch example

    ```python
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
        # Set sharding constraints.
        constraints = {}
        for eb_config in eb_configs:
            constraints[eb_config.name] = DynamicEmbParameterConstraints(···)

        # Create the sharder and planner.
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
        Optimizer creation
        ...
        # Save the model.
        torch.save(model.state_dict(), "model_dense.pt")
        # Save the dynamic sparse table.
        DynamicEmbDump(save_dir, model, optim=True)
    ```

## Rec SDK Torch Migration Examples

### Recsys-gr Model Adaptation

Rec SDK Torch supports the migration and adaptation of open-source Torch recommendation models. This section describes the main modifications required to migrate the open-source Recsys-gr model to the Rec SDK Torch framework and provides a performance tuning example based on Recsys-gr. For the complete code adaptation, see the following link:

[Rec SDK Torch Recsys-gr Example](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/gr)

#### Main Migration Changes

##### 1. Changing Device Management APIs

The namespace changes. Replace CUDA APIs with NPU APIs. For example, change `torch.cuda` to `torch_npu.npu`.

| CUDA API| NPU API | Location|
| --- | --- | --- |
|torch.cuda|torch_npu.npu|Entire file|
|torch.cuda.device_count() | torch_npu.npu.device_count() | distributed_utils.py|
|torch.cuda.current_device() | torch_npu.npu.current_device() | training.py, hstu_layer.py, pretrain_gr_ranking.py|
|torch.cuda.mem_get_info() | torch_npu.npu.mem_get_info()| pretrain_gr_ranking.py|
|torch.cuda.stream()|torch_npu.npu.stream()|embedding.py|

##### 2. Extending GIN Configuration Parameters

Extend the `NetworkArgs` class by adding NPU adaptation parameters and expanding support for `kernel_backend`.

```python
# Add NPU backend support.
layer_type: str = "fused"
···
assert self.kernel_backend.lower() in ["cutlass", "triton", "pytorch", "npu_fused"]
assert self.layer_type.lower() in ["fused", "native"]
···
elif network_args.kernel_backend == "npu_fused":
    kernel_backend = KernelBackend.NPU_FUSED
···
```

##### 3. Replacing `dynamic_emb` APIs

Configure the NPU `dynamic_emb` APIs to replace the GPU implementation.

```python
# Sparse table sharding-related APIs
from dynamic_emb import (
    DynamicEmbeddingEnumerator,
    DynamicEmbParameterConstraints,
    DynamicEmbTableOptions,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbeddingCollectionSharder,
)
···
# Sparse table configuration APIs
from dynamic_emb.distributed.dynamicemb_config import DynamicEmbEvictStrategy
from dynamic_emb import DynamicEmbCheckMode, DynamicEmbInitializerArgs, DynamicEmbInitializerMode
···
```

##### 4. Adapting Operators

Convert CUDA operators implemented with Triton and Cutlass to native PyTorch implementations or Rec SDK service operators. The operator implementations are in the `example/hstu/ops` directory.

- In the CUDA version, `jagged_to_padded_dense` and `dense_to_jagged` depend on FBGEMM operators. The NPU version replaces them with Rec SDK service operator implementations.
- The Triton operators `concat_2D_jagged` and `split_2D_jagged` are replaced with PyTorch implementations.
- The Triton implementation of the position encoding operator is replaced with a PyTorch implementation.

##### 5. Adapting the HSTU Layer

- Remove references to `TEColumnParallelLinear` and `TERowParallelLinear` in `megatron`. Replace the related code with `torch.nn` methods.

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

- Add `NpuFusedHSTUAttention` to adapt the `torch.ops.mxrec.hstu_jagged` operator.

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
                                          0,  # 0 defaults to the lower triangle, which differs from the mask implementation in TorchHSTUAttention
                                          max_seqlen,
                                          1.0 / max_seqlen,
                                          offsets.long(),
                                          num_contextuals,
                                          num_candidates,
                                          target_group_size,
                                          1.0 / (self.attention_dim**0.5),
                                          ).view(-1, self.num_heads * self.attention_dim)
```

#### Recsys-gr Performance Tuning Example

##### Background

Using non-FSDP2 mode as an example, the Recsys-gr model shows a significant difference in end-to-end training performance across CPU environments with different architectures (x86 and Arm) and clock frequencies. After comparing the two hardware environments, the device-side specifications are the same. The main difference is the CPU architecture and clock frequency. Performance drops sharply in the low-frequency Arm environment, which suggests a host-side bottleneck.

##### Analysis

###### 1. Verifying the Device-Side Operator Runtime

Set the `NPU_PROFILE=1` environment variable and use `torch.profiler` to collect model performance data.

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

A comparison of NPU-side kernel runtime shows almost no difference between the two environments. Therefore, device-side issues are not the cause of the performance degradation.

###### 2. Analyzing the Overall Computation Time

With the collected profiling file, use the [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html) visualization tool to analyze the timeline.

Free time accounts for more than 70 percent in both the x86 and Arm environments, and actual NPU computation accounts for very little.

**Main hypothesis**: The performance bottleneck is not the NPU compute capacity, but host-side CPU operations such as data movement, operator dispatch, and scheduling delays.

##### Tuning Plan

###### 1. Optimizing Multi-threaded Data Loading

The model uses a single-threaded `DataLoader` that runs entirely on the CPU. The main process reads data serially, so the CPU becomes the bottleneck and the NPU stays idle for long periods. The single-threaded mode greatly amplifies the performance gap caused by CPU frequency differences.

You can enable multi-core, multi-process data loading and support prefetching, persistent workers, data partitioning, and parallel shuffling. For example, optimize the data loading function as follows:

```python
def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
    num_workers: int = 8,
    prefetch_factor: int = 2,
) -> DataLoader:
    def worker_init_fn(worker_id: int) -> None:
        """
        Worker initialization function. Sets the data partition for each worker.
        """
        if hasattr(dataset, 'set_worker_id'):
            dataset.set_worker_id(worker_id, num_workers)
            # Reshuffle this worker's data using a different random seed.
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

Apply the same update to `recsys-example/examples/hstu/dataset/sequence.py`, and partition data for each worker to avoid duplicates:

```python
def set_worker_id(self, worker_id: int, num_workers: int) -> None:
    """
    Set the current worker ID and the total number of workers for multi-process data partitioning.
    Each worker processes only the subset of data assigned to it.

    Args:
        worker_id: current worker ID (0 to num_workers-1)
        num_workers: total number of workers
    """
    if num_workers <= 1:
        return

    total_samples = len(self._sample_ids)
    samples_per_worker = total_samples // num_workers

    # Calculate the data range handled by this worker.
    start_idx = worker_id * samples_per_worker
    end_idx = start_idx + samples_per_worker if worker_id < num_workers - 1 else total_samples

    # Slice the sample IDs handled by this worker.
    self._sample_ids = self._sample_ids[start_idx:end_idx]
    self._num_samples = len(self._sample_ids)

    # Recalculate the batch count.
    self._num_batches = math.ceil(self._num_samples / self._global_batch_size)
```

After multi-threaded loading is enabled, data loading time drops significantly, and model training accuracy is barely affected.

###### 2. Enabling a Huge Page Memory Pool

Linux's default 4 KB memory pages cause many TLB misses and page faults. Huge pages can greatly reduce this overhead. For details, see [Enabling a Huge Page Memory Pool](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0070.html).

Enable transparent huge pages in the OS in the model `run.sh` script:

```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

###### 3. Optimizing the `jemalloc` Memory Allocator

Use the CANN-optimized `jemalloc` to improve memory allocation efficiency. Load it in the model `run.sh` script:

```bash
export LD_PRELOAD=${ASCEND_CANN_PACKAGE_PATH}/${ARCH}-linux/lib64/libjemalloc.so
```

###### 4. Configuring Fine-Grained CPU Core Binding and NUMA Memory Binding

The original model script uses coarse-grained CPU core binding, which performs poorly. Bind the Python main process and `acl_thread` to dedicated cores, and configure memory affinity.

Enable operator queue optimization:

```bash
export TASK_QUEUE_ENABLE=2
```

Configure single-card NUMA memory binding:

```python
import numa
def bind_memory_to_numa0():
    numa.memory.set_membind_nodes(0)
    return True
```

###### 5. Other Tuning Directions

1. Optimize the data parallel strategy at the service framework layer, for example, by using pipeline parallelism.
2. Compilation optimization. For details, see [Introduction to Compilation Optimization Technologies](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0062.html).
3. BIOS parameter tuning, such as enabling overclocking and high-performance mode.

### DLRM (DCNv2) Model Adaptation

This section describes the main migration changes when migrating the open-source DLRM (DCNv2) model to the Rec SDK Torch framework. For the complete migration changes, see [README](https://gitcode.com/Ascend/RecSDK/tree/develop_examples_and_tools/torch_rec_v2_examples/dlrm/README.md) and review the code after applying the patch file.

The main migration change is to replace the TorchRec native sparse table configuration APIs used by the open-source model with APIs in the Rec SDK Torch framework. The training pipeline still uses the TorchRec native implementation.

Before migrating, download the open-source model code and switch to the specified commit:

```bash
git clone -b main https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout b631a99
```

The main changes are as follows:

1. Modify the distributed backend.

    Add the HCCL distributed backend option:

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

2. Modify the sparse table configuration.

    Configure sparse tables using `EmbeddingConfig` in EC mode.

    ```python
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

3. Configure `EmbeddingCollection`.

Add the EC version model definition `DLRM_DCN_EC` to replace the original model. The new model code is in `dlrm/torchrec_dlrm/ec_dcnv2.py`.

    ```python
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

4. Modify the sharding plan and create the distributed model.

    Use the related DynamicEmb APIs to replace the native TorchRec APIs.

    ```python
    from torchrec.distributed.planner import Topology
    from torchrec.distributed.planner.types import ShardingType
    from fbgemm_gpu.split_embedding_configs import SparseType
    from dynamic_emb_extensions import OptimizerType
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
                "learning_rate":args.learning_rate,
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
