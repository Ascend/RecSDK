# Pipeline APIs

## `HybridTrainPipelineSparseDist`

### Initialization

**Description**

Creates a distributed training data pipeline scheduler in full-device memory mode.

**Function Prototype**

```python
class HybridTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        return_loss: bool = False,
        pipe_n_batch: int = 6,
    ) -> None:
```

**Parameters**

| Parameter         | Type                   | Mandatory/Optional| Description                                                              |
|---------------------|-----------------------|-------|------------------------------------------------------------------|
| model               | torch.nn.Module       | Mandatory   | `nn.Module` that contains `EmbeddingBagCollection` or `HashEmbeddingBagCollection`.|
| optimizer           | torch.optim.Optimizer | Mandatory   | Optimizer. For details about how to create an optimizer, see the usage example of the current API.         |
| device              | torch.device          | Mandatory   | Device. The value must be `torch.device("npu")`, that is, an NPU device.                               |
| execute_all_batches | bool                  | Optional   | The default value is `True`. Only the default value is supported.                                       |
| apply_jit           | bool                  | Optional   | The default value is `False`. Only the default value is supported.                                      |
| return_loss         | bool                  | Optional   | Indicates whether the `progress` method of the `HybridTrainPipelineSparseDist` object returns the loss. The default value is `False`. The loss is not returned.                                             |
| pipe_n_batch        | int                   | Optional   | Prefetches `n` batches for parallel processing. The value range is [1, 12]. The default value is 6.                                |

**Returns**

- Success: The pipeline is returned.
- Failure: An exception is thrown.

**Example**

Note: In this example, `ddp_model` is a model object created by calling the [DistributedModelParallel (TorchRec)](subtable_apis.md#TOPIC_0000002338384297) API, and its detailed definition is omitted here.

```python
import torch
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist

npu_device = torch.device("npu")
# ddp_model is a model object created by DistributedModelParallel (TorchRec), and its detailed definition is omitted here
ddp_model = ......
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adagrad(params, lr=0.01),
)
optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
pipeline = HybridTrainPipelineSparseDist(ddp_model, optimizer, npu_device)
```

### `progress`

**Description**

Runs the training pipeline table lookup, including forward propagation, backward propagation, and parameter updates.

**Function Prototype**

```python
def progress(self, dataloader_iter: Iterator[In]) -> Out:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|dataloader_iter|Iterator[In]|Mandatory|Dataset iterator. This iterator returns a `Batch` class used for table lookup and training. For details about the implementation, see the usage example of the current API.|

**Returns**

- Success: The model output is returned.
- Failure: A `StopIteration` or `RuntimeError` exception is thrown.

**Example**

Note: The detailed definitions of `Batch` and `RandomRecDataset` are omitted in this example. For details, see [dataset.py](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/little_demo/dataset.py).

```python
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchrec.streamable import Pipelineable
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist

# The detailed definitions of Batch and RandomRecDataset are omitted here
@dataclass
class Batch(Pipelineable):
    pass

class RandomRecDataset(IterableDataset[Batch]):
    pass

BATCH_SIZE = 32
BATCH_NUM = 32
FEAT_NAMES = ["feat0", "feat1", "feat2"]
ID_RANGES = [400, 4000, 400]

dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
dataloader = DataLoader(
    dataset,
    batch_size=None,
    batch_sampler=None,
    pin_memory=True,
    prefetch_factor=32,
    pin_memory_device="npu",
    num_workers=4,
)
dataloader_iter = iter(dataloader)

# For pipeline creation, see the preceding HybridTrainPipelineSparseDist initialization method. Its detailed definition is omitted here
pipeline: HybridTrainPipelineSparseDist = ......
# If return_loss is False when you create the HybridTrainPipelineSparseDist object:
output = pipeline.progress(dataloader_iter)
# If return_loss is True when you create the HybridTrainPipelineSparseDist object:
output, loss = pipeline.progress(dataloader_iter)
```

## `EmbCacheTrainPipelineSparseDist`

### Initialization

**Description**

Creates a distributed training data pipeline scheduler in multi-level cache mode.

**Function Prototype**

```python
class EmbCacheTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cpu_device: torch.device,
        npu_device: torch.device,
        return_loss: bool = False,
        evict_step_interval: Optional[int] = None,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[EmbCacheTrainPipelineContext] = EmbCacheTrainPipelineContext,
        # Keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        custom_model_zero_grad: Optional[Callable] = None,
        custom_model_bwd: Optional[Callable] = None,
    ) -> None:
```

**Parameters**

| Parameter                   | Type                                | Mandatory/Optional| Description                                                                      |
|------------------------|------------------------------------|-------|--------------------------------------------------------------------------|
| model                  | torch.nn.Module                    | Mandatory   | `nn.Module` object that contains `EmbCacheEmbeddingBagCollection` or `EmbCacheEmbeddingCollection`.|
| optimizer              | torch.optim.Optimizer              | Mandatory   | Optimizer.                                                                     |
| cpu_device             | torch.device                       | Mandatory   | CPU device.                                                                   |
| npu_device             | torch.device                       | Mandatory   | NPU device.                                                                   |
| return_loss            | bool                               | Optional   | Indicates whether the `progress` method of the `EmbCacheTrainPipelineSparseDist` object returns the loss. The default value is `False`. The loss is not returned.                                                      |
| evict_step_interval    | int                                | Optional   | Eviction step interval. The default value is `None`. This parameter must be passed when creating a sparse table and enabling the eviction function. If the parameter value is not `None`, it must be greater than or equal to 10. |
| execute_all_batches    | bool                               | Optional   | Indicates whether to execute all batches. The default value is `True`. Custom values are not supported.                                              |
| apply_jit              | bool                               | Optional   | Indicates whether to apply JIT compilation. The default value is `False`. Custom values are not supported.                                            |
| context_type           | Type[EmbCacheTrainPipelineContext] | Optional   | Context type. The default value is `EmbCacheTrainPipelineContext`.                                  |
| pipeline_postproc      | bool                               | Optional   | Indicates whether to enable pipeline post-processing. The default value is `False`. This behaves the same as `TrainPipelineSparseDist` of `torchrec`.                |
| custom_model_fwd       | Callable                           | Optional   | Custom model forward function. The default value is `None`. This behaves the same as `TrainPipelineSparseDist` of `torchrec`.                  |
| custom_model_zero_grad | Callable                           | Optional   | Custom `zero_grad` function. The default value is `None`.                                                  |
| custom_model_bwd       | Callable                           | Optional   | Custom model backward function. The default value is `None`.                                                       |

**Returns**

- Success: An `EmbCacheTrainPipelineSparseDist` object is returned.
- Failure: An exception is thrown.

**Example**

Note: In this example, `ddp_model` is a model object created by calling the [DistributedModelParallel (TorchRec)](subtable_apis.md#TOPIC_0000002338384297) API, and its detailed definition is omitted here. This API is a pipeline in multi-level cache mode. When you call the DistributedModelParallel API, you must pass the sparse tables created by the [EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection) and [EmbCacheEmbeddingBagCollection](./table_creation_apis.md#embcacheembeddingbagcollection) APIs that correspond to the multi-level cache mode.

```python
import torch
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist

cpu_device: torch.device = torch.device("cpu")
npu_device: torch.device = torch.device("npu")
# ddp_model is a model object created by DistributedModelParallel (TorchRec), and its detailed definition is omitted here
ddp_model = ......
dense_optimizer: KeyedOptimizerWrapper = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adagrad(params, lr=0.01),
)
optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
pipeline = EmbCacheTrainPipelineSparseDist(
    ddp_model, optimizer, cpu_device=cpu_device, npu_device=npu_device, execute_all_batches=True
)
```

### `progress`

**Description**

Runs the training pipeline table lookup, including forward propagation, backward propagation, and parameter updates.

**Function Prototype**

```python
def progress(self, dataloader_iter: Iterator[In]) -> Out:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|dataloader_iter|Iterator[In]|Mandatory|Dataset iterator. This iterator returns a `Batch` class used for table lookup and training. For details about the implementation, see the usage example of the current API.|

**Returns**

- Success: The training output result is returned.

- Failure: A `StopIteration` or `RuntimeError` exception is thrown.

**Example**

Note: The detailed definitions of `Batch` and `RandomRecDataset` are omitted in this example. For details, see [dataset.py](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/little_demo/dataset.py). For the complete usage of the pipeline, see [test](../../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_pipeline.py).

```python
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchrec.streamable import Pipelineable
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist

# The detailed definitions of Batch and RandomRecDataset are omitted here
@dataclass
class Batch(Pipelineable):
    pass

class RandomRecDataset(IterableDataset[Batch]):
    pass

BATCH_SIZE = 32
BATCH_NUM = 32
FEAT_NAMES = ["feat0", "feat1", "feat2"]
ID_RANGES = [400, 4000, 400]

dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
dataloader = DataLoader(
    dataset,
    batch_size=None,
    batch_sampler=None,
    pin_memory=True,
    prefetch_factor=32,
    pin_memory_device="npu",
    num_workers=4,
)
dataloader_iter = iter(dataloader)

# For pipeline creation, see the preceding EmbCacheTrainPipelineSparseDist initialization method. Its detailed definition is omitted here
pipeline: EmbCacheTrainPipelineSparseDist = ......
# If return_loss is False when you create the EmbCacheTrainPipelineSparseDist object:
output = pipeline.progress(dataloader_iter)
# If return_loss is True when you create the EmbCacheTrainPipelineSparseDist object:
output, loss = pipeline.progress(dataloader_iter)
```
