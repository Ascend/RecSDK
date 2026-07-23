# Pipeline APIs

## `HybridTrainPipelineSparseDist`

### Initialization

**Description**

Creates a full-device memory mode pipeline for table lookups.

**Function Prototype**

```python
class HybridTrainPipelineSparseDist:
    def __init__(**kwargs):
```

**Parameters**

| Parameter         | Type                   | Mandatory/Optional| Description                                                              |
|---------------------|-----------------------|-------|------------------------------------------------------------------|
| model               | torch.nn.Module       | Mandatory   | An `nn.Module` that contains `EmbeddingBagCollection` or `HashEmbeddingBagCollection`.|
| optimizer           | torch.optim.Optimizer | Mandatory   | Optimizer. For details about how to create an optimizer, see the optimizer defined in [Step 7](../quick_start.md#interface-call-introduction).         |
| device              | torch.device          | Mandatory   | Device. The value must be `torch.device("npu")`, that is, an NPU device.                               |
| return_loss         | bool                  | Optional   | Indicates whether to return the loss. The default value is `False`.                                             |
| pipe_n_batch        | int                   | Optional   | Prefetches `n` batches for parallel processing. The value range is [1, 12]. The default value is 6.                                |
| execute_all_batches | bool                  | Optional   | The default value is `True`. Only the default value is supported.                                       |
| apply_jit           | bool                  | Optional   | The default value is `False`. Only the default value is supported.                                      |

**Returns**

- Success: The pipeline is returned.
- Failure: An exception is thrown.

**Example**

```python
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
pipeline = HybridTrainPipelineSparseDist(model, optimizer, device)
```

### progress

**Description**

Runs pipeline training.

**Function Prototype**

```python
def progress(dataloader_iter: Iterator[In]) -> Out:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|dataloader_iter|Iterator[In]|Mandatory|Dataset iterator. This iterator returns a `Batch` class used for table lookup and training. See [Step 1](../quick_start.md#interface-call-introduction).|

**Returns**

- Success: The model output is returned.
- Failure: An exception is thrown.

**Example**

```python
# If return_loss is False when you create the HybridTrainPipelineSparseDist object:
output = pipeline.progress(dataloader_iter)
# If return_loss is True when you create the HybridTrainPipelineSparseDist object:
output, loss = pipeline.progress(dataloader_iter)
```

## `EmbCacheTrainPipelineSparseDist`

### Initialization

**Description**

Creates a multi-level cache pipeline for table lookups.

**Function Prototype**

```python
class EmbCacheTrainPipelineSparseDist:
     def __init__(**kwargs):
```

**Parameters**

| Parameter                   | Type                                | Mandatory/Optional| Description                                                                      |
|------------------------|------------------------------------|-------|--------------------------------------------------------------------------|
| model                  | torch.nn.Module                    | Mandatory   | An `nn.Module` object that contains `EmbCacheEmbeddingBagCollection` or `EmbCacheEmbeddingCollection`.|
| optimizer              | torch.optim.Optimizer              | Mandatory   | Optimizer.                                                                     |
| cpu_device             | torch.device                       | Mandatory   | CPU device.                                                                   |
| npu_device             | torch.device                       | Mandatory   | NPU device.                                                                   |
| return_loss            | bool                               | Optional   | Indicates whether to return the loss. The default value is `False`.                                                      |
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

```python
import torch
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist
cpu_device: torch.device = torch.device("cpu")
optimizer = xx
pipeline = EmbCacheTrainPipelineSparseDist(
    model, optimizer, cpu_device=cpu_device, npu_device=device, execute_all_batches=True
)
```

### `progress`

**Description**

Runs one step of the training pipeline, including forward propagation, backward propagation, and parameter updates.

**Function Prototype**

```python
def progress(self, dataloader_iter: Iterator[In]) -> Out:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|dataloader_iter|Iterator[In]|Mandatory|Data loader iterator|

**Returns**

- Success: The training output result is returned.

- Failure: A `StopIteration` or `RuntimeError` exception is thrown.

**Example**

```python
# If return_loss is False when you create the EmbCacheTrainPipelineSparseDist object:
output = pipeline.progress(dataloader_iter)
# If return_loss is True when you create the EmbCacheTrainPipelineSparseDist object:
output, loss = pipeline.progress(dataloader_iter)
```
