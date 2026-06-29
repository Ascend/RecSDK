# Multi-Level Cache Management APIs

## `InitializerType`

**Description**

An enumeration of weight initialization types. It defines how to initialize embedding table weights.

**Function Prototype**

```python
class InitializerType(Enum):
    LINEAR ="linear"
    TRUNCATED_NORMAL ="truncated_normal"
    UNIFORM = "uniform"
```

**Parameters**

|Parameter|Description|
|--|--|
|LINEAR|Linear initialization.|
|TRUNCATED_NORMAL|Truncated normal initialization.|
|UNIFORM|Uniform initialization.|

**Returns**

- Success: An enumerated value of `InitializerType` is returned.
- Failure: An exception is thrown.

## Saver

**Description**

A utility class for saving and loading multilevel cache sparse tables. It provides save and load APIs for multilevel cache sparse table data, such as sparse table embeddings and optimizer parameters for those embeddings.

**Function Prototype**

```python
class Saver:
    def __init__(self, rank: int = None):
    ...
    def save(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:
    ...
    def load(self, module: torch.nn.Module, path: str, incremental: bool = False) -> None:
```

**Constraints**

1. The saving and loading APIs support only multilevel cache saving and loading of sparse table data, such as sparse table embeddings and optimizer parameters for the embeddings.
2. Dense data cannot be saved or loaded. You must call the native Torch APIs.
3. Saving and loading sparse tables in full-device memory mode is not supported.
4. The saving and loading APIs support only the local file system.
5. Incremental saving and loading does not support admission and eviction. If you enable both at the same time, configuration validation fails.
6. Incremental saving and loading applies only to incremental data generated in pipeline training mode.
7. To use incremental saving and loading, add the `is_incremental` parameter to `EmbCacheEmbeddingBagConfig` or `EmbCacheEmbeddingConfig` when you create the table. See [Table Creation APIs](table_creation_apis.md#embcacheembeddingbagconfig).
8. Differential card loading does not support admission and eviction. If you enable both at the same time, configuration validation fails.
9. In pipeline mode, if you need to save during training, that is, before the `Dataset` completes an iteration, you must manually trigger `wait_pipeline_compute_swapinfo` before saving.
    For details, see [saving and loading example](../../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_save_and_load.py).

**Parameters**

|**Parameter** |**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|rank|int|Optional|Rank of the current process in the full `world_size`. When the PyTorch distributed environment is already initialized, this parameter is optional. In that case, `torch.distributed.get_rank()` gets the rank. Otherwise, this parameter is required.|
|module|torch.nn.Module|Mandatory|Model object instance. The model, or one of its submodules, must include an instance of `EmbCacheShardedEmbeddingBagCollection` or `EmbCacheShardedEmbeddingCollection`, and the nesting depth must not exceed 500. This requirement is met when you create and shard the model with the table creation and sharding APIs supported by multi-level cache.|
|path|string|Mandatory|Path for saving and loading. The value length range is [1,1024]. <div><div>[!NOTICE]Note</div><div>The path for saving and loading cannot contain symbolic links or sensitive strings (`Key`, `password`, `privatekey`). Do not use special paths, such as paths in `/usr`, and keep the permissions on the path no higher than 750.</div></div>|
|incremental|bool|Optional|Indicates whether to enable incremental saving and loading. The default value is `False`. Incremental saving and loading applies to incremental data generated in pipeline training mode. To use this feature, add the `is_incremental` parameter to `EmbCacheEmbeddingBagConfig` or `EmbCacheEmbeddingConfig` when you create the table. See [Table Creation APIs](table_creation_apis.md#embcacheembeddingbagconfig).|

**Returns**

- Success: The API call completes without errors and saves or loads the sparse table data.
- Failure: An exception is thrown.

**Example**

```python
from torchrec_embcache.saver import Saver
...
saver = Saver(rank=rank)
saver.save(model, "save_dir/sparse") # Save.
saver.load(model, "save_dir/sparse") # Load.
```
