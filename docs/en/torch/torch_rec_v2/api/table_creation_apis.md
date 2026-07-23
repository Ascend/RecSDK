# Table Creation APIs

## `EmbeddingConfig`

>[!NOTE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Input to `EmbeddingCollection`. Use it to configure table size, `dim`, data type, and other settings.

**Function Prototype**

```python
@dataclass
class EmbeddingConfig:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|----|--|
|num_embeddings|int|Mandatory|Number of rows in the sparse table. The value ranges from 1 to 1 billion. The minimum value must be greater than or equal to the number of used cards.|
|embedding_dim|int|Mandatory|Number of columns in the sparse table. The value range is [8, 4096]. The value must be a multiple of 8.|
|name|str|Mandatory|Name of the sparse table. The value can contain only numbers, letters, and underscores (_). The length of the value must be in the range of [1,4096].|
|data_type|torchrec.types.DataType|Optional|Data type of the sparse table. Only the default value `DataType.FP32` is supported.|
|feature_names|List[str]|Mandatory|Names of the features queried by the sparse table. The value can contain only numbers, letters, and underscores (_).|
|weight_init_max|float|Optional|Maximum value for weight initialization. Only the default value `None` is supported.|
|weight_init_min|float|Optional|Minimum value for weight initialization. Only the default value `None` is supported.|
|num_embeddings_post_pruning|int|Optional|Number of sparse tables after inference pruning. Only the default value `None` is supported.|
|init_fn|Callable|Optional|Initialization function. A function of the `nn.Parameter` type can be passed. You need to ensure that the function is correct. The default value is `None`.|
|need_pos|bool|Optional|Position weight. Only the default value `False` is supported.|

**Example**

```python
ec_configs = [
    EmbeddingConfig(
    name="table_name",
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    feature_names=["user_id"],
    ),
    ···
    ]
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `EmbeddingCollection`

>[!NOTE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Creates a single-node table object.

**Function Prototype**

```python
class EmbeddingCollection:
    def __init__(**kwargs):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|tables|List[EmbeddingConfig]|Mandatory|List of sparse table configuration files. <p>See `EmbeddingConfig` for parameter details.</p>|
|device|Optional[torch.device]|Optional|Device for the sparse table. The default value is `torch.device("cpu")`.<br>If the type is `torch.device`, supported values are:<ul><li>`torch.device("npu")`: NPU device. </li><li>`torch.device("meta")`: Meta device. </li><li>`torch.device("cpu")`: CPU device. The CPU device does not support distributed tables and supports only single-node tables.</li></ul>|
|need_indices|bool|Optional|Indicates whether indexes are required. The default value is `False`.|

**Example**

```python
ec = EmbeddingCollection(device="npu", tables=table_configs)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
