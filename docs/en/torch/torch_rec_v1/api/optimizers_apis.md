# Optimizer APIs

## `apply_optimizer_in_backward` (TorchRec)

> [!NOTICE]
>
> This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Specifies the optimizer used by a table.

**Function Prototype**

```python
def apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
) -> None:
```

**Parameters**

| Parameter        | Type                          | Mandatory/Optional| Description |
|------------------|------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optimizer_class  | Type[torch.optim.Optimizer]  | Mandatory   | Optimizer type. Value range:<ul><li>`torch.optim.Adagrad`: Adagrad optimizer </li><li>`torch.optim.Adam`: Adam optimizer </li><li>`torch.optim.SGD`: SGD optimizer </li><li>`torchrec.optim.Adagrad`: Adagrad optimizer </li><li>`torchrec.optim.Adam`: Adam optimizer </li><li>`torchrec.optim.SGD`: SGD optimizer </li><li>`torchrec.optim.AccumulateAdagrad`: Adagrad optimizer with gradient accumulation </li><li>`torchrec.optim.AccumulateAdam`: Adam optimizer with gradient accumulation </li><li>`torchrec.optim.AccumulateSGD`: SGD optimizer with gradient accumulation </li></ul>Note: <ul><li>The optimizer type must be consistent with the `embedding_optimizer_cls` parameter used when you create a sparse table. This applies only to [`EmbCacheEmbeddingCollection`](./table_creation_apis.md#embcacheembeddingcollection) and [`EmbCacheEmbeddingBagCollection`](./table_creation_apis.md#embcacheembeddingbagcollection). </li><li>Gradient accumulation currently supports only the table creation API for [`EmbCacheEmbeddingCollection`](./table_creation_apis.md#embcacheembeddingcollection). For details about how to use gradient accumulation optimizers, see the [example](https://atomgit.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_aggregation.py). </li><li>When you use `torchrec.optim.AccumulateSGD`, you are advised to keep the number of aggregated items per embedding at no more than 10,000. If you exceed that limit, the calculation accuracy may fail to meet the dual 0.0001 accuracy criterion.</li></ul> |
| params           | Iterable[torch.nn.Parameter] | Mandatory   | `torch.nn.Parameter` objects used to configure the optimizer. Pass the parameters of a [`HashEmbeddingBagCollection`](./table_creation_apis.md#hashembeddingbagcollection), [`EmbCacheEmbeddingCollection`](./table_creation_apis.md#embcacheembeddingcollection), or [`EmbCacheEmbeddingBagCollection`](./table_creation_apis.md#embcacheembeddingbagcollection) object.<br>Note:<br>For performance reasons, the `params` parameter is not validated. You need to ensure that the type is correct.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| optimizer_kwargs | Dict[str, Any]               | Mandatory   | Set this parameter according to the parameters and valid range of `optimizer_class`, and make sure the values stay within the corresponding optimizer limits. <p>If you use gradient accumulation, set `optimizer_class` to an optimizer that supports gradient accumulation, and pass `use_accumulate` (Boolean type, defaulting to `False`) as `True` and `accumulate_step` (integer type, defaulting to 1) to specify the number of accumulation steps.</p>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
import torchrec
from hybrid_torchrec.modules.hash_embeddingbag import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward

def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


embedding_dims: list[int] = [64, 64, 64]
num_embeddings: list[int] = [400, 4000, 400]
table_num: int = len(num_embeddings)
embedding_configs: list[HashEmbeddingBagConfig] = []
table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
feat_names: list[str] = [f"feat{i}" for i in range(table_num)]
for i in range(table_num):
    emb_config = HashEmbeddingBagConfig(
        name=table_names[i],
        embedding_dim=embedding_dims[i],
        num_embeddings=num_embeddings[i],
        feature_names=[feat_names[i]],
        pooling=torchrec.PoolingType.MEAN,
        init_fn=weight_init,  # type: ignore
    )
    embedding_configs.append(emb_config)
# Take HashEmbeddingBagCollection as an example to obtain its parameters and pass them to the apply_optimizer_in_backward API
sparse_ebc: HashEmbeddingBagCollection = HashEmbeddingBagCollection(device="meta", tables=embedding_configs)
embedding_optimizer = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.001, "eps": 1e-5}
apply_optimizer_in_backward(
    embedding_optimizer,
    sparse_ebc.parameters(),
    optimizer_kwargs=optimizer_kwargs
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `in_backward_optimizer_filter` (TorchRec)

> [!NOTICE]
>
> This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Filters out parameters designated as `backward_optimizer`.

**Function Prototype**

```python
def in_backward_optimizer_filter(
    named_parameters: Iterator[Tuple[str, nn.Parameter]], include: bool = False
) -> Iterator[Tuple[str, nn.Parameter]]:
```

**Parameters**

| Parameter             | Type                                | Mandatory/Optional| Description                                                                                                      |
|------------------|------------------------------------|-------|----------------------------------------------------------------------------------------------------------|
| named_parameters | Iterator[Tuple[str, nn.Parameter]] | Mandatory   | Model parameter list. Make sure this variable comes from `torch.nn.Module.named_parameters()`.                                              |
| include          | bool                               | Optional   | If `include` is `True`, the returned result includes parameter names marked as `backward_optimizer`. Otherwise, it does not. The default value is `False`. <ul><li>`True`: Includes.</li><li>`False`: Does not include.</li></ul>|

**Returns**

- Success: The filtered parameters are returned.
- Failure: An exception is thrown.

**Example**

Note: In the example, `ddp_model` is a model object created by calling the [`DistributedModelParallel` (TorchRec)](subtable_apis.md#TOPIC_0000002338384297) API. Its detailed definition is omitted here.

```python
from torchrec.optim.optimizers import in_backward_optimizer_filter

# ddp_model is a model object created by DistributedModelParallel (TorchRec). Its detailed definition is omitted here
ddp_model = ......
bw_optimizer_iter = in_backward_optimizer_filter(ddp_model.named_parameters())
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `KeyedOptimizerWrapper` (TorchRec)

> [!NOTICE]
>
> This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Wraps an optimizer after table parameters are filtered out.

**Function Prototype**

```python
class KeyedOptimizerWrapper:
    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        optim_factory: OptimizerFactory,
    ) -> None:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|params|Mapping[str, Union[torch.Tensor, ShardedTensor]]|Mandatory|Parameter list. Make sure you obtain it through [`in_backward_optimizer_filter` (TorchRec)](#in_backward_optimizer_filter-torchrec).|
|optim_factory|Callable|Mandatory|Function that creates the optimizer. Make sure the function takes one parameter as input and returns a `torch.optim.Optimizer` object.|

**Returns**

- Success: An optimizer with filtered parameters is returned.
- Failure: An exception is thrown.

**Example**

Note: In the example, `ddp_model` is a model object created by calling the [`DistributedModelParallel` (TorchRec)](subtable_apis.md#TOPIC_0000002338384297) API. Its detailed definition is omitted here.

```python
import torch
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

# ddp_model is a model object created by DistributedModelParallel (TorchRec). Its detailed definition is omitted here
ddp_model = ......
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adam(params, lr=1e-1),
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `CombinedOptimizer` (TorchRec)

> [!NOTICE]
>
> This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Combines multiple optimizers into one.

**Function Prototype**

```python
class CombinedOptimizer(KeyedOptimizer):
    def __init__(
        self, optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]]
    ) -> None:
```

**Parameters**

| Parameter   | Type                                                                             | Mandatory/Optional| Description                                                                                                                                                           |
|--------|---------------------------------------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optims | <ul><li>List[KeyedOptimizer]</li><li>List[Tuple[str, KeyedOptimizer]]</li></ul> | Mandatory   | List of optimizer objects to combine. The list can be empty.<br>Ensure that the input value comes from `CombinedOptimizer` (TorchRec) or the `fused_optimizer` of [`DistributedModelParallel` (TorchRec)](./subtable_apis.md#TOPIC_0000002338384297).<br>Note: <ul><li>TorchRec v1.1.0 supports only the `List[KeyedOptimizer]` type, not the `List[Tuple[str, KeyedOptimizer]]` type.</li><li>TorchRec v1.2.0 and v1.5.0 support both the `List[KeyedOptimizer]` and `List[Tuple[str, KeyedOptimizer]]` types.</li></ul>|

**Returns**

- Success: A combined optimizer is returned.
- Failure: An exception is thrown.

**Example**

Note: In the example, `ddp_model` is a model object created by calling the [`DistributedModelParallel` (TorchRec)](subtable_apis.md#TOPIC_0000002338384297) API. Its detailed definition is omitted here.

```python
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


ddp_model = ......
dense_optimizer: KeyedOptimizerWrapper = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adagrad(params, lr=0.1),
)
optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
