# Optimizer APIs

## `apply_optimizer_in_backward` (TorchRec)

>[!NOTICE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

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

**Parameters**<a id="section888634319218"></a>

| Parameter        | Type                          | Mandatory/Optional| Description |
|------------------|------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optimizer_class  | Type[torch.optim.Optimizer]  | Mandatory   | Optimizer type. Value range:<ul><li>`torch.optim.Adagrad`: Adagrad optimizer </li><li>`torch.optim.Adam`: Adam optimizer </li><li>`torch.optim.SGD`: SGD optimizer </li><li>`torchrec.optim.AccumulateAdagrad`: Adagrad optimizer with gradient accumulation </li><li>`torchrec.optim.AccumulateAdam`: Adam optimizer with gradient accumulation </li><li>`torchrec.optim.AccumulateSGD`: SGD optimizer with gradient accumulation </li></ul>Note: <ul><li>The optimizer type must match the configuration used when the table is created. Gradient accumulation currently supports only the table creation API for [`EmbCacheEmbeddingCollection`](./table_creation_apis.md#embcacheembeddingcollection). For details about how to use gradient accumulation optimizers, see the [example](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_aggregation.py). </li><li>When you use `torchrec.optim.AccumulateSGD`, you are advised to keep the number of aggregated items per embedding at no more than 10,000. If you exceed that limit, the calculation accuracy may fail to meet the dual 0.0001 accuracy criterion.</li></ul> |
| params           | Iterable[torch.nn.Parameter] | Mandatory   | `torch.nn.Parameter` objects used to configure the optimizer. Pass the parameters of a `HashEmbeddingBagCollection`, `EmbCacheEmbeddingCollection`, or `EmbCacheEmbeddingBagCollection` object as described in [Step 5](../quick_start.md#interface-call-introduction).<br>Note:<br>For performance reasons, the `params` parameter cannot be validated. You need to ensure that the type is correct.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| optimizer_kwargs | Dict[str, Any]               | Mandatory   | Set this parameter according to the parameters and valid range of `optimizer_class`, and make sure the values stay within the corresponding optimizer limits. <p>If you use gradient accumulation, set `optimizer_class` to an optimizer that supports gradient accumulation, and pass `use_accumulate` (Boolean type, defaulting to `False`) as `True` and `accumulate_step` (integer type, defaulting to 1) to specify the number of accumulation steps.</p>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
import torch
embedding_optimizer = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.001, "eps": 0.1}
apply_optimizer_in_backward(
    embedding_optimizer,
    ebc.parameters(),
    optimizer_kwargs=optimizer_kwargs)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `in_backward_optimizer_filter` (TorchRec)

>[!NOTICE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

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

```python
from torchrec.optim.optimizers import in_backward_optimizer_filter
parameter = in_backward_optimizer_filter(model.named_parameters())
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `KeyedOptimizerWrapper` (TorchRec)

>[!NOTICE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Wraps an optimizer after table parameters are filtered out.

**Function Prototype**

```python
class KeyedOptimizerWrapper:
    def __init__(**kwargs)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|params|Mapping[str, Union[torch.Tensor, ShardedTensor]]|Mandatory|Parameter list. Make sure you obtain it through [`in_backward_optimizer_filter` (TorchRec)](#in_backward_optimizer_filter-torchrec).|
|optim_factory|Callable|Mandatory|Function that creates the optimizer. Make sure the function takes one parameter as input and returns a `torch.optim.Optimizer` object.|

**Returns**

- Success: An optimizer with filtered parameters are returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec.optim.keyed import KeyedOptimizerWrapper
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(model.named_parameters())),
    lambda params: torch.optim.Adam(params, lr=1e-1),
)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## CombinedOptimizer (TorchRec)

>[!NOTICE]
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Combines multiple optimizers into one.

**Function Prototype**

```python
class CombinedOptimizer:
    def __init__(**kwargs)
```

**Parameters**

| Parameter   | Type                                                                             | Mandatory/Optional| Description                                                                                                                                                           |
|--------|---------------------------------------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optims | <ul><li>List[KeyedOptimizer]</li><li>List[Tuple[str, KeyedOptimizer]]</li></ul> | Mandatory   | List of optimizer objects to combine. The list can be empty.<br>Make sure the input value comes from `CombinedOptimizer` (TorchRec) or [`DistributedModelParallel` (TorchRec)](./subtable_apis.md#distributedmodelparallel-torchrec).|

**Returns**

- Success: A combined optimizer is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec.optim.keyed import CombinedOptimizer
optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
