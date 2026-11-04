# Data APIs

## `JaggedTensor` (TorchRec)

> [!NOTICE]
>
> This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Class that stores sparse IDs and feature lengths for table lookups. For example, if `values` are `[id1, id2, id3, id4]` and `lengths` are `[1, 2, 1]`, the embedding for `id2` and `id3` should be pooled.

**Function Prototype**

```python
class JaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                                                                              |
|--|--|------|-------------------------------------------------------------------------------------------------------------------------------------------|
|values|torch.Tensor| Mandatory  | Sparse table lookup IDs.                                                                                                                                 |
|weights|torch.Tensor| Optional  | When you use an NPU device, only the default value `None` is supported.                                                                                                           |
|lengths|torch.Tensor| Optional  | Length of the feature sequence in each sample. The default value is `None`. When `lengths` is not `None`, ensure that the sum of `lengths` equals the length of `values`. Rec SDK Torch does not support variable batch sizes currently. The `lengths` of all `JaggedTensor` objects in a training job must have the same length.|
|offsets|torch.Tensor| Optional  | Cumulative sum of `lengths`. The default value is `None`. When `offsets` is not `None`, the first element of `offsets` must be `0`, and the remaining elements are the cumulative sums of `lengths`. You must validate the correctness of `offsets`. `lengths` and `offsets` cannot both be `None`.                                                          |

**Returns**

- Success: A `JaggedTensor` object is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
from torchrec import JaggedTensor

values = torch.tensor([1, 3, 4])
lengths = torch.tensor([1, 1, 1])
offsets = torch.tensor([0, 1, 2, 3])
jagged_tensor = JaggedTensor(values=values, lengths=lengths, offsets=offsets)
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).

## `KeyedJaggedTensor` (TorchRec)

### `from_jt_dict`

> [!NOTICE]
>
> The APIs in this class are TorchRec open-source APIs, not external Rec SDK Torch APIs. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Creates a `KeyedJaggedTensor` from a `JaggedTensor`.

**Function Prototype**

```python
@staticmethod
def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|jt_dict|Dict[str, JaggedTensor]|Mandatory|Dictionary consisting of feature names and their corresponding `JaggedTensor` objects. The length cannot be 0. For the value range of `JaggedTensor`, see [`JaggedTensor` (TorchRec)](#jaggedtensor-torchrec).|

**Returns**

- Success: A `KeyedJaggedTensor` object is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
from torchrec import KeyedJaggedTensor, JaggedTensor

values = torch.tensor([1, 3, 4])
lengths = torch.tensor([1, 1, 1])
offsets = torch.tensor([0, 1, 2, 3])
jt = JaggedTensor(values=values, lengths=lengths, offsets=offsets)
kjt = KeyedJaggedTensor.from_jt_dict({"feat0": jt})
```

**References**

For the interface calling process and examples, see [Migration and Training](../04_migration_and_training/migration_and_training.md).
