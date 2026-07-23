# Data APIs

## `JaggedTensor` (TorchRec)

>[!NOTICE]
>
>This API is a TorchRec open-source API, not an external Rec SDK Torch API. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Class that stores sparse IDs and feature lengths for table lookups. For example, if `values` are `[id1, id2, id3, id4]` and `lengths` are `[1, 2, 1]`, the embedding for `id2` and `id3` should be pooled.

**Function Prototype**

```python
class JaggedTensor:
 def __init__(**kwargs):
```

**Parameters**

|Parameter|Type| Mandatory/Optional| Description                                                                                                              |
|--|--|------|-------------------------------------------------------------------------------------------------------------------------------------------|
|values|torch.Tensor[int64]| Mandatory  | Sparse table lookup IDs.                                                                                                                                 |
|weights|torch.Tensor| Optional  | When you use an NPU device, only the default value `None` is supported.                                                                                                           |
|lengths|torch.Tensor[int64]| Optional  | Length of the feature sequence in each sample. When you use an NPU device, this field is required. The value range is [1, 10000]. Make sure the sum of `lengths` equals the length of `values`. Rec SDK Torch does not support variable batch sizes currently. The `lengths` of all `JaggedTensor` objects in a training job must have the same length.|
|offsets|torch.Tensor[int64]| Optional  | Cumulative sum of `lengths`. The first element of `offsets` is `0`, and the remaining elements are the cumulative sums of `lengths`. The default value is `None`. You must validate the correctness of `offsets`.                                                          |

**Example**

```python
from torchrec import JaggedTensor
JaggedTensor(values=[1, 3, 4], lengths=[1, 1, 1], offsets=[0, 1, 2, 3])
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `KeyedJaggedTensor` (TorchRec)

### `from\_jt\_dict`

>[!NOTICE]
>
>The APIs in this class are TorchRec open-source APIs, not external Rec SDK Torch APIs. This section describes the parameter ranges supported by the TorchRec API when you use Rec SDK Torch.

**Description**

Creates a `KeyedJaggedTensor` from a dictionary of `JaggedTensor`.

**Function Prototype**

```python
def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor"
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
from torchrec import KeyedJaggedTensor, JaggedTensor
jt = JaggedTensor(values=[1, 3, 4], lengths=[1, 1, 1])
kjt = KeyedJaggedTensor.from_jt_dict({"feat0": jt})
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
