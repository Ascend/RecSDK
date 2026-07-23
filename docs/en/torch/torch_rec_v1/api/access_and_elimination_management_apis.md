# Admission and Eviction Management APIs

## `AdmitAndEvictPolicyType`

**Description**

Enumerates admission and eviction policy types that defines the policy type for embedding table feature admission and eviction.

**Function Prototype**

```python
class AdmitAndEvictPolicyType(Enum):
    NONE = 0
    POLICY_COUNT = 1
    POLICY_SHOWCLICK = 2
```

**Parameters**

| Parameter           | Description                                                       |
|------------------|-----------------------------------------------------------|
| NONE             | No admission or eviction policy.                                                 |
| POLICY_COUNT     | Admission and eviction policy based on count and timestamp. Feature admission depends on the repeat count, and feature eviction depends on a time threshold.|
| POLICY_SHOWCLICK | Admission and eviction policy based on show and click events. Feature admission and eviction depend on a weighted score of show and click counts.       |

**Returns**

- Success: An enumerated value of `AdmitAndEvictPolicyType` is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec_embcache.distributed.configs import AdmitAndEvictPolicyType
policy = AdmitAndEvictPolicyType.POLICY_COUNT
```

## `ShowClickParams`

**Description**

Defines the parameters for the show/click policy. The class provides parameters for configuring show/click admission and eviction, allowing you to control feature admission and eviction behavior based on show and click counts.

**Function Prototype**

```python
@dataclass
class ShowClickParams:
    alpha: float = 1.0
    beta: float = 0.0
    admit_threshold: float = 0.0
    evict_percentage: float = 0.0
    score_decay: float = 1.0
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|alpha|float|Optional|Weight coefficient for show count, used to calculate admission and eviction scores. The default value is 1.0. Admission score formula: `score = alpha * showCount + beta * clickCount`.|
|beta|float|Optional|Weight coefficient for click count, used to calculate admission and eviction scores. The default value is 0.0. Admission score formula: `score = alpha * showCount + beta * clickCount`.|
|admit_threshold|float|Optional|Feature admission threshold. When admission is enabled, features with scores lower than this value are discarded. A value greater than 0 enables admission. The default value is 0.0, indicating that feature admission is disabled.|
|evict_percentage|float|Optional|Feature eviction ratio. When eviction is enabled, features with smaller scores that fall within this ratio are evicted. A value greater than 0 enables eviction. The default value is 0.0, indicating that feature eviction is disabled. The value range is [0.0, 1.0].|
|score_decay|float|Optional|Score decay coefficient, used to calculate and update eviction scores. The value range is [0.0, 1.0]. The default value is 1.0, indicating no decay. A value of 0.0 indicates full decay. Eviction score formula: `newScore = (oldScore + alpha * showCount + beta * clickCount) * scoreDecay`.|

**Returns**

- Success: A `ShowClickParams` object is returned.
- Failure: An exception is thrown.

**Example**

```python
from torchrec_embcache.distributed.configs import ShowClickParams
showclick_params = ShowClickParams(alpha=1, beta=1, admit_threshold=0.1, evict_percentage=0.1, score_decay=0.9)
```

## `AdmitAndEvictConfig`

**Description**

Defines the admission and eviction configuration for a single embedding table. The class provides parameters for configuring feature admission and eviction under specific conditions. It supports two policy types: the count-based policy (`POLICY_COUNT`) and the show/click-based policy (`POLICY_SHOWCLICK`).

**Function Prototype**

```python
@dataclass
class AdmitAndEvictConfig:
    admit_threshold: Optional[int] = _DEFAULT_ADMIT_THRESHOLD
    not_admitted_default_value: Optional[float] = 0.0
    evict_threshold: Optional[int] = _DEFAULT_EVICT_THRESHOLD  # unit: seconds
    evict_step_interval: Optional[int] = 0
    showclick_params: ShowClickParams = field(default_factory=lambda: ShowClickParams())
    policy_type: AdmitAndEvictPolicyType = AdmitAndEvictPolicyType.POLICY_COUNT
```

**Parameters**

When `policy_type` is `POLICY_COUNT`:

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|admit_threshold|int|Optional|Feature admission threshold. After input distribution, a feature is admitted when its repeat count is greater than `admit_threshold`. The default value is -1, indicating that feature admission is disabled.|
|not_admitted_default_value|float|Optional|Embedding value for feature IDs that are not admitted. The default value is 0.0. This parameter takes effect only when `admit_threshold` is set to a non-default value.|
|evict_threshold|int|Optional|Feature eviction threshold, in seconds. The default value is 0, indicating that feature eviction is disabled.|
|evict_step_interval|int|Optional|Step interval for feature eviction. The default value is 0. This parameter takes effect only when `evict_threshold` is set to a non-default value.|

When `policy_type` is `POLICY_SHOWCLICK`:

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|not_admitted_default_value|float|Optional|Embedding value for feature IDs that are not admitted. The default value is 0.0.|
|evict_step_interval|int|Optional|Step interval for feature eviction. The default value is 0.|
|showclick_params|ShowClickParams|Optional|Parameter settings for the show/click policy. See [ShowClickParams](#showclickparams) for details.|

**Constraints**

The `POLICY_COUNT` and `POLICY_SHOWCLICK` policies cannot be used together.

**Returns**

- Success: An `AdmitAndEvictConfig` object is returned.
- Failure: An exception is thrown.

**Example**

When `policy_type` is `POLICY_COUNT`:

```python
from torchrec_embcache.distributed.configs import AdmitAndEvictConfig
admit_and_evict_config = AdmitAndEvictConfig(
    admit_threshold=2,
    not_admitted_default_value=0.999,
    evict_threshold=2000_0000,
    evict_step_interval=evict_step_interval,
)
```

When `policy_type` is `POLICY_SHOWCLICK`:

```python
from torchrec_embcache.distributed.configs import AdmitAndEvictConfig, AdmitAndEvictPolicyType, ShowClickParams
showclick_params = ShowClickParams(alpha=1, beta=1, admit_threshold=0.1, evict_percentage=0.1, score_decay=0.9)
admit_and_evict_config = AdmitAndEvictConfig(
    showclick_params=showclick_params,
    not_admitted_default_value=0.999,
    evict_step_interval=evict_step_interval,
    policy_type=AdmitAndEvictPolicyType.POLICY_SHOWCLICK,  # type: ignore
)
```

## `JaggedTensorWithTimestamp`

**Description**

Extends `JaggedTensor` and represents a jagged tensor with timestamp information. It adds a `_timestamps` attribute to `JaggedTensor` to store the timestamp information that corresponds to `values`. It is used to calculate time during feature eviction.

**Function Prototype**

```python
class JaggedTensorWithTimestamp(ExtendedJaggedTensor):
    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> None:
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|values|torch.Tensor|Mandatory|Values of the jagged tensor.|
|weights|torch.Tensor|Optional|Weight of each value. The default value is `None`.|
|lengths|torch.Tensor|Optional|Length of each sample. The default value is `None`.|
|offsets|torch.Tensor|Optional|Starting offset of each sample. The default value is `None`.|
|timestamps|torch.Tensor|Optional|Timestamp information that corresponds to `values`. The default value is `None`.|

**Returns**

- Success: A `JaggedTensorWithTimestamp` object is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
from torchrec_embcache.sparse.jagged_tensor_with_timestamp import (
    JaggedTensorWithTimestamp,
)
id_range = 10000
lookup_lens = 100
start_time = 1745897370
end_time = 1777433370
ids = torch.randint(0, id_range, (lookup_lens,))
timestamp_data = torch.randint(start_time, end_time, ids.size(), dtype=torch.int64)
lengths = torch.ones(lookup_lens).long()
jagged_tensor_with_ts = JaggedTensorWithTimestamp(values=ids, lengths=lengths, timestamps=timestamp_data)
```

## `KeyedJaggedTensorWithTimestamp`

**Description**

Extends `KeyedJaggedTensor` and represents a keyed jagged tensor with timestamp information. It adds a `_timestamps` attribute to `KeyedJaggedTensor` to store the timestamp information that corresponds to `values`. It is used to calculate time during feature eviction.

**Function Prototype**

```python
def from_jt_dict(jt_dict: Dict[str, JaggedTensorWithTimestamp],) -> "KeyedJaggedTensorWithTimestamp"
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|jt_dict|Dict[str, JaggedTensorWithTimestamp]|Mandatory|Dictionary composed of feature names and the corresponding `JaggedTensorWithTimestamp` objects. The length cannot be 0. For the value range of `JaggedTensorWithTimestamp`, see [JaggedTensor (TorchRec)](./data_apis.md).|

**Returns**

- Success: A `KeyedJaggedTensorWithTimestamp` object is returned.
- Failure: An exception is thrown.

**Example**

```python
import torch
from torchrec_embcache.sparse.jagged_tensor_with_timestamp import (
    JaggedTensorWithTimestamp,
    KeyedJaggedTensorWithTimestamp
)
lookup_lens = 100
start_time = 1745897370
end_time = 1777433370
feature_len = 4
num_embeddings = [1000, 1000, 1000, 1000]
for ind in range(feature_len):
    name = f"feat{ind}"
    id_range = num_embeddings[ind]
    ids = torch.randint(0, id_range, (lookup_lens,))
    timestamp_data = torch.randint(start_time, end_time, ids.size(), dtype=torch.int64)
    lengths = torch.ones(lookup_lens).long()
    input_dict[name] = JaggedTensorWithTimestamp(values=ids, lengths=lengths, timestamps=timestamp_data)
kjt_tensor_with_ts = KeyedJaggedTensorWithTimestamp.from_jt_dict(input_dict)
```
