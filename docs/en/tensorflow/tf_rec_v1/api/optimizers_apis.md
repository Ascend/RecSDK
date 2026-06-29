# Optimizers

## Adagrad

Customized Adagrad optimizer.

**Function Prototype**

```python
def create_hash_optimizer(learning_rate=0.001, initial_accumulator_value=0.9, use_locking=False, name="Adagrad")
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate|float or tf.Tensor|Optional|Learning rate. The default value is 0.001. The value range is [0.0, 10.0].|
|initial_accumulator_value|float|Optional|Initial value for the accumulator. The value range is (0.0, 1.0]. The default value is 0.9.|
|use_locking|bool|Optional|Prevents concurrent updates to variables in the optimizer. The default value is `False`. The value can be `True` or `False`.|
|name|string|Optional|Name of the optimizer. The default value is `Adagrad`. The name can contain 1 to 200 characters.|

**Returns**

An instance of `CustomizedAdagrad` (customized Adagrad optimizer).

**Example**

```python
from mx_rec.optimizers.adagrad import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer(0.001)
```

## Ftrl

Customized Ftrl optimizer.

**Function Prototype**

```python
def create_hash_optimizer(learning_rate, use_locking=False, name="Ftrl", **kwargs)
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate|float or tf.Tensor|Mandatory|Learning rate. The value range is [0.0, 10.0].|
|use_locking|bool|Optional|Prevents concurrent updates to variables in the optimizer. The default value is `False`. The value can be `True` or `False`.|
|name|string|Optional|Name of the optimizer. The default value is Ftrl. The name can contain 1 to 200 characters.|

**\*\*kwargs parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate_power|float|Optional|Controls the decrease of the learning rate during training. The default value is -0.5. The value range is [-2147483647.0, 0.0].|
|initial_accumulator_value|float|Optional|Initial value for the accumulator. The default value is 0.1. The value range is (0.0, 1.0].|
|l1_regularization_strength|float|Optional|L1 regularization penalty. The default value is 0.0. The value range is [0.0, 10000.0].|
|l2_regularization_strength|float|Optional|L2 regularization penalty. The default value is 0.0. The value range is [0.0, 10000.0].|
|accum_name|string|Optional|Suffix for the variable that saves the squared gradient accumulator. The default value is `None`. The value is a string of 1 to 255 characters.|
|linear_name|string|Optional|Suffix for the variable that saves the linear gradient accumulator. The default value is `None`. The value is a string of 1 to 255 characters.|
|l2_shrinkage_regularization_strength|float|Optional|L2 shrinkage regularization penalty. The default value is 0.0. The value range is [0.0, 10000.0].|

>[!NOTE]
>If you pass other undocumented parameters through `kwargs`, Rec SDK TensorFlow will not use them internally.

**Returns**

An instance of `CustomizedFtrl` (customized Ftrl optimizer).

**Example**

```python
from mx_rec.optimizers.ftrl import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer(0.001)
```

## SGD

Customized SGD optimizer.

**Function Prototype**

```python
def create_hash_optimizer(learning_rate, use_locking=False, name="GradientDescent", use_fusion_optim=False, weight_decay=None)
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate|float or tf.Tensor|Mandatory|Learning rate. The value range is [0.0, 10.0].|
|use_locking|bool|Optional|Prevents concurrent updates to variables in the optimizer. The default value is `False`. The value can be `True` or `False`.|
|name|string|Optional|Name of the optimizer. The default value is `GradientDescent`. The name can contain 1 to 200 characters.|
|use_fusion_optim|bool|Optional|Specifies whether to enable operator acceleration. The default value is `False`.<br>Value range: <li>`True`: Enables fusion operator acceleration. </li><li>`False`: Disables fusion operator acceleration.</li>|
|weight_decay|float|Optional|Weight decay coefficient. The default value is `None`, which means that weight decay is disabled. The value range is [1e-5, 1e-2].|

**Returns**

An instance of `CustomizedGradientDescent` (customized SGD optimizer).

**Example**

```python
from mx_rec.optimizers.gradient_descent import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer(0.001)
```

## SGDByAddr

Customized SGDByAddr optimizer.

**Function Prototype**

```python
def create_hash_optimizer_by_addr(learning_rate, weight_decay=0.0001, use_locking=False, name="GradientDescentByAddr")
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate|float or tf.Tensor|Mandatory|Learning rate. The value range is [0.0, 10.0].|
|weight_decay|float|Optional|Weight decay. The default value is 0.0001. The value range is [0.0, 1.0].|
|use_locking|bool|Optional|Prevents concurrent updates to variables in the optimizer. The default value is `False`. The value can be `True` or `False`.|
|name|string|Optional|Name of the optimizer. The default value is `GradientDescentByAddr`. The name can contain 1 to 200 characters.|

**Returns**

An instance of `CustomizedGradientDescentByAddr` (customized SGD address optimizer).

**Example**

```python
from mx_rec.optimizers.gradient_descent_by_addr import create_hash_optimizer_by_addr
hashtable_optimizer = create_hash_optimizer_by_addr(0.001)
```

## LazyAdam

Customized LazyAdam optimizer.

**Function Prototype**

```python
def create_hash_optimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdam", use_fusion_optim=False)
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**| **Description**                                                                                                                                                                                                                                                                              |
|--|--|--|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|learning_rate|float or tf.Tensor|Optional| Learning rate. The default value is 0.001. The value range is [0.0, 10.0].                                                                                                                                                                                                                                                     |
|beta1|float|Optional| Exponential decay rate for the first moment estimates. The default value is 0.9. The value range is (0.0, 1.0).                                                                                                                                                                                                                                                |
|beta2|float|Optional| Exponential decay rate for the second moment estimates. The default value is 0.999. The value range is [0.0, 1.0].                                                                                                                                                                                                                                              |
|epsilon|float|Optional| Value added to the denominator to improve numerical stability. The default value is 1e-8. The value range is (0.0, 1.0].                                                                                                                                                                                                                                          |
|name|string|Optional| Name of the optimizer. The default value is `LazyAdam`. The name can contain 1 to 200 characters.                                                                                                                                                                                                                                                 |
|use_fusion_optim|bool|Optional| Specifies whether to use the LazyAdam fusion operator for computing and updating `slot_m`, `slot_v`, and `variable` data. The default value is `False`.<br>Value range: <li>`True`: Uses the fusion operator. You need to manually compile and deploy the LazyAdam fusion operator. For details, see [README](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/fused_lazy_adam/v220/README.md). </li><li>`False`: Does not use the LazyAdam fusion operator.</li> |

**Returns**

An instance of `CustomizedLazyAdam` (customized LazyAdam optimizer).

**Example**

```python
from mx_rec.optimizers.lazy_adam import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer()
```

## LazyAdamByAddress

Customized LazyAdamByAddress optimizer.

**Function Prototype**

```python
def create_hash_optimizer_by_address(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdamByAddress")
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate|float or tf.Tensor|Optional|Learning rate. The default value is 0.001. The value range is [0.0, 10.0].|
|beta1|float|Optional|Exponential decay rate for the first moment estimates. The default value is 0.9. The value range is (0.0, 1.0).|
|beta2|float|Optional|Exponential decay rate for the second moment estimates. The default value is 0.999. The value range is [0.0, 1.0].|
|epsilon|float|Optional|Value added to the denominator to improve numerical stability. The default value is 1e-8. The value range is (0.0, 1.0].|
|name|string|Optional|Name of the optimizer. The default value is `LazyAdamByAddress`. The name can contain 1 to 200 characters.|

**Returns**

An instance of `CustomizedLazyAdamByAddress` (customized LazyAdam address optimizer).

**Example**

```python
from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address
hashtable_optimizer = create_hash_optimizer_by_address()
```

## AdagradByAddress

Customized AdagradByAddress optimizer.

**Function Prototype**

```python
def create_hash_optimizer_by_address(learning_rate=0.001, initial_accumulator_value=0.9, name="Adagrad")
```

**Parameters**

|**Parameter**|**Type**|**Mandatory/Optional**|**Description**|
|--|--|--|--|
|learning_rate|float or tf.Tensor|Optional|Learning rate. The default value is 0.001. The value range is [0.0, 10.0].|
|initial_accumulator_value|float|Optional|Initial value for the accumulator. The default value is 0.9. The value range is (0.0, 1.0].|
|name|string|Optional|Name of the optimizer. The default value is `Adagrad`. The name can contain 1 to 200 characters.|

**Returns**

An instance of `CustomizedAdagradByAddress` (customized Adagrad address optimizer).

**Example**

```python
from mx_rec.optimizers.adagrad_by_addr import create_hash_optimizer_by_address
hashtable_optimizer = create_hash_optimizer_by_address()
```
