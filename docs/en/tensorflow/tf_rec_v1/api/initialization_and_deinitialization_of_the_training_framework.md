# Initialization and Deinitialization of the Training Framework

## `init`

**Description**

Initializes the Rec SDK TensorFlow model training framework.

**Function Prototype**

```python
def init(**kwargs)
```

**\*\*kwargs parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|max_steps|int|Optional|Total number of training steps. The default value is -1, which means that training continues until all training data is exhausted. Value range: [-1, 2147483647].|
|train_steps|int|Optional|Number of training steps before performing test prediction. The default value is -1, which means that prediction is performed after the entire training dataset is trained. Value range: [-1, 2147483647].|
|eval_steps|int|Optional|Number of test prediction steps. The default value is `-1`, which means that training resumes after the entire test dataset is predicted. Value range: [-1, 2147483647].|
|if_load|bool|Optional|Specifies whether to load the model. The default value is `False`.<br>Value range: <li>`True`: Loads the model. </li><li>`False`: Does not load the model.</li>|
|use_dynamic|bool|Optional|Specifies whether to use the dynamic shape feature. The default value is `True`.<br>Value range: <li>`True`: Uses the dynamic shape feature. </li><li>`False`: Does not use the dynamic shape feature.</li>|
|use_dynamic_expansion|bool|Optional|Specifies whether to use the on-chip memory dynamic expansion feature. The default value is `False`.<br>Value range: <li>`True`: Uses the dynamic expansion feature. </li><li>`False`: Does not use the dynamic expansion feature.</li>|
|bind_cpu|bool|Optional|Specifies whether to use the automatic CPU core binding feature. The default value is `True`.<br>Value range: <li>`True`: Uses the automatic CPU core binding feature. </li><li>`False`: Does not use the automatic CPU core binding feature.</li>|
|save_steps|int|Optional|Save the model after training for `save_steps`. The default value is `-1`, which means that saving occurs after the entire training data is trained. Value range: [-1, 2147483647].|
|save_checkpoint_due_time|int|Optional|Time interval (in seconds) for saving full models. Value range: [1, 2147483647]. Generally, `save_checkpoint_due_time` should be greater than `save_delta_checkpoints_secs`.<br>This parameter is required when `is_incremental_checkpoint` is set to `True`.|
|save_delta_checkpoints_secs|int|Optional|Time interval (in seconds) for saving incremental models. Value range: [1, 2147483647]. Generally, `save_checkpoint_due_time` should be greater than `save_delta_checkpoints_secs`.<br>This parameter is required when `is_incremental_checkpoint` is set to `True`.|
|is_incremental_checkpoint|bool|Optional|Specifies whether to enable incremental model saving and loading. The default value is `False`. <li>`True`: Enables incremental model saving and loading. </li><li>`False`: Disables incremental model saving and loading.</li>|
|restore_model_version|int|Optional|Specific training step (`step`) of the model to load. If this parameter is not provided, the latest model is loaded by default. If a specific `step` value is provided, the model corresponding to that step is loaded.<br>Value range: [0, 2147483647].|
|recent_key_count_threshold|int|Optional|Minimum number of occurrences for a key during the incremental saving period, used for low-frequency filtering. Keys with an occurrence frequency lower than this parameter are filtered out when saving the incremental model. Default value: `0`.<br>Value range: [0, 2147483647].|
|use_lccl|bool|Optional|When running multi-device tasks with low communication bandwidth utilization, you can use the Low Latency Collective Communication Library (LCCL) feature to accelerate collective communication. When this feature is enabled, the following LCCL operators will be used in certain scenarios. This feature only supports the non-expansion mode of single-node on-chip memory. For detailed usage, see [LCCL Communication Optimization Operators and Example Description](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/lccl/v220/README.md). <li>All2All operator</li><li>GatherAll operator (Gather and AllToAll fusion operator)</li><li>GatherUss operator (Gather and UnsortedSegmentSum fusion operator)</li>The default value is `False`, indicating that this feature is disabled.|

>[!NOTE]
>
>- When `sess.run()` is used for training, the number of steps for training, evaluation, and saving using the session must be consistent with the `train_steps`, `eval_steps`, and `save_steps` parameters.
>- When Estimator is used for training:
>
>> - `save_steps` must match the `save_checkpoints_steps` parameter used when the `NPURunConfig` object is defined. TensorFlow does not support a value of `-1` for this parameter.
>> - `max_steps` must match the `max_steps` parameter passed to `est.train()` or `tf.estimator.TrainSpec()`. TensorFlow does not support a value of `-1` for this parameter.
>> - In `train_and_evaluate` mode, the requirements for `save_steps` and `max_steps` are the same as those mentioned in the preceding sections. Additionally, `train_steps` must match `save_steps`, and `eval_steps` must match the `steps` parameter passed to `tf.estimator.EvalSpec()`. TensorFlow does not support a value of `-1` for this parameter.
>
>- If you pass other undocumented parameters through `kwargs`, Rec SDK TensorFlow will not use them internally.
>- `max_steps`, `train_steps`, and `eval_steps` cannot all be 0, and the parameters passed must reflect the actual training scenario.
>- When the `use_dynamic_expansion` parameter is `True`, use `ByAddr` optimizers, such as [SGDByAddr](optimizers_apis.md#sgdbyaddr) and [LazyAdamByAddress](optimizers_apis.md#lazyadambyaddress).
>- Multi-round evaluation is not supported in `train_and_evaluate` scenarios.
>- `max_steps`, `train_steps`, `eval_steps`, and `save_steps` must be consistent with the actual training situation. Inconsistencies may lead to issues such as training failure or accuracy problems.
>

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
from mx_rec.util.initialize import init
init(max_steps=200, train_steps=100, eval_steps=10, save_steps=100, use_dynamic=True, use_dynamic_expansion=False)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `terminate_config_initializer`

**Description**

Deinitializes the framework and releases resources.

**Function Prototype**

```python
def terminate_config_initializer()
```

**Example**

```python
from mx_rec.util.initialize import terminate_config_initializer
terminate_config_initializer()
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
