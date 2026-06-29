# Class Reference

## `FeatureSpec`

Configuration description class for features to be queried, which applies to non-automatic graph modification mode.

|Parameter|Type|**Mandatory/Optional**|Description|
|--|--|--|--|
|index_key|int/string|Optional|Index key. Default value: value of `table_name`.<br>Value range: <li>[0, 255] for an int value </li><li>1 to 255 characters for a string value</li>|
|table_name|string|Optional.|Table name.<br>The value can contain 1 to 255 characters.|
|access_threshold|int|Optional|Feature admission threshold.<br>Value range: [-1, 2147483647]. <li>0: Enables the admission function. The occurrence count of keys in new batches is not accumulated. Historical feature counts are used. </li><li>Greater than 0: Enables the admission function. The occurrence count of keys in new batches is accumulated, and feature count records are updated. </li><li>-1: Disables the function.</li>|
|eviction_threshold|int|Optional|Feature eviction threshold.<br>Value range: [-1, 2147483647]. <li>Greater than or equal to 0: Enables the eviction function. </li><li>-1: Disables the eviction function. </li>If you need to set a feature eviction threshold, you must also set the feature admission threshold.|
|is_timestamp|bool|Optional|Specifies whether it is a timestamp.<br>Value range: `True` or `False`.|
|batch_size|int|Optional|Dataset batch size.<br>Value range: [1, 2147483647].|
|faae_coefficient|int|Optional|Feature admission coefficient. Default value: `1`.<br>Value range: [1, 2147483647].|
|name|string|Mandatory|`FeatureSpec` name. The value can contain 1 to 255 characters.|

**Example**

```python
from mx_rec.core.asc.feature_spec import FeatureSpec
feature_spec_list = FeatureSpec("user_ids", table_name="user_table",
                                access_threshold=1,
                                eviction_threshold=1,
                                faae_coefficient=1)

```

## `GraphModifierHook`

Automatic graph modification hook class, used only in [Training with Estimator](../migration_and_training.md#training-with-estimator) mode. The automatic graph modification feature is enabled after this hook is added.

|Parameter|Type|**Mandatory/Optional**|Description|
|--|--|--|--|
|dump_graph|bool|Optional|Specifies whether to save the current TensorFlow computational graph. Default value: `False`.|
|modify_graph|bool|Optional|Specifies whether to enable automatic graph modification. Default value: `True`.|

**Example**

```python
from mx_rec.graph.modifier import GraphModifierHook

#Define the data processing function.
def input_fn():
     pass

est.train(input_fn=lambda: input_fn(), hooks=[GraphModifierHook()])   # est is the created NPUEstimator object.
```

## `EvictHook`

Feature eviction hook class, used only in feature admission and eviction mode. It works with the feature eviction threshold `eviction_threshold`. The feature eviction function is enabled after this hook is added.

>[!NOTE]
>The feature eviction hook class supports training scenarios only.

|Parameter|Type|**Mandatory/Optional**|Description|
|--|--|--|--|
|evict_enable|bool|Optional|Specifies whether to enable feature eviction. Default value: `False`.|
|evict_time_interval|int|Optional|Interval for triggering the eviction function, in seconds. The default value is `24 * 60 * 60`. Value range: [1, MAXINT32].|
|evict_step_interval|int|Optional|Interval for triggering the eviction function, in steps. Default value: `None`. Value range: [1, MAXINT32].|

**Example**

```python
from mx_rec.core.feature_process import EvictHook
hooks_list = []
hook_evict = EvictHook(evict_enable=True, evict_time_interval=30, evict_step_interval=20)
hooks_list.append(hook_evict)

#Define the data processing function.
def input_fn():
     pass

est.train(input_fn=lambda: input_fn(), hooks=hooks_list) # est is the created NPUEstimator object.
```

## `ConfigInitializer`

Management class for saving global configuration information, which uses the singleton pattern.

This class is automatically initialized via the `init()` function and does not require manual construction. This section lists only the public interfaces of this class. Other interfaces not mentioned here are internal and should not be called directly.

**Call Example**

|Interface|Purpose|Prototype|
|--|--|--|
|get_instance()|Gets the unique global instance of `ConfigInitializer`.|from mx_rec.util.initialize import ConfigInitializerConfigInitializer.get_instance()|
|use_dynamic_expansion()|See [use_dynamic_expansion](other_apis.md#use_dynamic_expansion).||
|get_target_batch()|See [get_target_batch](other_apis.md#get_target_batch).||
|if_load()|See [if_load](model_apis.md#if_load).||
|get_initializer(is_training)|See [get_initializer](automatic_graph_modification.md#get_initializer).||
|ascend_global_hashtable_collection()|See [ascend_global_hashtable_collection](other_apis.md#ascend_global_hashtable_collection).||

## `TrainParamsConfig`

Data class for saving training task parameter configurations, such as the name of the hash table collection.

This class is automatically initialized via the `init()` function and does not require manual construction. This section lists only the public interfaces of this class. Other interfaces not mentioned here are internal and should not be called directly.

**Call Example**

|Interface|Purpose|
|--|--|
|ascend_global_hashtable_collection()|See [ascend_global_hashtable_collection](other_apis.md#ascend_global_hashtable_collection).|
