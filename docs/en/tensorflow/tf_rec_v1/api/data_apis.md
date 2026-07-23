# Data APIs

## get_asc_insert_func

**Description**

Obtains the data preprocessing function.

**Function Prototype**

```python
def get_asc_insert_func(tgt_key_specs=None, args_index_list=None, table_names=None, **kwargs)
```

**Parameters**

|Parameter|Type|Description|
|--|--|--|
|tgt_key_specs|<li>FeatureSpec</li><li>list[FeatureSpec]</li>|Feature object, list of feature objects, or tuple of feature objects. Default value: `None`.|
|args_index_list|list[int]|List of parameter indexes. Default value: `None`. Value range: [1, 2^31-1]|
|table_names|list[str]|List of table names. Default value: `None`. Value range: [1, 2^31-1]|

>[!NOTE]
>You can pass interface parameters in one of the following ways.
>
>- Pass only `tgt_key_specs`.
>- Pass `args_index_list` and `table_names`.

**\*\*kwargs parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|is_training|bool|Optional|Specifies whether the training mode is used. Default value: `True`.<br>Value range: <li>`True`: training mode </li><li>`False`: evaluation or prediction mode</li>|
|dump_graph|bool|Optional|Specifies whether to save the model graph. Default value: `False`.<br>Value range: <li>`True`: Saves the model graph. </li><li>`False`: Does not save the model graph.</li>|

>[!NOTE]
>
>- The `is_training` and `dump_graph` parameters in `**kwargs` are for internal use. You are advised not to pass these two parameters through `kwargs`.
>- If you pass other undocumented parameters through `kwargs`, Rec SDK TensorFlow will not use them internally.

**Returns**

- Success: Returns the data preprocessing function.
- Failure: Throws an exception.

**Example**

```python
import tensorflow as tf
from mx_rec.core.asc.helper import get_asc_insert_func

dataset = tf.data.TFRecordDataset(data_path) # data_path is the dataset path.
dataset = dataset.map(get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=True)) # Elements in feature_spec_list are FeatureSpec objects.
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `modify_graph_and_start_emb_cache`

**Description**

Enables data loading and preprocessing in automatic graph modification mode.

**Function Prototype**

```python
def modify_graph_and_start_emb_cache(full_graph = None, dump_graph = False)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|full_graph|tf.Graph|Optional|A graph instance can be passed for graph modification. Default value: `None`. `None` will be assigned as `tf.compat.v1.get_default_graph()`.|
|dump_graph|bool|Optional|Specifies whether to save the model graph. Default value: `False`.<br>Value range: <li>`True`: Saves the model graph. </li><li>`False`: Does not save the model graph.</li>|

**Returns**

- Success: No value is returned.
- Failure: Throws an exception.

**Example**

```python
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
MODIFY_GRAPH_FLAG = True
if MODIFY_GRAPH_FLAG:
    modify_graph_and_start_emb_cache(dump_graph=True)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `start_asc_pipeline`

**Description**

Initializes and starts the data preprocessing pipeline in non-automatic graph modification mode.

**Function Prototype**

```python
def start_asc_pipeline()
```

**Returns**

- Success: No value is returned.
- Failure: Throws an exception.

**Example**

```python
from mx_rec.core.asc.manager import start_asc_pipeline
MODIFY_GRAPH_FLAG = False
if not MODIFY_GRAPH_FLAG:
    start_asc_pipeline()
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).
