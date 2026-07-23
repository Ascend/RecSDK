# Other APIs

## version

**Description**

Queries the version number of the Rec SDK TensorFlow framework.

**Function Prototype**

```python
def version()
```

**Returns**

- Success: The version number is returned.
- Failure: An exception is thrown.

**Example**

You can query the version number of the Rec SDK TensorFlow framework using `version()` or `__version__`. For example:

```python
import mx_rec
print(mx_rec.version())
print(mx_rec.__version__)
```

## `ascend_global_hashtable_collection`

**Description**

Modifies or retrieves the name of a hash table collection.

**Function Prototype**

```python
# Get the hash table collection name.
@property
def ascend_global_hashtable_collection(self):
    return self._ascend_global_hashtable_collection
# Modify the hash table collection name.
@ascend_global_hashtable_collection.setter
def ascend_global_hashtable_collection(self, name):
    self._ascend_global_hashtable_collection = name
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|name|string|Optional|Modified name of the hash table collection. The value is a string of 1 to 255 characters.|

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```bash
# Get the hash table collection name.
from mx_rec.util.initialize import ConfigInitializer
hashtable_collection = ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection
# Modify the hash table collection name.
from mx_rec.util.initialize import ConfigInitializer
ConfigInitializer.get_instance().train_params_config.ascend_global_hashtable_collection = "test"
```

## `get_rank_id`

**Description**

Returns the sequence number of the current process in MPI communication.

**Function Prototype**

```python
def get_rank_id()
```

**Returns**

- Success: The sequence number of the current process in MPI communication is returned.
- Failure: An exception is thrown.

**Example**

```python
from rec_sdk_common.communication.hccl.hccl_info import get_rank_id
rank_id = get_rank_id()
```

## `get_rank_size`

**Description**

Returns the total number of processes in the MPI communicator.

**Function Prototype**

```python
def get_rank_size()
```

**Returns**

- Success: The total number of processes in the MPI communicator is returned.
- Failure: An exception is thrown.

**Example**

```python
from rec_sdk_common.communication.hccl.hccl_info import get_rank_size
rank_size = get_rank_size()
```

## `import_host_pipeline_ops`

**Description**

Retrieves custom TensorFlow operators in Rec SDK TensorFlow.

**Function Prototype**

```python
def import_host_pipeline_ops(so_pkg_name: str = LIBASC_OPS_SO) -> ModuleType
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|so_pkg_name|string|Optional|Name of the Rec SDK TensorFlow .so package. The value is a string of 1 to 100 characters.|

**Returns**

- Success: A Python wrapper module containing TensorFlow operators defined in the Rec SDK TensorFlow .so file is returned.
- Failure: `RuntimeError: when unable to load the library or get the python wrappers.` is returned.

**Example**

```python
from mx_rec.util.ops import import_host_pipeline_ops
host_pipeline_ops = import_host_pipeline_ops()
```

**Introduction to the `set_threshold` Operator**

<table><tbody><tr id="row181439428296"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.1.1"><p id="p1144642112915">Name</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.1.1 "><p id="p160744914308">set_threshold</p>
</td>
</tr>
<tr id="row6144242202912"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.2.1"><p id="p1214414202911">Function</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.2.1 "><p id="p76061449173019">Changes the feature admission threshold.</p>
</td>
</tr>
<tr id="row11441042102913"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.3.1"><p id="p9144124217290">Parameters</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.3.1 "><ul id="ul1065662883114"><li>The first input parameter is a top-level tf.Tensor in the TensorFlow context. </li><li>emb_name: List[str]: Feature tables that require admission threshold modification. </li><li>ids_name: List[str]: Legacy parameter for compatibility reasons that currently has no effect.</li></ul>
</td>
</tr>
<tr id="row183421350113111"><th class="firstcol" valign="top" width="13.98%" id="mcps1.1.3.4.1"><p id="p1534235018317">Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="86.02%" headers="mcps1.1.3.4.1 "><p id="p1034245020313">If the first input parameter of set_threshold is "0", the corresponding embedding table is changed to a non-accumulation mode. In this mode, the admission threshold remains unchanged, but feature counting no longer accumulates, and historical values are used.</p>
</td>
</tr>
</tbody>
</table>

## `use_dynamic_expansion`

**Description**

Returns whether dynamic expansion is used.

**Function Prototype**

```python
@property
def use_dynamic_expansion(self)
```

**Returns**

- `True`: Dynamic expansion is used.
- `False`: Dynamic expansion is not used.

**Example**

```python
from mx_rec.util.initialize import ConfigInitializer
use_dynamic_expansion = ConfigInitializer.get_instance().use_dynamic_expansion
```

## `hccl_ops`

**Description**

Adapts to different versions of HCCL operators.

**Returns**

Returns the corresponding version of HCCL operators.

**Example**

```python
from mx_rec.util.tf_version_adapter import hccl_ops
```

## `get_target_batch`

**Description**

Returns the batch records of a new dataset generated in automatic graph modification mode.

**Function Prototype**

```python
def get_target_batch(self, is_training: bool)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|is_training|bool|Mandatory|Specifies whether training mode is active. <li>`True`: training mode </li><li>`False`: non-training mode</li>|

**Returns**

- Success: The batch records of a new dataset generated in automatic graph modification mode is returned.
- Failure: An exception is thrown.

**Example**

```python
from mx_rec.util.initialize import ConfigInitializer
ConfigInitializer.get_instance().train_params_config.get_target_batch(False)
```
