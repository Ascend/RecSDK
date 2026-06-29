# Model APIs

## `create_table`

**Description**

Creates a sparse table.

**Function Prototype**

```python
def create_table(key_dtype, dim, name, emb_initializer, device_vocabulary_size=1, host_vocabulary_size=0, ssd_vocabulary_size=0, ssd_data_path=(os.getcwd(),), is_save=True, is_dp=False, init_param=1.0, all2all_gradients_op=All2allGradientsOp.SUM_GRADIENTS.value, enable_merge=False, padding_keys=None, padding_keys_mask=False, padding_keys_len=None, value_dtype=tf.float32, shard_num=1, fusion_optimizer_var=True, hashtable_threshold=0)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|key_dtype|TensorFlow dtype|Mandatory|Data type of the sparse feature keys. Only tf.int64 and tf.int32 are supported.|
|dim|<li>int</li><li>tf.TensorShape</li>|Mandatory|Embedding layer dimension. The value range is [1, 8192]. If `dim` needs to be greater than 512, ensure sufficient memory and drive space, use DDR mode, or reduce the vocabulary size of the sparse table.<br>If the input is of tf.TensorShape type, its `ndims` must be 1, representing the embedding layer dimension.<br>Set this parameter based on the actual server configuration.|
|name|str|Mandatory|Name of the sparse table, which can only contain numbers, letters, underscores (_), and periods (.). The value can contain 1 to 100 characters.<br>The sparse table name must be unique.|
|emb_initializer|TensorFlow initializer|Mandatory|Generator for initial values of the embedding layer.|
|device_vocabulary_size|int|Optional|Number of embedding layers on the device. The default value is 1. The value ranges from 1 to 1 billion. If this value exceeds 25,600,000, ensure sufficient memory and drive space, enable on-chip memory dynamic expansion, or reduce `dim` of the sparse table. Set this parameter based on the actual server configuration.<br>If DDR/SSD storage is enabled (that is, `host_vocabulary_size` is not 0), `device_vocabulary_size` must be greater than or equal to the number of keys after two consecutive batches are deduplicated. The on-chip memory must store at least two batches of data, as it acts only as a cache in this case.|
|host_vocabulary_size|int|Optional|Number of embedding layers for DDR storage on the host side. The default value is 0. The value ranges from 0 to 1 billion. <li>A value of 0 indicates that host-side DDR is disabled. If SSD storage is not enabled, ensure that the DDR can store all data.</li><li>A non-zero value enables the feature. In this case, disable on-chip memory dynamic expansion (`use_dynamic_expansion=False`). By default, the system uses the dynamic expansion mode for DDR memory. If `host_vocabulary_size` is greater than 100 million, ensure sufficient memory and drive space, or reduce `dim` of the sparse table. </li>In dynamic expansion mode (when `use_dynamic_expansion=True`), on-chip memory is the sole storage by default, and this variable is set to 0 Out-of-memory (OOM) errors occur if the size exceeds the single-machine memory. Set this parameter based on the actual server configuration.|
|ssd_vocabulary_size|int|Optional|Enables the SSD storage feature for embedding data. The default value is `0`, indicating that the feature is disabled. To enable this feature, set this value to greater than 0 and ensure `host_vocabulary_size` is also greater than 0. The value ranges from 0 to 1 billion.<br>In dynamic expansion mode (when `use_dynamic_expansion=True`), on-chip memory is the sole storage by default, and this variable is set to 0 Set this parameter based on the actual server configuration.|
|ssd_data_path|<li>list[str]</li><li>Tuple[str]</li>|Optional|The default value is the path where the current script is located. <li>If the parameter is an empty list, the default SSD storage path is the directory of the current script. </li><li>If the list is not empty and paths are valid, data is stored in the corresponding paths sequentially. </li><li>If a drive has insufficient space, the system attempts the next path until an exception is thrown when all drives are full.</li>|
|is_save|bool|Optional|Specifies whether to save embedding data. The default value is `True`.<br>Range:<li>`True`: Saves embedding data. </li><li>`False`: Does not save embedding data.</li>|
|is_dp|bool|Optional|Specifies whether to enable data parallelism (DP) for the sparse table. The default value is `False`.<br>Enabling DP mode (`is_dp=True`) configures the table for data parallelism. You are advised to enable this when using small tables (approximately 10 GB), the NPU side is an end-to-end bottleneck, and the NPU ALL2ALL communication volume for the sparse table is less than 16 MB. This can improve performance by about 15% in sparse communication.<br>Note: Mixing DP and model parallelism (MP) modes for resumable training does not affect functionality or accuracy but is not recommended.|
|init_param|float|Optional|Coefficient for embedding initialization parameters. The default value is 1.0. The value range is [-10, 10].<br>If `init_param` is greater than 1.0 or less than -1.0, you are advised to reduce the [batch_size](class_reference.md#featurespec) to avoid exceptions caused by excessive memory usage.|
|all2all_gradients_op|string|Optional|Method for gradient aggregation after distributed gradient backpropagation. The default value is `sum_gradients`. <li>`sum_gradients`: Adds gradients from all ranks. </li><li>`sum_gradients_and_div_by_ranksize`: Adds gradients from all ranks and divides the sum by the `ranksize`.</li>|
|enable_merge|bool|Optional|The EmbeddingTable merge feature determines merge eligibility based on the consistency of creation parameters. If two tables have identical `key_dtype`, `dim`, `emb_initializer`, `is_save`, `is_dp`, `init_param`, `all2all_gradients_op`, and the optional `padding_keys_mask` parameters, they can be merged into one. Merging EmbeddingTables effectively reduces the number of CPU threads and communication channels between the host and device, thereby saving system resources.<br>When merging multiple embedding tables, ensure that the IDs of each table are independent to avoid impact on accuracy. For example, if `UserEmbeddingTable` and `ItemEmbeddingTable` are to be merged, `UserID` and `ItemID` must not share any identical values.<br>Currently, this only supports the on-chip memory dynamic expansion mode.<br>Enabling automatic table merging requires enabling automatic graph modification and the multi-lookup for single tables feature. <li>`False`: Default value, indicating that automatic merging is disabled. </li><li>`True`: Indicates that automatic merging is enabled.</li>|
|padding_keys|<li>int64</li><li>list[int64]</li><li>None</li>|Optional|These are usually keys in the sparse features of the dataset. The default value is `None`. In this case, set `padding_keys_mask=False` and `padding_keys_len=None` for normal training updates. If the value is of int64 or list[int64] type, set `padding_keys_mask=True` and `padding_keys_len=shape` (where `shape` is the shape of the corresponding sparse feature in the dataset), indicating that embeddings for these keys do not need updates.|
|padding_keys_mask|bool|Optional|Indicates whether embeddings for `padding_keys` require updates. The default value is `False`. In this case, set `padding_keys=None` and `padding_keys_len=None` for normal training updates. When this parameter is set to `True`, set a value of int64 or list[int64] type for `padding_keys` and a value of int32 type for `padding_keys_len`, indicating that no updates are needed.|
|padding_keys_len|<li>int32</li><li>None</li>|Optional|Represents the shape of sparse features in the dataset, usually `batch size * feature vector dimension`. The default value is `None`. In this case, set `padding_keys=None` and `padding_keys_mask=False` for normal training updates. When this parameter is set to a value of int32 type, set a value of int64 or list[int64] type for `padding_keys` and `padding_keys_mask=True`, indicating that embeddings for `padding_keys` do not need updates.|
|value_dtype|TensorFlow dtype|Optional|Data type of sparse feature values. `tf.float32` is the default and only supported value.|
|shard_num|int|Optional|Number of embedding layer partitions. The default value is 1. The value range is [1,8192].|
|fusion_optimizer_var|bool|Optional|Specifies whether to use fusion optimization parameters. The default value is `True`.<br>Value range: <li>`True`: Uses fusion optimization parameters. </li><li>`False`: Does not use fusion optimization parameters.</li>|
|hashtable_threshold|int|Optional|Hash table threshold. A hash table is used when the value exceeds this threshold. Otherwise, a linear table is used. The default value is 0. The value range is [0,2147483647].|

>[!NOTE]
>
>- For `padding_keys`, `padding_keys_mask`, and `padding_keys_len`:
>
>> - The DP mode or lazy Adam fusion operator mode is not supported.
>> - If all tables require padding keys, set the static shape mode in the initialization interface, such as `init(use_dynamic=False)`. In this case, ensure `drop_remainder=True` in the model script, for example, `dataset = dataset.batch(batch_size, drop_remainder=True)`.
>
>- When enabling DDR/SSD mode, ensure that the [automatic graph modification](../appendix.md#automatic-graph-modification) mode is also enabled.
>- If the on-chip memory dynamic expansion solution is used (that is, `use_dynamic_expansion=True` is set for the [`init`](initialization_and_deinitialization_of_the_training_framework.md#init) interface), the `host_vocabulary_size` and `ssd_vocabulary_size` parameters are set to 0 and do not take effect. You can also leave them blank.
>- If the on-chip memory dynamic expansion solution is not used, the storage mode is determined by whether `host_vocabulary_size` and `ssd_vocabulary_size` are 0.
>- When `host_vocabulary_size` is 0, host-side DDR is disabled. All embedding tables must either simultaneously use host-side DDR or not use it. That is, the `host_vocabulary_size` parameter for all tables **must be 0 at the same time** or **not be 0 at the same time**. Otherwise, an error occurs during parameter validation, as shown below:
>
> ```bash
>  ValueError: The host-side DDR function of all tables must be used or not used at the same time. However, host voc size of each table is [].
> ```

**Returns**

- Success: A sparse table instance is returned.

    You can access two methods of the sparse table instance as described below.

    |Method|Function|Prototype|Parameters|Returns|
    |--|--|--|--|--|
    |`size`|Obtains the size of the sparse table.|` def size() `|None|<li>Success: The size of the sparse table is returned. </li><li>Failure: An exception is thrown.</li>|
    |`capacity`|Obtains the capacity of the sparse table.|` def capacity() `|None|<li>Success: The capacity of the sparse table is returned. </li><li>Failure: An exception is thrown.</li>|

- Failure: An exception is thrown.

**Example**

```python
import tensorflow as tf
from mx_rec.core.embedding import create_table
sparse_hashtable = create_table(key_dtype=tf.int32,
                                dim=tf.TensorShape([128]),
                                name="sparse_embeddings_table",
                                emb_initializer=tf.truncated_normal_initializer(),
                                device_vocabulary_size=24_000_000 * 8,
                                host_vocabulary_size=0)
table_size = sparse_hashtable.size()      # Get the used size of the returned sparse table.
table_capacity = sparse_hashtable.capacity()   # Get the capacity of the returned sparse table.
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `sparse_lookup`

**Description**

Lookup interface for the sparse feature table in the Rec SDK TensorFlow model training framework.

Currently, only single-lookup for single tables and multi-lookup for single tables are supported. In the case of multi-lookup for single tables, the maximum number of lookups is 128.

The tf.SparseTensor data type is not currently supported. If you use tf.SparseTensor, convert it to tf.Tensor. Example code:

```bash
# Example code
sparse_ids = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
dense_ids = tf.sparse.to_dense(sparse_ids, default_value=0)
embedding = sparse_lookup(sparse_hashtable, dense_ids)
```

**Function Prototype**

```python
def sparse_lookup(hashtable, ids, send_count, is_train=True, name=None, modify_graph=False, batch=None, access_and_evict_config=None, is_grad=True, serving_default_value, **kwargs)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|hashtable|BaseSparseEmbedding|Mandatory|Sparse table to be queried.|
|ids|FeatureSpec/tf.Tensor|Mandatory|Lookup keys. The parameter type varies depending on the functional mode as follows: <li>In non-automatic graph modification mode, the `ids` parameter type is FeatureSpec. </li><li>In automatic graph modification mode, the `ids` parameter type is tf.Tensor.</li>|
|send_count|int|Mandatory when static shape is enabled|Specifies the send count for All2All communication. The value range is [1, 2147483647].<br>This parameter is not required when dynamic shape is enabled. You can pass `None`. The default value is `None`.|
|is_train|bool|Mandatory|Specifies whether training mode is active. The default value is `True`.<br>Value range: <li>`True`: training mode </li><li>`False`: evaluation or prediction mode</li>|
|name|str|Optional|Creates a name for this lookup operation. The string can contain 1 to 255 characters. The default value is `None`.|
|modify_graph|bool|Optional|Switch for the automatic graph modification feature, which optimizes the model graph before creating a `Session` instance. The default value is `False`.<br>Value range: <li>`True`: Enables the automatic graph modification feature. </li><li>`False`: Disables the automatic graph modification feature.</li>|
|batch|dict{str:tf.Tensor}|Optional|Iterator for the dataset.<br>When `ids` uses the FeatureSpec type and dynamic shape is active, pass the `batch` parameter. The key is the feature name, and the value is the result after `get_next()` is called on the corresponding `tf.Dataset`. The default value is `None`.|
|access_and_evict_config|dict{str:int}|Optional|Used when feature admission and eviction are enabled in automatic graph modification mode. This dictionary consists of three key-value pairs with keys `access_threshold`, `eviction_threshold`, and `faae_coefficient`.<br>The default value for `access_threshold` and `eviction_threshold` is `None`, and the default value for `faae_coefficient` is 1. <li>The values of `access_threshold` and `eviction_threshold` are the corresponding thresholds. </li><li>The value of `faae_coefficient` is the feature admission coefficient.</li>|
|is_grad|bool|Optional|Specifies whether gradient updates are required for this lookup. The default value is `True`.<br>Value range:<li>`True`: Gradient updates are required. </li><li>`False`: Gradient updates are not required.</li>|
|serving_default_value|tf.Tensor|Optional|Default embedding value for non-admitted features during training or new features during prediction. If not specified, the default value is `None`.|

**\*\*kwargs parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|feature_spec_name_ids_dict|dict|Optional|Dictionary structure where the key is the `FeatureSpec` name and the value is the `ids` parameter of the public interface `sparse_lookup()`. There is no default value.|
|multi_lookup|bool|Optional|Specifies whether multi-lookup for single tables exists. There is no default value.<br>Value range: <li>`True`: Multi-lookup for single tables exists. </li><li>`False`: Multi-lookup for single tables does not exist.</li>|
|lookup_ids|FeatureSpec/tf.Tensor|Optional|Lookup keys. The parameter type varies depending on the functional mode. There is no default value. Details:<li>In non-automatic graph modification mode, the `ids` parameter type is `FeatureSpec`. </li><li>In automatic graph modification mode, the `ids` parameter type is tf.Tensor.</li>|

>[!NOTE]
>
>- The `feature_spec_name_ids_dict`, `multi_lookup`, and `lookup_ids` parameters in `**kwargs` are for internal use. You are advised not to pass these parameters through `kwargs`.
>- If you pass other undocumented parameters through `kwargs`, Rec SDK TensorFlow will not use them internally.
>

**Returns**

- Success: The queried `Tensor` class result is returned.
- Failure: An exception is thrown.

**Example**

```python
from mx_rec.core.embedding import sparse_lookup
from mx_rec.core.asc.feature_spec import FeatureSpec
feature_spec = FeatureSpec("sparse_feature", table_name="sparse_embeddings_table",
                                batch_size=1)
embedding = sparse_lookup(sparse_hashtable,
                          feature_spec,
                          send_count=6000,
                          is_train=True,
                          name="sparse_embeddings")
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `get_dense_and_sparse_variable`

**Description**

Gets the parameter variables for dense layers and sparse layers in the model.

**Function Prototype**

```python
def get_dense_and_sparse_variable()
```

**Returns**

- Success: The dense layer parameter variables and sparse layer parameter variables are returned.
- Failure: An exception is thrown.

**Example**

```python
from mx_rec.util.variable import get_dense_and_sparse_variable
dense_variables, sparse_variables = get_dense_and_sparse_variable()
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `export`

**Description**

Reads the sparse table checkpoint saved for the current step and saves it as a NumPy file in key-value format, where the key is the lookup ID and the value is the embedding layer representation. The `export` interface supports exporting sparse table data in the following cases:

- On-chip memory mode (non-dynamic expansion)
- DDR mode

**Function Prototype**

```python
def export(table_list=None):
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|table_list|list[str]|Optional|List of sparse tables to export. All sparse tables are exported by default if no parameter is passed. The list length range is [1, 2^31-1].|

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
from mx_rec.saver.sparse import export
table_list=["sparse_embedding_table"]
export(table_list=table_list)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `if_load`

**Description**

Sets whether to load a previously trained model.

**Function Prototype**

```python
@property
 def if_load(self)
```

**Returns**

- True: The trained model is loaded.
- False: The trained model is not loaded.

**Example**

```python
from mx_rec.util.initialize import ConfigInitializer
if_load = ConfigInitializer.get_instance().if_load
```
