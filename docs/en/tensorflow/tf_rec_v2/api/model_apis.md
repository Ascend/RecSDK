# Model APIs

## `get_embedding_table`

**Description**

Creates a sparse table.

**Function Prototype**

```python
def get_embedding_table(
    name: str,
    dimension: int,
    device_vocabulary_size: int,
    initializer: Union[InitializerV1, InitializerV2] = tf.compat.v1.random_normal_initializer(),
    key_dtype: tf.DType = tf.int64,
    value_dtype: tf.DType = tf.float32,
    distribution_strategy: str = EmbDistributionStrategy.MP.value,
    min_used_times: Optional[int] = None,
    max_cold_secs: Optional[int] = None,
):
```

**Parameters**

| Parameter                   | Type          | Mandatory/Optional| Description                                                                                                  |
|------------------------|-------------------|------|------------------------------------------------------------------------------------------------------|
| name                   | str               | Mandatory  | Name of the sparse table. It can only contain [0-9A-Za-z_.] and its length must be in range [1, 128].                                                             |
| dimension              | int               | Mandatory  | Embedding dimension of the sparse table. The value range is [1, 512].                                                                      |
| device_vocabulary_size | int               | Mandatory  | Capacity of the sparse table on the device. The value range is [1, 10**9]. Ensure sufficient drive and memory space and set the parameter based on the server configuration.                                           |
| initializer            | TensorFlow initializer| Optional  | Initial value generator for the sparse table. The default is a random normal distribution initializer.                                                                           |
| key_dtype              | tf.int64          | Optional  | Data type of the sparse feature key. `tf.int64` is the default and only supported value.                                                           |
| value_dtype            | tf.float32        | Optional  | Data type of the sparse feature value. `tf.float32` is the default and only supported value.                                                     |
| distribution_strategy  | str               | Optional  | Distributed parallel mode for the sparse table. `MP` (model parallelism) is the default and only supported value.                                                                |
| min_used_times         | int / None        | Optional  | Feature admission: an ID must appear more times than this value to take effect. Otherwise, querying the table with this ID returns the default embedding value. The value range is [0, 2^31-1].                                 |
| max_cold_secs          | int / None        | Optional  | Feature eviction: the maximum tolerated idle time (in seconds) for an ID and its embedding since the last access. IDs exceeding this threshold are removed from the embedding hash table during saving. The default value is [0, 2^64-1].|

>[!NOTE]
>
> - - Only non-dynamic expansion mode is supported, that is, `device_vocabulary_size` must be greater than 0.
> - After feature eviction is enabled, saving and loading are not supported.

**Returns**

- Success: A sparse table instance is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
    initializer=tf.truncated_normal_initializer(),
    key_dtype=tf.int64,
    value_dtype=tf.float32,
)
```

## `embedding_lookup`

**Description**

Queries the sparse table for features.

`tf.SparseTensor` data type is not currently supported. If you have a `tf.SparseTensor`, convert it to a `tf.Tensor` first. Example code:

```python
sparse_ids = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
dense_ids = tf.sparse.to_dense(sparse_ids, default_value=0)
```

**Function Prototype**

```python
def embedding_lookup(
    emb_table: Union[StaticEmbTable],
    ids: tf.Tensor,
):
```

**Parameters**

| Parameter      | Type       | Mandatory/Optional| Description                                |
|-----------|-----------|------|------------------------------------|
| emb_table | Sparse table instance    | Mandatory  | Sparse table instance obtained through the `get_embedding_table` API.|
| ids       | tf.Tensor | Mandatory  | Keys to look up.                     |

**Returns**

- Success: The queried tensor result is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# Mxrec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
)

# Embedding lookup.
ids = tf.convert_to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.int64)
embedding = mxrec.embedding_lookup(table, ids)
```

## `get_sparse_embedding`

**Description**

Obtains the trainable parameters of created sparse tables for optimizer gradient calculation and updates.

**Function Prototype**

```python
def get_sparse_embedding():
```

**Returns**

- Success: A list of trainable parameters of the created sparse tables is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
)
init_hashtable_op = mxrec.get_init_hashtable_op()
# The model's loss.
loss = ...
sparse_optimizer = mxrec.AdamWOptimizer(learning_rate=0.01)
sparse_embeddings = mxrec.get_sparse_embedding()
sparse_grads = tf.gradients(loss, sparse_embeddings)
train_ops = sparse_optimizer.apply_gradients(zip(sparse_grads, sparse_embeddings))
```

## `get_init_hashtable_op`

**Description**

Obtains the list of initialization operators for the sparse tables. This operator list must be run before the computational graph executes.

**Function Prototype**

```python
def get_init_hashtable_op():
```

**Returns**

- Success: The list of initialization operators for the sparse tables is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
    initializer=tf.truncated_normal_initializer(),
    key_dtype=tf.int64,
    value_dtype=tf.float32,
)
init_hashtable_op = mxrec.get_init_hashtable_op()
with tf.compat.v1.Session as sess:
    sess.run(init_hashtable_op)
    sess.run(tf.compat.v1.global_variables_initializer())
```

## `get_existing_tables`

**Description**

Obtains all currently created sparse table objects.

**Function Prototype**

```python
def get_existing_tables():
```

**Returns**

- Success: All currently created sparse table objects are returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
    initializer=tf.truncated_normal_initializer(),
    key_dtype=tf.int64,
    value_dtype=tf.float32,
)

# Get all created embedding table instance.
emb_tables = mxrec.get_existing_table()
```

## `EmbeddingTableSaver`

**Type Description**

The `EmbeddingTableSaver` class saves and restores sparse tables (NPU device implementation) in TensorFlow-based models. It provides a convenient interface to save and restore sparse table data.

### `__init__`

**Description**

Initializes an `EmbeddingTableSaver` object.

**Function Prototype**

```python
def __init__(self, emb_tables: List[BaseEmbeddingTable]):
```

**Parameters**

| Parameter       | Type                      | Mandatory/Optional| Description                                          |
|------------|--------------------------|------|----------------------------------------------|
| emb_tables | List[BaseEmbeddingTable] | Mandatory  | Sparse table objects to save or restore, obtained through the `get_existing_tables` API.|

> [!NOTE]NOTE
>
> - - `BaseEmbeddingTable` represents a sparse table instance, which can be obtained through the `get_embedding_table` API.

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
    initializer=tf.truncated_normal_initializer(),
    key_dtype=tf.int64,
    value_dtype=tf.float32,
)

# Get all created embedding table instance.
emb_tables = mxrec.get_existing_table()

emb_table_saver = mxrec.EmbeddingTableSaver(emb_tables)
```

### `save`

**Description**

Saves sparse tables (NPU implementation).

**Function Prototype**

```python
def save(self, sess: tf.compat.v1.Session, save_path: str, global_step: int):
```

**Parameters**

| Parameter        | Type        | Mandatory/Optional| Description                      |
|-------------|------------|-------|--------------------------|
| sess        | tf.Session | Mandatory   | The currently running TensorFlow session.|
| save_path   | str        | Mandatory   | Path to save the data.                    |
| global_step | int        | Mandatory   | Step number for the save. The value range is [0, 2^32-1].    |

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
    initializer=tf.truncated_normal_initializer(),
    key_dtype=tf.int64,
    value_dtype=tf.float32,
)

# Get all created embedding table instance.
emb_tables = mxrec.get_existing_table()

emb_table_saver = mxrec.EmbeddingTableSaver(emb_tables)
with tf.compat.v1.Session() as sess:
    # run your session
    emb_table_saver.save(sess, save_path, global_step)
```

### load

**Description**

Loads sparse tables (NPU implementation).

**Function Prototype**

```python
def load(self, sess: tf.compat.v1.Session, save_path: str, global_step: int):
```

**Parameters**

| Parameter        | Type        | Mandatory/Optional| Description                      |
|-------------|------------|-------|--------------------------|
| sess        | tf.Session | Mandatory   | The currently running TensorFlow session.|
| save_path   | str        | Mandatory   | Path to load the data from.                    |
| global_step | int        | Mandatory   | Step number for the load. The value range is [0, 2^32-1].    |

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

```python
import mxrec
import tensorflow as tf

# MxRec init.
mxrec.init("toml_path")

# Create an embedding table.
table = mxrec.get_embedding_table(
    name="example_name",
    dimension=8,
    device_vocabulary_size=10000,
    initializer=tf.truncated_normal_initializer(),
    key_dtype=tf.int64,
    value_dtype=tf.float32,
)

# Get all created embedding table instance.
emb_tables = mxrec.get_existing_table()

emb_table_saver = mxrec.EmbeddingTableSaver(emb_tables)
with tf.compat.v1.Session() as sess:
    emb_table_saver.load(sess, save_path, global_step)
    # run your session
```
