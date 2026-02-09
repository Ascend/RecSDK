# 模型接口<a name="ZH-CN_TOPIC_0000001630246509"></a>

## get\_embedding\_table<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

创建稀疏表。

**函数原型<a name="section1483104721911"></a>**

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

**参数说明<a name="section888634319218"></a>**

| 参数名                    | 类型                | 可选/必选 | 说明                                                                                                   |
|------------------------|-------------------|------|------------------------------------------------------------------------------------------------------|
| name                   | str               | 必选   | 稀疏表名，只能包含[0-9A-Za-z_.]，表名长度范围：[1, 128]。                                                              |
| dimension              | int               | 必选   | 稀疏表的embedding维度，取值范围：[1, 512]。                                                                       |
| device_vocabulary_size | int               | 必选   | Device侧稀疏表容量，取值范围：[1, 10**9]。请保证内存和磁盘空间足够，根据服务器的实际配置进行设置。                                            |
| initializer            | Tensorflow的初始化器类型 | 可选   | 稀疏表初始值生成器，默认值为随机正态分布初始化器。                                                                            |
| key_dtype              | tf.int64          | 可选   | 稀疏特征key数据类型，默认值为tf.int64，可选类型仅限于tf.int64。                                                            |
| value_dtype            | tf.float32        | 可选   | 稀疏特征value数据类型，默认值为tf.float32，可选类型仅限于tf.float32。                                                      |
| distribution_strategy  | str               | 可选   | 稀疏表分布式并行模式，默认值为"MP"（模型并行），当前仅支持"MP"。                                                                 |
| min_used_times         | int / None        | 可选   | 特征准入功能，ID历史出现次数必须超过该值才能生效，否则使用该ID查表返回默认的Embedding值，取值范围[0, 2^31-1]。                                  |
| max_cold_secs          | int / None        | 可选   | 特征淘汰功能，该值用于表示ID和Embedding自最后一次被访问后，最久能够容忍的未访问时长。超过这个阈值的ID将会在保存时，从存储Embedding的哈希表中删除，取值范围[0, 2^64-1]。 |

>[!NOTE] 说明 
>-   当前仅支持非扩容模式，即`device_vocabulary_size`需大于0。

**返回值说明<a name="section651195312311"></a>**

-   成功：返回稀疏表实例
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

## embedding\_lookup<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

稀疏表特征查询接口。

暂不支持`tf.SparseTensor`数据类型，若为`tf.SparseTensor`需转成`tf.Tensor`。示例代码如下：
```python
sparse_ids = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
dense_ids = tf.sparse.to_dense(sparse_ids, default_value=0)
```

**函数原型<a name="section1483104721911"></a>**

```python
def embedding_lookup(
    emb_table: Union[StaticEmbTable],
    ids: tf.Tensor,
):
```

**参数说明<a name="section888634319218"></a>**

| 参数名       | 类型        | 可选/必选 | 说明                                 |
|-----------|-----------|------|------------------------------------|
| emb_table | 稀疏表实例     | 必选   | 稀疏表实例，通过`get_embedding_table`接口得到。 |
| ids       | tf.Tensor | 必选   | 待查询的关键字（key）。                      |

**返回值说明<a name="section651195312311"></a>**

-   成功：返回查询到Tensor结果
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

## get\_sparse\_embedding<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

获取已创建稀疏表的可训练参数，用于优化器计算/更新梯度。

**函数原型<a name="section1483104721911"></a>**

```python
def get_sparse_embedding():
```

**返回值说明<a name="section651195312311"></a>**

-   成功：返回已创建稀疏表的可训练参数列表
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

## get\_init\_hashtable\_op<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

获取稀疏表的初始化算子列表，计算图执行之前需先执行此算子列表。

**函数原型<a name="section1483104721911"></a>**

```python
def get_init_hashtable_op():
```

**返回值说明<a name="section651195312311"></a>**

-   成功：稀疏表的初始化算子列表
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

## get\_existing\_tables<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

获取当前所有已创建的稀疏表对象。

**函数原型<a name="section1483104721911"></a>**

```python
def get_existing_tables():
```

**返回值说明<a name="section651195312311"></a>**

-   成功：当前所有已创建的稀疏表对象
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

## EmbeddingTableSaver<a name="ZH-CN_TOPIC_0000001630246521"></a>

**类型描述<a name="section634582619155"></a>**

`EmbeddingTableSaver`类被设计用于在基于`Tensorflow`框架的模型中管理稀疏表（NPU设备实现）的保存和恢复。它提供了一个方便的接口来保存和恢复稀疏表的数据。

### \_\_init\_\_<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

初始化`EmbeddingTableSaver`对象。

**函数原型<a name="section1483104721911"></a>**

```python
def __init__(self, emb_tables: List[BaseEmbeddingTable]):
```

**参数说明<a name="section888634319218"></a>**

| 参数名        | 类型                       | 可选/必选 | 说明                                           |
|------------|--------------------------|------|----------------------------------------------|
| emb_tables | List[BaseEmbeddingTable] | 必选   | 待执行保存恢复操作的稀疏表对象，通过`get_existing_tables`接口得到。 |

>[!NOTE] 说明 
>-   `BaseEmbeddingTable`表示稀疏表实例，可通过`get_embedding_table`接口获取。

**返回值说明<a name="section651195312311"></a>**

-   成功：None
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

### save<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

执行稀疏表（NPU实现）的保存操作。

**函数原型<a name="section1483104721911"></a>**

```python
def save(self, sess: tf.compat.v1.Session, save_path: str, global_step: int):
```

**参数说明<a name="section888634319218"></a>**

| 参数名         | 类型         | 可选/必选 | 说明                       |
|-------------|------------|-------|--------------------------|
| sess        | tf.Session | 必选    | Tensorflow当前执行中的session。 |
| save_path   | str        | 必选    | 保存路径                     |
| global_step | int        | 必选    | 保存步数，取值范围[0, 2^32-1]     |

**返回值说明<a name="section651195312311"></a>**

-   成功：None
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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

### load<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

执行稀疏表（NPU实现）的加载操作。

**函数原型<a name="section1483104721911"></a>**

```python
def load(self, sess: tf.compat.v1.Session, save_path: str, global_step: int):
```

**参数说明<a name="section888634319218"></a>**

| 参数名         | 类型         | 可选/必选 | 说明                       |
|-------------|------------|-------|--------------------------|
| sess        | tf.Session | 必选    | Tensorflow当前执行中的session。 |
| save_path   | str        | 必选    | 加载路径                     |
| global_step | int        | 必选    | 加载步数，取值范围[0, 2^32-1]     |

**返回值说明<a name="section651195312311"></a>**

-   成功：None
-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

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