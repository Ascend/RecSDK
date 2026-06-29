# Optimizers

## `AdamWOptimizer`

**Type Description**

Customized AdamW optimizer, which is used to calculate and update the gradients of sparse tables (NPU implementation).

### `\_\_init\_\_`

**Description**

Initializes an `AdamWOptimizer` object.

**Function Prototype**

```python
def __init__(
    self,
    learning_rate: Union[float, tf.Tensor] = 0.01,
    weight_decay: Union[float, tf.Tensor] = 0.004,
    beta_1: Union[float, tf.Tensor] = 0.9,
    beta_2: Union[float, tf.Tensor] = 0.999,
    epsilon: Union[float, tf.Tensor] = 1e-8,
    name: str = "AdamWOptimizer",
):
```

**Parameters**

| Parameter          | Type                | Mandatory/Optional| Description                                       |
|---------------|-------------------------|-------|-------------------------------------------|
| learning_rate | Union[float, tf.Tensor] | Optional   | Learning rate. The default value is 0.01. The value range is [0.0, 10.0].           |
| weight_decay  | Union[float, tf.Tensor] | Optional   | Weight decay. The default value is 0.004. The value range is [0.0, 10.0].         |
| beta_1        | Union[float, tf.Tensor] | Optional   | Exponential decay rate for the first moment estimate. The default value is 0.9. The value range is [0.0, 1.0).      |
| beta_2        | Union[float, tf.Tensor] | Optional   | Exponential decay rate for the second moment estimate. The default value is 0.999. The value range is [0.0, 1.0).    |
| epsilon       | Union[float, tf.Tensor] | Optional   | Value added to the denominator to improve numerical stability. The default value is 1e-8. The value range is [0.0, 1.0).|
| name| str                     | Optional   | Name of the optimizer. The default value is `"AdamWOptimizer"`. The length range is [1, 128]. |

**Returns**

- Success: An `AdamWOptimizer` instance is returned.
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

# The loss of model.
loss = ...
sparse_optimizer = mxrec.AdamWoptimizer(learning_rate=0.01)
sparse_embeddings = mxrec.get_sparse_embedding()
sparse_grads = tf.gardients(loss, sparse_embeddings)
train_ops = sparse_optimizer.apply_gradients(zip(sparse_grads, sparse_embeddings))
with tf.compat.v1.Session() as sess:
    sess.run(init_hashtable_op)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run([loss, train_ops])
```

>[!NOTE]
>
>- For `apply_gradients`, refer to the usage of the open-source TensorFlow optimizer.
