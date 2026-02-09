# 优化器<a name="ZH-CN_TOPIC_0000001629887049"></a>

## AdamWOptimizer<a name="ZH-CN_TOPIC_0000001579847272"></a>

**类型描述<a name="section634582619155"></a>**

自定义AdamW优化器，用于计算和更新稀疏表（NPU实现）的梯度。

### \_\_init\_\_<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

初始化`AdamWOptimizer`对象。

**函数原型<a name="section1483104721911"></a>**

```bash
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

**参数说明<a name="section888634319218"></a>**

| 参数名           | 类型                      | 可选/必选 | 说明                                        |
|---------------|-------------------------|-------|-------------------------------------------|
| learning_rate | Union[float, tf.Tensor] | 可选    | 学习率，默认值为0.01，取值范围：[0.0, 10.0]。            |
| weight_decay  | Union[float, tf.Tensor] | 可选    | 权重衰减值，默认值为0.004，取值范围：[0.0, 10.0]          |
| beta_1        | Union[float, tf.Tensor] | 可选    | 第一矩的指数衰减率估计，默认值为0.9，取值范围：[0.0, 1.0)       |
| beta_2        | Union[float, tf.Tensor] | 可选    | 第二矩的指数衰减率估计，默认值为0.999，取值范围：[0.0, 1.0)     |
| epsilon       | Union[float, tf.Tensor] | 可选    | 加入此值到分母中，提高数据稳定性，默认值为1e-8，取值范围：[0.0, 1.0) |
| name| str                     | 可选    | 优化器名称，默认值为"AdamWOptimizer"，长度范围：[1, 128]  |

**返回值说明<a name="section651195312311"></a>**

-   成功：`AdamWOptimizer`实例
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
>[!NOTE] 说明 
>-   `apply_gradients`方法可参考开源Tensorflow优化器的用法。
