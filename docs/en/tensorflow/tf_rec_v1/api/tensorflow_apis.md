# TensorFlow-Related APIs

## Description

This section describes the Rec SDK TensorFlow APIs modified based on the TensorFlow framework patches. For more information about native TensorFlow APIs, visit the official TensorFlow website. For details about the support of the Ascend AI Processor for TensorFlow APIs, see the official documents [TensorFlow 1.15](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/API/tfadapter1x/tfmigr1_tfadapi_0131.html) and [TensorFlow 2.6](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/API/tfadapter2x/tfmigr2_tfadapi_0029.html).

>[!NOTICE]
>
>For open-source and third-party software that you integrate, track vulnerabilities and issues in the community and fix them in a timely manner. If the native TensorFlow methods involved in this chapter have vulnerabilities, refer to the security suggestions in the official TensorFlow community for mitigation and remediation.
>

## `tf.compat.v1.train.Saver.save`

**Description**

TensorFlow API used to save models.

**Function Prototype**

```python
def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix="meta", write_meta_graph=True, write_state=True, strip_default_attrs=False, save_debug_info=False, is_incremental_checkpoint=False, save_delta=False)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|sess|Session|Mandatory|Session in which the model is saved.|
|save_path|str|Mandatory|Path to save the model. Local file systems and HDFS file systems are supported. The path can contain 1 to 1024 characters. <li>When you use HDFS paths to save data, the log `hdfsWrite: FSDataOutputStream#write error` is printed. This log does not affect data saving and can be ignored. </li><li>When you use HDFS for multi-node training, use the same HDFS path as the saving path. </li><li>In multi-device training, `save_path` does not need to include card ID parameters. Results from multi-device training are automatically merged and saved in `save_path`. During subsequent model loading for multi-device training, parameters belonging to the current card are also automatically loaded.</li>|
|global_step|int, np.int64|Optional|Training steps added to the checkpoint filename, defaulting to `None`. The value range is [0, 2147483647].|
|latest_filename|str|Optional|Optional name for the protocol buffer file that contains the list of the latest checkpoints, defaulting to `None`. The length range is [1, 50].|
|meta_graph_suffix|str|Optional|Suffix for the MetaGraphDef file, defaulting to `meta`. The length range is [1, 50].|
|write_meta_graph|bool|Optional|Specifies whether to write the MetaGraph file, defaulting to `True`.<br>Value range: <li>`True`: Writes the MetaGraph file. </li><li>`False`: Does not write the MetaGraph file.</li>|
|write_state|bool|Optional|Specifies whether to write the CheckpointStateProto file, defaulting to `True`.<br>Value range: <li>`True`: Writes the CheckpointStateProto file. </li><li>`False`: Does not write the CheckpointStateProto file.</li>|
|strip_default_attrs|bool|Optional|Specifies whether to remove default value attributes from NodeDefs when saving model files, defaulting to `False`. <li>If the parameter is set to `True`, default value attributes are removed from NodeDefs when the API is called. </li><li>If the parameter is set to `False`, no removal operation is performed.</li>|
|save_debug_info|bool|Optional|Specifies whether to save debug information, defaulting to `False`. <li>If the parameter is set to `True`, graph debugging information is saved to a separate file in the directory corresponding to `save_path`, with `_debug` added before the generated file name extension. This feature takes effect only when `write_meta_graph` is `True`. </li><li>If the parameter is set to `False`, debug information is not saved.</li>|
|is_incremental_checkpoint|bool|Optional|Specifies whether to enable incremental model saving and loading. The default value is `False`. <li>`True`: Enables incremental model saving and loading. </li><li>`False`: Disables incremental model saving and loading.</li>|
|save_delta|bool|Optional|Specifies whether to save incremental models. <li>`True`: Saves incremental models. </li><li>`False`: Does not save incremental models but saves full models.</li>|

**Returns**

- Success: `model_checkpoint_path` is returned, which is the model saving path.
- Failure: An exception is thrown.

**Example**

For specific usage, see the little demo in the [Rec SDK](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/examples/demo/little_demo/run_mode.py) repository. The following provides a workflow example.

```python
# 1. Import required libraries.
import tensorflow as tf
from mx_rec.util.initialize import init, get_rank_id
# 2. Build the computational graph.
# ...
# 3. Create a saver.
saver = tf.compat.v1.train.Saver()
# 4. Obtain the rank ID.
rank_id = get_rank_id()
# 5. Set the training steps for model saving.
global_step = 200
with tf.compat.v1.Session() as sess:
    saver.save(sess, f"./saved-model/model-{rank_id}", global_step=global_step)
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `tf.compat.v1.train.Saver.restore`

**Description**

TensorFlow API used to load models.

**Function Prototype**

```python
def restore(self, sess, save_path)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|sess|Session|Mandatory|TensorFlow Session in which the model is imported.|
|save_path|str|Mandatory|<li>Path of the saved model checkpoint file. </li><li>Local file systems and HDFS file systems are supported. The path can contain 1 to 1024 characters. </li><li>In multi-device training, `save_path` can be set to the same loading path (where the results of multi-device training are saved), and each card automatically loads the parameters belonging to it.</li><br>Currently, the maximum size for a single loaded file is 500 GB. Concurrent reading may cause out-of-memory (OOM).|

**Returns**

- Success: No value is returned.
- Failure: An exception is thrown.

**Example**

For specific usage, see the little demo in the [Rec SDK](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/examples/demo/little_demo/run_mode.py) repository. The following provides a workflow example.

```python
# 1. Import required libraries.
import tensorflow as tf
from mx_rec.util.initialize import init, get_rank_id
# 2. Build the computational graph.
# ...
# 3. Create a saver.
saver = tf.compat.v1.train.Saver()
# 4. Obtain the rank ID.
rank_id = get_rank_id()
# 5. Set the training steps for model loading. For example:
latest_step = 200
with tf.compat.v1.Session() as sess:
    saver.restore(sess, f"./saved-model/model-{rank_id}-{latest_step}")
```

**References**

For the interface calling process and examples, see [Migration and Training](../migration_and_training.md).

## `tensorflow.python.client.session.BaseSession.run`

**Description**

TensorFlow method used to execute the computation graph.

**Function Prototype**

```python
def run(self, fetches, feed_dict=None, options=None, run_metadata=None)
```

**Parameters**

|Parameter|Type|Mandatory/Optional|Description|
|--|--|--|--|
|fetches|<li>str</li><li>tf.Operation</li><li>tf.Variable</li><li>tf.Tensor</li><li>tf.sparse.SparseTensor</li><li>list</li><li>tuple</li><li>dict</li>|Mandatory|Run operations or fetch tensors from them.|
|feed_dict|<li>tf.Variable</li><li>tf.Tensor</li><li>tf.sparse.SparseTensor</li><li>list</li><li>tuple</li><li>dict</li>|Optional|Overrides the values of tensors in the graph.|
|options|tf.compat.v1.RunOptions|Optional|Controls the behavior for a specific step.|
|run_metadata|tf.compat.v1.RunMetadata|Optional|Collects non-tensor outputs during a specific step.|

**Returns**

- Success: A single value is returned if `fetches` is a single element. A list of values is returned if `fetches` is a list. A dictionary with the same keys is returned if `fetches` is a dictionary.
- Failure: An exception is thrown.

**Example**

The following provides a workflow example.

```python
# 1. Import required libraries.
import tensorflow as tf
from mx_rec.util.initialize import init
# 2. Build the computational graph.
# ...
# 3. Call the API for training.
with tf.compat.v1.Session() as sess:
     sess.run([train_ops])      # train_ops is the training operator constructed in the computational graph.
```
