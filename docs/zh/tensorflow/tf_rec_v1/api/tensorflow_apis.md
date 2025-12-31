# TensorFlow相关接口<a name="ZH-CN_TOPIC_0000001749200933"></a>

## TensorFlow接口说明<a name="ZH-CN_TOPIC_0000001749081125"></a>

本章节列出Rec SDK TensorFlow基于TensorFlow框架patch修改的接口。更多TensorFlow原生接口信息请参考TensorFlow官网。

>[!NOTICE] 须知 
>对于用户集成的开源和第三方软件，漏洞和问题请自行跟踪社区并及时进行修复。本章中涉及的TensorFlow原生方法若存在漏洞，请参照TensorFlow官网社区中的安全建议进行规避和修复。


## tf.compat.v1.train.Saver.save<a name="ZH-CN_TOPIC_0000001630046405"></a>

**功能描述<a name="section634582619155"></a>**

TensorFlow用于模型保存的接口。

**函数原型<a name="section1483104721911"></a>**

```bash
def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix="meta", write_meta_graph=True, write_state=True, strip_default_attrs=False, save_debug_info=False, is_incremental_checkpoint=False, save_delta=False)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|sess|Session|必选|需要保存模型的Session。|
|save_path|str|必选|模型保存路径。支持本地文件系统和HDFS文件系统，长度范围为[1, 1024]。<li>使用HDFS路径保存数据时，日志中会打印“hdfsWrite: FSDataOutputStream#write error”的日志，该日志不影响数据保存，可忽略。</li><li>多节点使用HDFS进行训练时，要使用相同的HDFS路径作为保存路径。</li><li>​在使用多卡训练时，save_path不需要传入带卡ID的参数，多卡训练的结果将自动合并保存在save_path中。在后续多卡进行模型加载时，也会自动加载属于本卡的参数。</li>|
|global_step|int, np.int64|可选|在checkpoint文件名补充训练步数，默认值为None，取值范围为[0, 2147483647]。|
|latest_filename|str|可选|protocol buffer文件的可选名称，该文件将包含最新checkpoint列表，默认为None。长度范围为[1, 50]。|
|meta_graph_suffix|str|可选|MetaGraphDef文件的后缀，默认为meta，长度范围为[1, 50]。|
|write_meta_graph|bool|可选|是否写入MetaGraph文件，默认为True。<br>取值范围：<li>True：写入MetaGraph文件。</li><li>False：不写入MetaGraph文件。</li>|
|write_state|bool|可选|是否写入CheckpointStateProto文件，默认为True。<br>取值范围：<li>True：写入CheckpointStateProto文件。</li><li>False：不写入CheckpointStateProto文件。</li>|
|strip_default_attrs|bool|可选|保存模型文件时，是否删除NodeDefs中的默认值属性，默认为False。<li>参数值为True，则默认值属性将在接口调用时从NodeDefs中删除。</li><li>参数值为False，则不进行删除操作。</li>|
|save_debug_info|bool|可选|是否保存Debug信息，默认为False。<li>参数值为True，则将图形调试信息保存到一个单独的文件中，该文件位于save_path对应的目录中，并在生成文件的扩展名之前添加“_debug”。仅当write_meta_graph为True时，此功能才会生效。</li><li>参数值为False，则不保存Debug信息。</li>|
|is_incremental_checkpoint|bool|可选|是否开启模型增量保存与加载，默认为False。<li>True：开启模型增量保存与加载。</li><li>False：关闭模型增量保存与加载。</li>|
|save_delta|bool|可选|是否保存增量模型。<li>True：保存增量模型。</li><li>False：不保存增量模型，保存全量模型。</li>|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回“model\_checkpoint\_path”，即模型保存路径。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

具体使用方法可参考[Rec SDK](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/examples/demo/little_demo/run_mode.py)代码仓中的little demo，以下仅提供一个使用的流程示例。

```bash
# 1、导入需要的库
import tensorflow as tf
from mx_rec.util.initialize import init, get_rank_id
# 2、构建计算图
# ...
# 3、创建saver
saver = tf.compat.v1.train.Saver() 
# 4、获取rank_id
rank_id = get_rank_id()
# 5、设置需要保存模型时的训练步数
global_step = 200
with tf.compat.v1.Session() as sess:
    saver.save(sess, f"./saved-model/model-{rank_id}", global_step=global_step)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


## tf.compat.v1.train.Saver.restore<a name="ZH-CN_TOPIC_0000001580166484"></a>

**功能描述<a name="section634582619155"></a>**

TensorFlow用于模型加载的接口。

**函数原型<a name="section1483104721911"></a>**

```bash
def restore(self, sess, save_path)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|sess|Session|必选|需要导入模型TensorFlow的Session。|
|save_path|str|必选|<li>模型checkpoint文件的保存路径。</li><li>支持本地文件系统和HDFS文件系统，长度范围为[1,1024]。</li><li>在使用多卡训练加载模型时，多卡save_path可以输入同一加载路径（该路径下保存了多卡训练的结果），各卡会自动加载属于本卡的参数。</li>[!NOTE] 说明<br>当前加载文件单个大小上限为500G，并发读取可能会引发系统OOM。|


**返回值说明<a name="section651195312311"></a>**

-   成功：None。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

具体使用方法可参考[Rec SDK](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/examples/demo/little_demo/run_mode.py)中的little demo，以下仅提供一个使用的流程示例。

```bash
# 1、导入需要的库
import tensorflow as tf
from mx_rec.util.initialize import init, get_rank_id
# 2、构建计算图
# ...
# 3、创建saver
saver = tf.compat.v1.train.Saver() 
# 4、获取rank_id
rank_id = get_rank_id()
# 5、设置需要加载的模型保存时的训练步数，比如：
latest_step = 200
with tf.compat.v1.Session() as sess:
    saver.restore(sess, f"./saved-model/model-{rank_id}-{latest_step}")
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


## tensorflow.python.client.session.BaseSession.run<a name="ZH-CN_TOPIC_0000001701361128"></a>

**功能描述<a name="section867474604710"></a>**

TensorFlow执行计算图的方法。

**函数原型<a name="section36612317482"></a>**

```bash
def run(self, fetches, feed_dict=None, options=None, run_metadata=None)
```

**参数说明<a name="section02191743124812"></a>**

|参数|类型|可选/必选|说明|
|--|--|--|--|
|fetches|<li>str</li><li>tf.Operation</li><li>tf.Variable</li><li>tf.Tensor</li><li>tf.sparse.SparseTensor</li><li>list</li><li>tuple</li><li>dict</li>|必选|运行操作或者获取其中的Tensor。|
|feed_dict|<li>tf.Variable</li><li>tf.Tensor</li><li>tf.sparse.SparseTensor</li><li>list</li><li>tuple</li><li>dict</li>|可选|覆盖图中Tensor的值。|
|options|tf.compat.v1.RunOptions|可选|控制特定步骤的行为。|
|run_metadata|tf.compat.v1.RunMetadata|可选|在特定步骤时，收集非张量输出。|


**返回值说明<a name="section191731546399"></a>**

-   成功：如果fetches是单个元素，则为单个值；如果fetches是list，则为值list；如果fetches是dict，则为具有相同键的dict。
-   失败：抛出异常。

**使用示例<a name="section204450261015"></a>**

以下仅提供使用流程的示例。

```bash
# 1、导入需要的库
import tensorflow as tf
from mx_rec.util.initialize import init
# 2、构建计算图
# ...
# 3、调用接口训练
with tf.compat.v1.Session() as sess:
     sess.run([train_ops])      #train_ops为构建计算图中构建的训练算子
```


