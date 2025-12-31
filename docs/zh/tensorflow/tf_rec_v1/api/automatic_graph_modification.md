# 自动改图<a name="ZH-CN_TOPIC_0000001630246497"></a>

## get\_initializer<a name="ZH-CN_TOPIC_0000001630246481"></a>

**功能描述<a name="section634582619155"></a>**

获取tensorflow.data.Iterator的初始化算子（Operation），该算子需要通过使用sess.run\(\)来初始化Iterator。

**函数原型<a name="section1483104721911"></a>**

```bash
from mx_rec.util.initialize import ConfigInitializer
ConfigInitializer.get_instance().train_params_config.get_initializer(is_training)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|is_training|bool|必选|是否为训练模式。<li>True：训练（train）模式。</li><li>False：评估（eval）模式。</li>|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回初始化Iterator的TensorFlow算子（tf.Operation）。
-   失败：抛出异常。

**使用示例<a name="section193151694205"></a>**

```bash
import tensorflow as tf
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
# train，需要开启自动改图
# train模式下，自动改图需要在计算梯度之后
计算梯度........
modify_graph_and_start_emb_cache(dump_graph=True)
with tf.compat.v1.Session() as sess:
    # 请确保已调用过modify_graph_and_start_emb_cache()接口
    initializer = ConfigInitializer.get_instance().train_params_config.get_initializer(True)
    sess.run(initializer)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[自动改图](../appendix.md#自动改图)。


## LookupSubgraphSlicerHook<a name="ZH-CN_TOPIC_0000001911330801"></a>

该Hook用于在查表KeyTensor的子图中查找指定类型算子，然后将查找到的指定类型算子及其最小依赖子图切换到CPU预取阶段执行。如果没有找到目标类型的算子，不会执行切分操作。

>[!NOTE] 说明 
>该Hook的使用场景是NPUEstimator模式，启用自动改图功能。该Hook的调用时机需要在自动改图的GraphModifierHook之前。

|参数名|类型|必选/可选|参数说明|
|--|--|--|--|
|op_types|list[str]|必选|指定需要切分的算子类型，目前仅支持列表格式。|


**使用示例<a name="section11333109979"></a>**

```bash
from mx_rec.graph import LookupSubgraphSlicerHook, GraphModifierHook


def input_fn():
    """
    用户自定义Estimator输入函数。
    """

lookup_slicer_hook = LookupSubgraphSlicerHook(op_types=["StringToNumber"] )
modifier_hook = GraphModifierHook(modify_graph=params.modify_graph)
hooks_list = [lookup_slicer_hook, modifier_hook]

est = NPUEstimator(...)
est.train(input_fn=lambda: input_fn, hooks=npu_hooks_append(hooks_list))
```


## OrphanLookupKeySlicerHook<a name="ZH-CN_TOPIC_0000001865451328"></a>

该Hook用于支持稀疏表查询时传入向上无法找到Dataset的孤儿Key类型，主要用于拓展自动改图模式下的稀疏表查询功能。如果没有找到目标类型的算子，不会执行切分操作。

>[!NOTE] 说明 
>该Hook的使用场景是NPUEstimator模式，启用自动改图功能。该Hook的调用时机需要在自动改图的GraphModifierHook之前。

**使用示例<a name="section473316525914"></a>**

```bash
from mx_rec.graph import OrphanLookupKeySlicerHook, GraphModifierHook

def input_fn():
    """
    用户自定义Estimator输入函数。
    """

orphan_slicer_hook = OrphanLookupKeySlicerHook()
modifier_hook = GraphModifierHook(modify_graph=params.modify_graph)
hooks_list = [orphan_slicer_hook, modifier_hook]

est = NPUEstimator(...)
est.train(input_fn=lambda: input_fn, hooks=npu_hooks_append(hooks_list))
```


## do\_merge\_lookup<a name="ZH-CN_TOPIC_0000001935068449"></a>

**功能描述<a name="section634582619155"></a>**

该接口用于自动改图模式下，对多次查询的表进行lookup合并操作。

在模型中，此函数在Optimizer.compute\_gradients\(\)中利用patch执行，确保train时拥有正确的梯度和计算图；eval时在改图阶段执行。

**函数原型<a name="section1483104721911"></a>**

```bash
from mx_rec.graph.merge_lookup import do_merge_lookup
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|is_train|bool|必选|当前是否为训练模式。<li>True：训练（train）模式。</li><li>False：评估（eval）模式。</li>|


**使用示例<a name="section193151694205"></a>**

例如，train模式，全部的梯度计算都使用tf.gradients，则需要主动调用do\_merge\_lookup。

```bash
from mx_rec.graph.merge_lookup import do_merge_lookup
do_merge_lookup(is_train=True)
sparse_grads = tf.gradients(loss, sparse_variables)
grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
optimizer.apply_gradients(grads_and_vars)
```


