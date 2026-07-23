# Automatic Graph Modification

## `get_initializer`

**Description**

Obtains the initialization operator (`Operation`) of `tensorflow.data.Iterator`. You must run this operator using `sess.run()` to initialize `Iterator`.

**Function Prototype**

```python
from mx_rec.util.initialize import ConfigInitializer
ConfigInitializer.get_instance().train_params_config.get_initializer(is_training)
```

**Parameters**

|Parameter|Type|Mandatory(Yes/No)|Description|
|--|--|--|--|
|is_training|bool|Yes|Specifies whether the training mode is used. <li>`True`: training mode </li><li>`False`: evaluation mode</li>|

**Returns**

- Success: Returns the TensorFlow operator (`tf.Operation`) for initializing `Iterator`.
- Failure: Throws an exception.

**Example**

```python
import tensorflow as tf
from mx_rec.util.initialize import ConfigInitializer
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
# For training, automatic graph modification must be enabled.
# In training mode, automatic graph modification must be performed after gradients are calculated.
# (The implementation details of gradient calculation are omitted.)
modify_graph_and_start_emb_cache(dump_graph=True)
with tf.compat.v1.Session() as sess:
    # Ensure that the modify_graph_and_start_emb_cache() interface has been called.
    initializer = ConfigInitializer.get_instance().train_params_config.get_initializer(True)
    sess.run(initializer)
```

**References**

For the interface call process and examples, see [Automatic Graph Modification](../appendix.md#automatic-graph-modification).

## `LookupSubgraphSlicerHook`

**Description**

This hook is used to find operators of a specified type in the subgraph of the table lookup `KeyTensor`. It then switches the identified operators and their minimum dependency subgraphs to the CPU prefetch stage for execution. If no operator of the target type is found, no slicing operation is performed.

>[!NOTE]
>This hook is used in `NPUEstimator` mode with the automatic graph modification function enabled. Call this hook before `GraphModifierHook` of the automatic graph modification.

**Parameters**

|Parameter|Type|Mandatory (Yes/No)|Description|
|--|--|--|--|
|op_types|list[str]|Yes|Specifies the types of operators to be sliced. Currently, only the list format is supported.|

**Example**

```python
from mx_rec.graph import LookupSubgraphSlicerHook, GraphModifierHook


def input_fn():
    """
    User-defined Estimator input function
    """

lookup_slicer_hook = LookupSubgraphSlicerHook(op_types=["StringToNumber"])
modifier_hook = GraphModifierHook(modify_graph=params.modify_graph)
hooks_list = [lookup_slicer_hook, modifier_hook]

est = NPUEstimator(...)
est.train(input_fn=lambda: input_fn, hooks=npu_hooks_append(hooks_list))
```

## `OrphanLookupKeySlicerHook`

**Description**

This hook is used to support orphan key types that cannot find a `Dataset` upwards during sparse table lookup. It primarily extends the sparse table lookup function in automatic graph modification mode. If no operator of the target type is found, no slicing operation is performed.

>[!NOTE]
>This hook is used in `NPUEstimator` mode with the automatic graph modification function enabled. Call this hook before `GraphModifierHook` of the automatic graph modification.

**Example**

```python
from mx_rec.graph import OrphanLookupKeySlicerHook, GraphModifierHook

def input_fn():
    """
    User-defined Estimator input function
    """

orphan_slicer_hook = OrphanLookupKeySlicerHook()
modifier_hook = GraphModifierHook(modify_graph=params.modify_graph)
hooks_list = [orphan_slicer_hook, modifier_hook]

est = NPUEstimator(...)
est.train(input_fn=lambda: input_fn, hooks=npu_hooks_append(hooks_list))
```

## `do_merge_lookup`

**Description**

This interface is used to merge lookups for tables queried multiple times in automatic graph modification mode.

In a model, this function is performed using a patch in `Optimizer.compute_gradients()` to ensure that the correct gradients and calculation graph are available during training. During evaluation, it is performed during the graph modification stage.

**Function Prototype**

```python
from mx_rec.graph.merge_lookup import do_merge_lookup
```

**Parameters**

|Parameter|Type|Mandatory (Yes/No)|Description|
|--|--|--|--|
|is_train|bool|Yes|Specifies whether training mode is currently enabled. <li>`True`: training mode </li><li>`False`: evaluation mode</li>|

**Example**

For example, in training mode, if you use `tf.gradients` for all gradient calculations, you must call `do_merge_lookup` manually.

```python
from mx_rec.graph.merge_lookup import do_merge_lookup
do_merge_lookup(is_train=True)
sparse_grads = tf.gradients(loss, sparse_variables)
grads_and_vars = [(grad, variable) for grad, variable in zip(sparse_grads, sparse_variables)]
optimizer.apply_gradients(grads_and_vars)
```
