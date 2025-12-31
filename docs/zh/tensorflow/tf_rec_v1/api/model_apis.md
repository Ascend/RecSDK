# 模型接口<a name="ZH-CN_TOPIC_0000001630246509"></a>

## create\_table<a name="ZH-CN_TOPIC_0000001630246521"></a>

**功能描述<a name="section634582619155"></a>**

创建稀疏表。

**函数原型<a name="section1483104721911"></a>**

```bash
def create_table(key_dtype, dim, name, emb_initializer, device_vocabulary_size=1, host_vocabulary_size=0, ssd_vocabulary_size=0, ssd_data_path=(os.getcwd(),), is_save=True, is_dp=False, init_param=1.0, all2all_gradients_op=All2allGradientsOp.SUM_GRADIENTS.value, enable_merge=False, padding_keys=None, padding_keys_mask=False, padding_keys_len=None, value_dtype=tf.float32, shard_num=1, fusion_optimizer_var=True, hashtable_threshold=0)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|key_dtype|TensorFlow的dtype类型|必选|稀疏特征键（key）数据类型。可选类型仅限于tf.int64和tf.int32。|
|dim|<li>int</li><li>tf.Tensorshape</li>|必选|嵌入层（Embedding）维度。取值范围：[1, 8192]。如果dim需要设置为大于512的值，请保证内存和磁盘空间足够或者使用DDR模式，或者减小稀疏表的vocabulary size。<br>如入参为tf.Tensorshape类型，则要求其ndims为1，该值表示嵌入层维度。<br>请根据服务器的实际配置进行设置。|
|name|str|必选|稀疏表的表名，只能包含数字、字母、下划线和特殊符号“.”。表名长度范围：[1, 100]。<br>稀疏表的表名需要保持唯一，不能重复。|
|emb_initializer|TensorFlow的初始化器类型|必选|嵌入层初始值生成器。|
|device_vocabulary_size|int|可选|Device侧嵌入层数量，默认值为1。取值范围：1~10亿。当设置超过25600000时，请保证内存和磁盘空间足够，或者开启片上内存侧动态扩容功能，或者减小稀疏表dim的大小。请根据服务器的实际配置进行设置。<br>如果启用DDR/SSD存储，即host_vocabulary_size不为0，则需要device_vocabulary_size≥连续2个batch去重后的key个数，要求片上内存能存放至少2个batch数据，此时片上内存仅作为cache。|
|host_vocabulary_size|int|可选|Host侧DDR存储的嵌入层数量，默认值为0。取值范围为：0~10亿。<li>取值为0时表示不开启Host侧DDR功能，如果不启用SSD存储，需要确保DDR能存储全量数据；</li><li>不为0时表示开启，此时需要关闭片上内存侧的动态扩容，即use_dynamic_expansion=False，默认使用DDR内存侧的动态扩容模式。如果host_vocabulary_size设置为大于1亿的值，请保证内存和磁盘空间足够，或者减小稀疏表dim的大小。</li>在动态扩容模式下（即use_dynamic_expansion=True时），默认使用片上内存作为唯一存储，此变量会被置0。超过单机内存时会出现OMM。请根据服务器的实际配置进行设置。|
|ssd_vocabulary_size|int|可选|开启SSD存储Embedding数据功能。默认值为“0”表示不开启。当值大于“0”时，要求host_vocabulary_size也大于“0”才能开启该功能。取值范围：0~10亿。<br>在动态扩容模式下（即use_dynamic_expansion=True时），默认使用片上内存作为唯一存储，此变量会被置0。请根据服务器的实际配置进行设置。|
|ssd_data_path|<li>list[str]</li><li>Tuple[str]</li>|可选|默认为当前运行脚本所在路径。<li>当参数为空列表时，默认SSD存储路径为当前运行脚本所在路径。</li><li>当列表非空且路径有效时，将按顺序存储到相应路径中。</li><li>当路径对应磁盘空间不足时，会尝试下一个路径，直到所有磁盘空间不足时抛出异常。</li>|
|is_save|bool|可选|是否保存Embedding数据，默认值为True。<br>取值范围：<li>True：保存Embedding数据。</li><li>False：不保存Embedding数据。</li>|
|is_dp|bool|可选|是否使能稀疏表数据并行功能，默认值为“False”。<br>开启DP模式（is_dp=True）可以将表配置为数据并行，建议在存在10GB级小表，且当NPU侧为端到端瓶颈和稀疏表的NPU ALL2ALL通信量小于16MB时开启，能够带来一定的性能提升，稀疏通信收益15%左右。<br>注意，混用DP和MP模式进行续训时，功能、精度不会受到影响，但不建议使用。|
|init_param|float|可选|Embedding初始化参数系数，默认值为1.0。取值范围：[-10, 10]。<br>当init_param参数设置超过1.0或者小于-1.0时，建议减小[batch_size](class_reference.md#featurespec)，避免显存占用过大导致程序异常。|
|all2all_gradients_op|string|可选|分布式梯度回传后梯度聚合的方式，默认值为sum_gradients。<li>sum_gradients：所有rank的梯度相加。</li><li>sum_gradients_and_div_by_ranksize：所有rank梯度相加除以ranksize。</li>|
|enable_merge|bool|可选|合并EmbeddingTable功能会根据创建时的key_dtype、dim、emb_initializer、is_save、is_dp、init_param、all2all_gradients_op、padding_keys_mask（可选）几个参数是否完全一致确定是否能合并，合并后的EmbeddingTable会有效减少CPU线程、Host与Device之间通信Channel的数量，从而节省系统资源。<br>合并的多个EmbeddingTable需要确定各表的ID相互独立，否则会影响精度（比如存在UserEmbeddingTable和ItemEmbeddingTable两张表待合并，那么UserID和ItemID内不能存在任何相同的值）。<br>当前仅支持片上内存动态扩容模式。<br>启用自动合表需要开启自动改图和一表多查功能。<li>False：默认值，表示不启用自动合并。</li><li>True：表示启用自动合并。</li>|
|padding_keys|<li>int64</li><li>list[int64]</li><li>None</li>|可选|该参数通常为数据集中sparse特征中的key。默认值为None，此时需设置padding_keys_mask=False、padding_keys_len=None，表示正常训练更新。若值为int64/list[int64]，则需设置padding_keys_mask=True、padding_keys_len=shape，其中shape为该key对应数据集中sparse特征的shape，表示该key对应的Embedding不需要更新。|
|padding_keys_mask|bool|可选|该参数用于表示padding_keys对应的Embedding是否需要更新。默认值为False，此时需设置padding_keys=None、padding_keys_len=None，表示正常训练更新；为True时，需设置padding_keys=int64/list[int64]、padding_keys_len=int32，表示不需要更新。|
|padding_keys_len|<li>int32</li><li>None</li>|可选|该参数表示数据集中sparse特征的shape，通常为batch size * 特征向量维度。默认值为None，此时需设置padding_keys=None、padding_keys_mask=False，表示正常训练更新。若值为int32，则需设置padding_keys=int64/list[int64]、padding_keys_mask=True，表示padding_keys对应的Embedding不需要更新。|
|value_dtype|TensorFlow的dtype类型|可选|稀疏特征值（value）数据类型，默认值为tf.float32，仅支持tf.float32。|
|shard_num|int|可选|嵌入层分区数，默认值为1。取值范围：[1,8192]。|
|fusion_optimizer_var|bool|可选|是否使用融合优化参数，默认值为True。<br>取值范围：<li>True：使用融合优化参数。</li><li>False：不使用融合优化参数。</li>|
|hashtable_threshold|int|可选|哈希表阈值，高于阈值时使用哈希表，低于阈值时使用线性表，默认值为0。取值范围：[0,2147483647]。|


>[!NOTE] 说明 
>-   对于“padding\_keys”、“padding\_keys\_mask”和“padding\_keys\_len”：
>>    -   不支持DP模式和lazy adam融合算子模式。
>>   -   所有表都需要设置padding key的情况下，需在初始化接口中设置为静态shape模式，如init\(use\_dynamic=False\)，此时约束模型脚本的“drop\_remainder=True”，如“dataset = dataset.batch\(batch\_size, drop\_remainder=True\)”。
>-   开启DDR/SSD模式必须确保[自动改图](../appendix.md#自动改图)模式也同时开启。
>-   若使用片上内存侧的动态扩容方案，即[init](initialization_and_deinitialization_of_the_training_framework.md#init)接口中传入use\_dynamic\_expansion=True，此时传入的host\_vocabulary\_size和ssd\_vocabulary\_size参数会被设置为“0”，即参数不生效，也可均不填。
>-   若不使用片上内存侧的动态扩容方案，则根据host\_vocabulary\_size和ssd\_vocabulary\_size是否为0，来判断具体存储模式。
>-   当“host\_vocabulary\_size”为“0”时，不开启Host侧DDR功能，不为“0”时开启。所有Embedding表必须保持同时使用Host侧DDR功能或同时不使用Host侧DDR功能，即所有表“host\_vocabulary\_size”参数**同时为“0”**或**同时不为“0”**，否则进行参数校验时会报错，报错信息参考如下。
>    ```bash
>    ValueError: The host-side DDR function of all tables must be used or not used at the same time. However, host voc size of each table is [].
>    ```

**返回值说明<a name="section651195312311"></a>**

-   成功：返回稀疏表实例。

    通过返回的稀疏表实例可以访问该实例的两个方法说明如下：

    |方法|功能描述|函数原型|参数说明|返回值说明|
    |--|--|--|--|--|
    |size方法|获取稀疏表的大小。|``` def size() ```|无|<li>成功：返回稀疏表的大小。</li><li>失败：抛出异常。</li>|
    |capacity方法|获取稀疏表的容量。|``` def capacity() ```|无|<li>成功：返回稀疏表的容量。</li><li>失败：抛出异常。</li>|


-   失败：抛出异常

**使用示例<a name="section2553042232"></a>**

```bash
import tensorflow as tf
from mx_rec.core.embedding import create_table
sparse_hashtable = create_table(key_dtype=tf.int32,
                                dim=tf.Tensorshape([128]),
                                name="sparse_embeddings_table",
                                emb_initializer=tf.truncated_normal_initializer(),
                                device_vocabulary_size=24_000_000 * 8,
                                host_vocabulary_size=0)
table_size = sparse_hashtable.size()      # 获取返回稀疏表的使用大小
table_capacity = sparse_hashtable.capacity()   # 获取返回稀疏表的容量大小
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## sparse\_lookup<a name="ZH-CN_TOPIC_0000001630127061"></a>

**功能描述<a name="section634582619155"></a>**

Rec SDK TensorFlow模型训练框架，稀疏特征表查询接口。

当前仅支持一表一查和一表多查。若存在一表多查的情况下，查询次数最大值为128。

暂不支持tf.SparseTensor数据类型，若是tf.SparseTensor需转成tf.Tensor。示例代码如下：

```bash
# 示例代码
sparse_ids = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
dense_ids = tf.sparse.to_dense(sparse_ids, default_value=0)
embedding = sparse_lookup(sparse_hashtable, dense_ids)
```

**函数原型<a name="section1483104721911"></a>**

```bash
def sparse_lookup(hashtable, ids, send_count, is_train=True, name=None, modify_graph=False, batch=None, access_and_evict_config=None, is_grad=True, serving_default_value, **kwargs)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|hashtable|BaseSparseEmbedding|必选|待查询的稀疏表。|
|ids|FeatureSpec/tf.Tensor|必选|查询的关键字（key），对应参数类型在不同功能模式下存在区别，具体参见如下。<li>非自动改图模式下，ids参数类型为FeatureSpec。</li><li>自动改图模式下，ids参数类型为tf.Tensor。</li>|
|send_count|int|开启静态shape时为必选参数|作为All2All通信技术，取值范围：[1, 2147483647]。<br>开启动态shape时无需传该参数，或传None即可。默认值为None。|
|is_train|bool|必选|是否为训练模式。默认值为True。<br>取值范围：<li>True：训练模式。</li><li>False：评估或预测模式。</li>|
|name|str|可选|为该次查询操作创建对应的名称，字符串长度为[1,255]。默认值为None。|
|modify_graph|bool|可选|自动改图功能开关，该功能将在创建Session实例前对模型原图进行修改优化，默认值为False。<br>取值范围：<li>True：开启自动改图功能。</li><li>False：关闭自动改图功能。</li>|
|batch|dict{str:tf.Tensor}|可选|数据集的迭代器。<br>当ids使用FeatureSpec类型，且为动态shape时，batch参数必须传入，key为特征名，value为通过对应tf.Dataset执行get_next()之后的结果，默认为None。|
|access_and_evict_config|dict{str:int}|可选|自动改图模式下开启特征准入与淘汰时使用。该dict由三个key-value对组成，key分别为access_threshold、eviction_threshold和faae_coefficient。<br>access_threshold和eviction_threshold的默认值为None，faae_coefficient的默认值为1。<li>access_threshold和eviction_threshold的value为对应的阈值。</li><li>faae_coefficient的value为特征准入系数。</li>|
|is_grad|bool|可选|此次查询是否需要梯度更新，默认值为True。<br>取值范围：<li>True：需要梯度更新。</li><li>False：不需要梯度更新。</li>|
|serving_default_value|tf.Tensor|可选|训练时未准入特征/预测时的新特征的默认emb值。如果不指定，默认为None。|


**\*\*kwargs参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|feature_spec_name_ids_dict|dict|可选|字典结构，key为FeatureSpec名称，value为公开接口sparse_lookup()的参数ids，无默认值。|
|multi_lookup|bool|可选|是否存在一表多查的情况，无默认值。<br>取值范围：<li>True：存在一表多查的情况。</li><li>False：不存在一表多查的情况。</li>|
|lookup_ids|FeatureSpec/tf.Tensor|可选|查询的关键字（key），对应参数类型在不同功能模式下存在区别，无默认值。具体参见如下：<li>非自动改图模式下，ids参数类型为FeatureSpec。</li><li>自动改图模式下，ids参数类型为tf.Tensor。</li>|


>[!NOTE] 说明 
>-   \*\*kwargs参数中的“feature\_spec\_name\_ids\_dict”、“multi\_lookup”和“lookup\_ids”作为内部使用参数，不建议用户通过kwargs传递这三个参数。
>-   如果通过kwargs传递其他未说明参数，则Rec SDK TensorFlow内部不会使用到该参数。

**返回值说明<a name="section651195312311"></a>**

-   成功：返回查询到的Tensor类结果。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
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

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## get\_dense\_and\_sparse\_variable<a name="ZH-CN_TOPIC_0000001630046441"></a>

**功能描述<a name="section634582619155"></a>**

获取模型中密集层参数变量和稀疏层参数变量。

**函数原型<a name="section1483104721911"></a>**

```bash
def get_dense_and_sparse_variable()
```

**返回值说明<a name="section651195312311"></a>**

-   成功：返回密集层参数变量和稀疏层参数变量。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
from mx_rec.util.variable import get_dense_and_sparse_variable
dense_variables, sparse_variables = get_dense_and_sparse_variable()
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


## export<a name="ZH-CN_TOPIC_0000001629887077"></a>

**功能描述<a name="section634582619155"></a>**

读取当前step已保存的稀疏表checkpoint，以key-value形式保存为numpy文件，其中key为查询ID，value为嵌入层表示。以下情况可支持使用export接口导出稀疏表数据：

-   片上内存模式（非动态扩容）
-   DDR模式

**函数原型<a name="section1483104721911"></a>**

```bash
def export(table_list=None):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|table_list|list[str]|可选|需要导出的稀疏表列表，不传参默认导出所有稀疏表。列表长度范围：[1, 2^31-1]|


**返回值说明<a name="section651195312311"></a>**

-   成功：None。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
from mx_rec.saver.sparse import export
table_list=["sparse_embedding_table"]
export(table_list=table_list)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## if\_load<a name="ZH-CN_TOPIC_0000001675798377"></a>

**功能描述<a name="section123217321652"></a>**

设置是否载入已训练过的模型。

**函数原型<a name="section8465133218513"></a>**

```bash
@property
 def if_load(self)
```

**返回值说明<a name="section46439326512"></a>**

-   True：加载已训练的模型。
-   False：不加载已训练的模型。

**使用示例<a name="section7851532751"></a>**

```bash
from mx_rec.util.initialize import ConfigInitializer
if_load = ConfigInitializer.get_instance().if_load
```


