# 数据接口<a name="ZH-CN_TOPIC_0000001580326408"></a>

## get\_asc\_insert\_func<a name="ZH-CN_TOPIC_0000001580326412"></a>

**功能描述<a name="section634582619155"></a>**

获取数据预处理函数。

**函数原型<a name="section1483104721911"></a>**

```bash
def get_asc_insert_func(tgt_key_specs=None, args_index_list=None, table_names=None, **kwargs)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|说明|
|--|--|--|
|tgt_key_specs|<li>FeatureSpec</li><li>list[FeatureSpec]</li>|特征对象或特征对象列表或者特征对象元组，默认值为None。|
|args_index_list|list[int]|参数索引列表，默认值为None。取值范围：[1, 2^31-1]|
|table_names|list[str]|表名称列表，默认值为None。取值范围：[1, 2^31-1]|


>[!NOTE] 说明 
>接口参数可选择以下其中一种方式传入。
>-   仅传入“tgt\_key\_specs”。
>-   传入“args\_index\_list”和“table\_names”。

**\*\*kwargs参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|is_training|bool|可选|是否为训练模式，默认值为True。<br>取值范围：<li>True：训练模式。</li><li>False：评估或预测模式。</li>|
|dump_graph|bool|可选|是否保存模型图，默认值为False。<br>取值范围：<li>True：保存模型图。</li><li>False：不保存模型图。</li>|


>[!NOTE] 说明 
>-   \*\*kwargs参数中的“is\_training”和“dump\_graph”作为内部使用参数，不建议用户通过kwargs传递这两个参数。
>-   如果通过kwargs传递其他未说明参数，则Rec SDK TensorFlow内部不会使用到该参数。

**返回值说明<a name="section651195312311"></a>**

-   成功：数据预处理函数。
-   失败：抛出异常。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
import tensorflow as tf
from mx_rec.core.asc.helper import get_asc_insert_func

dataset = tf.data.TFRecordDataset(data_path) # data_path为数据集路径
dataset = dataset.map(get_asc_insert_func(tgt_key_specs=feature_spec_list, is_training=True)) # feature_spec_list中元素为FeatureSpec对象
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


## modify\_graph\_and\_start\_emb\_cache<a name="ZH-CN_TOPIC_0000001630246525"></a>

**功能描述<a name="section634582619155"></a>**

开启自动改图模式下的数据加载和预处理接口。

**函数原型<a name="section1483104721911"></a>**

```bash
def modify_graph_and_start_emb_cache(full_graph = None, dump_graph = False)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|full_graph|tf.Graph|可选|改图支持传入图实例，默认为None，None会被赋值为tf.compat.v1.get_default_graph()。|
|dump_graph|bool|可选|是否保存模型图，默认值为False。<br>取值范围：<li>True：保存模型图。</li><li>False：不保存模型图。</li>|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回None。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
from mx_rec.graph.modifier import modify_graph_and_start_emb_cache
MODIFY_GRAPH_FLAG = True
if MODIFY_GRAPH_FLAG:
    modify_graph_and_start_emb_cache(dump_graph=True)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


## start\_asc\_pipeline<a name="ZH-CN_TOPIC_0000001579847256"></a>

**功能描述<a name="section634582619155"></a>**

非自动改图模式下，初始化并启动数据预处理流水。

**函数原型<a name="section1483104721911"></a>**

```bash
def start_asc_pipeline()
```

**返回值说明<a name="section651195312311"></a>**

-   成功：返回None。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
from mx_rec.core.asc.manager import start_asc_pipeline
MODIFY_GRAPH_FLAG = False
if not MODIFY_GRAPH_FLAG:
    start_asc_pipeline()
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


