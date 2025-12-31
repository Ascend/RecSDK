# 类参考<a name="ZH-CN_TOPIC_0000001580007112"></a>

## FeatureSpec<a name="ZH-CN_TOPIC_0000001580007124"></a>

待查询特征的配置描述类，适用于非自动改图模式。

|参数名|类型|**必选/可选**|说明|
|--|--|--|--|
|index_key|int/string|可选|索引键。默认值：table_name的值。<br>取值范围：<li>int：取值范围为[0,255]。</li><li>string：长度范围为[1,255]</li>|
|table_name|string|可选|表名。<br>表名长度范围：[1, 255]。|
|access_threshold|int|可选|特征准入阈值。<br>取值范围：[-1, 2147483647]。<li>等于0：开启准入功能。不累加新batch中key出现的次数，使用历史特征计数记录。</li><li>大于0：开启准入功能。累加新batch中key出现的次数，更新特征计数记录。</li><li>等于-1：关闭功能。</li>|
|eviction_threshold|int|可选|特征淘汰阈值。<br>取值范围：[-1, 2147483647]。<li>等于或大于0：开启淘汰功能。</li><li>等于-1：关闭淘汰功能。</li>如果需要设置特征淘汰阈值，需要同时设置特征准入阈值。|
|is_timestamp|bool|可选|是否为时间戳。<br>取值范围：True、False。|
|batch_size|int|可选|数据集batch的大小。<br>取值范围：[1, 2147483647]。|
|faae_coefficient|int|可选|特征准入系数。默认值：1。<br>取值范围：[1, 2147483647]。|
|name|string|必选|FeatureSpec名称。长度范围：[1,255]。|


**使用示例<a name="section6456105313583"></a>**

```bash
from mx_rec.core.asc.feature_spec import FeatureSpec
feature_spec_list = FeatureSpec("user_ids", table_name="user_table",
                                access_threshold=1,
                                eviction_threshold=1,
                                faae_coefficient=1)

```


## GraphModifierHook<a name="ZH-CN_TOPIC_0000001630127057"></a>

自动改图Hook类，仅在[使用Estimator训练](../migration_and_training.md#使用estimator训练)模式下使用，添加后即可开启自动改图功能。

|参数名|类型|**必选/可选**|说明|
|--|--|--|--|
|dump_graph|bool|可选|是否保存TensorFlow当前计算图，默认为False。|
|modify_graph|bool|可选|是否开启自动改图功能，默认为True。|


**使用示例<a name="section14589163715471"></a>**

```bash
from mx_rec.graph.modifier import GraphModifierHook

#定义数据处理函数
def input_fn():
     pass

est.train(input_fn=lambda: input_fn(), hooks=[GraphModifierHook()])   #est为创建的NPUEstimator对象
```


## EvictHook<a name="ZH-CN_TOPIC_0000001580007108"></a>

特征淘汰Hook类，仅在特征准入与淘汰模式下使用，配合特征淘汰的阈值“eviction\_threshold”设置，添加后即可开启特征淘汰功能。

>[!NOTE] 说明 
>特征淘汰Hook类仅支持在训练场景下使用。

|参数名|类型|**必选/可选**|说明|
|--|--|--|--|
|evict_enable|bool|可选|是否开启特征淘汰功能，默认为False。|
|evict_time_interval|int|可选|淘汰功能触发时间间隔，单位：秒，默认为24 * 60 * 60。取值范围：[1, MAXINT32]。|
|evict_step_interval|int|可选|淘汰功能触发步数间隔，单位：步，默认为None。取值范围：[1, MAXINT32]。|


**使用示例<a name="section14589163715471"></a>**

```bash
from mx_rec.core.feature_process import EvictHook
hooks_list = []
hook_evict = EvictHook(evict_enable=True, evict_time_interval=30, evict_step_interval=20)
hooks_list.append(hook_evict)

#定义数据处理函数
def input_fn():
     pass

est.train(input_fn=lambda: input_fn(), hooks=hooks_list)    #est为创建的NPUEstimator对象
```


## ConfigInitializer<a name="ZH-CN_TOPIC_0000002095874621"></a>

保存全局配置信息的管理类，为单例模式。

该类通过init\(\)函数自动初始化，不需要手动进行构建。同时，本章节只列举该类中对外公开的接口，剩余未在此处公示的为内部接口，不推荐直接调用。

**调用示例<a name="section73615361858"></a>**

|接口|作用|原型|
|--|--|--|
|get_instance()|获取ConfigInitializer的全局唯一实例。|from mx_rec.util.initialize import ConfigInitializerConfigInitializer.get_instance()|
|use_dynamic_expansion()|请参见[use_dynamic_expansion](other_apis.md#use_dynamic_expansion)。|
|get_target_batch()|请参见[get_target_batch](other_apis.md#get_target_batch)。|
|if_load()|请参见[if_load](model_apis.md#if_load)。|
|get_initializer(is_training)|请参见[get_initializer](automatic_graph_modification.md#get_initializer)。|
|ascend_global_hashtable_collection()|请参见[ascend_global_hashtable_collection](other_apis.md#ascend_global_hashtable_collection)。|



## TrainParamsConfig<a name="ZH-CN_TOPIC_0000002470669008"></a>

保存训练任务参数配置的数据类，例如哈希表集合的名字。

该类通过init\(\)函数自动初始化，不需要手动进行构建。同时，本章节只列举该类中对外公开的接口，剩余未在此处公示的为内部接口，不推荐直接调用。

**调用示例<a name="section73615361858"></a>**

|接口|作用|
|--|--|
|ascend_global_hashtable_collection()|请参见[ascend_global_hashtable_collection](other_apis.md#ascend_global_hashtable_collection)。|



