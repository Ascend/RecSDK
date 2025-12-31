# 训练框架初始化与去初始化<a name="ZH-CN_TOPIC_0000001630127073"></a>

## init<a name="ZH-CN_TOPIC_0000001630046449"></a>

**功能描述<a name="section634582619155"></a>**

初始化Rec SDK TensorFlow模型训练框架。

**函数原型<a name="section1483104721911"></a>**

```bash
def init(**kwargs)
```

**\*\*kwargs参数说明<a name="section1643017411155"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|max_steps|int|可选|进行训练的总步数，默认值-1，表示将训练数据全部耗尽后结束。取值范围：[-1, 2147483647]|
|train_steps|int|可选|进行测试预测的训练步数，默认值为-1，代表将训练数据集全部训练完后进行预测。取值范围：[-1, 2147483647]。|
|eval_steps|int|可选|测试预测步数，默认值为-1，代表将测试数据集全部预测完后继续训练。取值范围：[-1, 2147483647]。|
|if_load|bool|可选|选择是否进行模型加载，默认值为False。<br>取值范围：<li>True：进行模型加载。</li><li>False：不进行模型加载。</li>|
|use_dynamic|bool|可选|是否使用动态shape功能，默认值True。<br>取值范围：<li>True：使用动态shape功能。</li><li>False：不使用动态shape功能。</li>|
|use_dynamic_expansion|bool|可选|是否使用片上内存侧动态扩容功能，默认值False。<br>取值范围：<li>True：使用动态扩容功能。</li><li>False：不使用动态扩容功能。</li>|
|bind_cpu|bool|可选|是否使用自动CPU绑核功能，默认值True。<br>取值范围：<li>True：使用自动CPU绑核功能。</li><li>False：不使用自动CPU绑核功能。</li>|
|save_steps|int|可选|训练save_steps后进行保存，默认值-1，表示将训练数据全部训练完后进行保存，取值范围：[-1, 2147483647]。|
|save_checkpoint_due_time|int|可选|保存全量模型的时间间隔（单位：秒）。取值范围：[1, 2147483647]，通常save_checkpoint_due_time参数值大于save_delta_checkpoints_secs。<br>当is_incremental_checkpoint设置为True时，该选项为必选。|
|save_delta_checkpoints_secs|int|可选|保存增量模型的时间间隔（单位：秒）。取值范围：[1, 2147483647]，通常save_checkpoint_due_time参数值大于save_delta_checkpoints_secs。<br>当is_incremental_checkpoint设置为True时，该选项为必选。|
|is_incremental_checkpoint|bool|可选|是否开启模型增量保存与加载，默认为False。<li>True：开启模型增量保存与加载。</li><li>False：关闭模型增量保存与加载。</li>|
|restore_model_version|int|可选|需要加载的模型的步数step，不传该参数时，默认加载最新的模型；当取值为某个具体的step时，加载对应step的模型。<br>取值范围[0, 2147483647]。|
|recent_key_count_threshold|int|可选|在增量保存的这段时间内key出现的最小次数，用于低频过滤，在保存增量模型的时候过滤掉出现频次小于这个参数的key。默认是0。<br>取值范围：[0, 2147483647]|
|use_lccl|bool|可选|运行多卡任务，且通信带宽利用率低时，可使用LCCL（Low Latency Collective Communication Library）功能对集合通信进行加速，开启此功能后，将在部分场景下启用LCCL的以下算子。仅支持单机片上内存的非扩容模式。该功能的具体使用方法请参考[LCCL通信优化算子及样例说明](https://gitcode.com/Ascend/RecSDK/blob/develop/cust_op/ascendc_op/ai_core_op/lccl/v220/README.md)。<li>All2All算子</li><li>GatherAll算子（Gather&AllToAll融合算子）</li><li>GatherUss算子（Gather&UnsortedSegmentSum融合算子）</li>默认值为“False”，表示关闭此功能。|


>[!NOTE] 说明 
>-   使用sess run训练时：使用sess进行train/eval/save的步数，需要和train\_steps/eval\_steps/save\_steps参数一致；
>-   使用Estimator训练时：
>>    -   save\_steps需要和定义NPURunConfig对象时的save\_checkpoints\_steps参数相同，且TF不支持设置为-1。
>>    -   max\_steps需要和传递给est.train\(\)/tf.estimator.TrainSpec\(\)的max\_steps参数相同，且TF不支持设置为-1。
>>    -   train\_and\_evaluate模式时，save\_steps、max\_steps要求同上；train\_steps需要和save\_steps参数相同；eval\_steps需要和传递给tf.estimator.EvalSpec\(\)的steps参数相同，且TF不支持设置为-1。
>-   如果通过kwargs传递其他未说明参数，则Rec SDK TensorFlow内部不会使用到该参数。
>-   “max\_steps”、“train\_steps”和“eval\_steps”不能同时为“0”，且传入的参数需要与实际保持一致。
>-   当“use\_dynamic\_expansion”动态扩容参数为True时，请选用ByAddr类的优化器，如[SGDByAddr](optimizers_apis.md#sgdbyaddr)、[LazyAdamByAddress](optimizers_apis.md#lazyadambyaddress)等。
>-   在train\_and\_evaluate场景下不支持多轮eval。
>-   “max\_steps”、“train\_steps”、“eval\_steps”和“save\_steps”必须与实际训练情况保持一致。若不一致，可能会导致训练无法正常进行、训练出现精度问题等情况。

**返回值说明<a name="section651195312311"></a>**

-   成功：None。
-   失败：抛出异常。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.util.initialize import init
init(max_steps=200, train_steps=100, eval_steps=10, save_steps=100, use_dynamic=True, use_dynamic_expansion=False)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。


## terminate\_config\_initializer<a name="ZH-CN_TOPIC_0000001630246489"></a>

**功能描述<a name="section634582619155"></a>**

去初始化并释放资源。

**函数原型<a name="section1483104721911"></a>**

```bash
def terminate_config_initializer()
```

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.util.initialize import terminate_config_initializer
terminate_config_initializer()
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，请参见[迁移与训练](../migration_and_training.md)。


