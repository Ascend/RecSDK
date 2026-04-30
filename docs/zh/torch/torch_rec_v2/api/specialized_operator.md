# 自定义算子<a name="ZH-CN_TOPIC_0000002452078416"></a>

在推荐训练中，存在部分算子无NPU实现，或已有NPU实现但性能较差，不能满足推荐训练需求。动态稀疏表Rec SDK Torch基于HKV提供了以下[表1 自定义算子列表](#table23231336143616)，用于支持/加速推荐模型NPU训练。

在安装动态稀疏表Rec SDK Torch软件包后，可直接通过dynamic_emb_extensions算子库调用到自定义算子。

>[!NOTE] 须知
>自定义算子为高性能计算，用户调用自定义算子时需自行确保输入的参数满足算子约束条件、参数类型、参数shape等要求，否则可能会出现数组越界，显存不够等问题导致算子执行失败。

**表 1**  自定义算子列表

<a id="table23231336143616"></a>

|算子名称|功能介绍|
|--------|--------|
|block_bucketize_sparse_features|稀疏特征输入（keys）分块桶化。|
|dynamic_emb_Adam_with_pointer|使用Adam优化器，通过指针来定位并更新特定的嵌入向量。|
|dynamic_emb_adamW_with_pointer|使用AdamW优化器，通过指针来定位并更新特定的嵌入向量。|
|dynamic_emb_adagrad_with_pointer|使用Adagrad优化器，通过指针来定位并更新特定的嵌入向量。|
|dynamic_emb_adagrad_with_table|使用Adagrad优化器，通过table指针管理来定位并更新特定的嵌入向量。|
|dynamic_emb_adagrad_fused|使用Adagrad优化器，通过张量管理来定位并更新特定的嵌入向量。|
|dynamic_emb_rowwise_adagrad_with_pointer|使用RowWiseAdagrad优化器，通过指针来定位并更新特定的嵌入向量。|
|dynamic_emb_rowwise_adagrad_with_table|使用RowWiseAdagrad优化器，通过table指针管理来定位并更新特定的嵌入向量。|
|dynamic_emb_rowwise_adagrad_fused|使用RowWiseAdagrad优化器，通过张量管理来定位并更新特定的嵌入向量。|
|find_pointers|在动态嵌入表中查找key对应的value指针并标记查找成功与否。|
|unique_op|对一维indices张量去重操作。|
|segmented_unique_op|多表indices分段后的数据进行去重。|
|get_table_range_op|基于offsets和featureOffsets计算表的范围|
|dedup_input_indices_op|输入索引去重，整合分段去重、范围计算、长度更新等逻辑。|
|gather_embedding|嵌入向量的聚合操作，按indices从inputs中聚合对应维度的嵌入值。|
|reduce_grads|根据indices张量进行梯度规约操作。|
|load_from_pointer|从指针地址加载数据到目标张量。|
|device_timestamp|获取NPU设备的时间戳。|

**表 2**  DynamicEmb配置类

|类名称|功能介绍|类方法/参数|
|--------|--------|--------|
|DynamicEmbTable|动态嵌入表管理类|<ul><li>get_key_type：获取表的键数据类型</li><li>get_value_type：获取表的值数据类型</li><li>get_evict_strategy：获取表的淘汰策略</li><li>get_max_capacity：获取表的最大容量</li><li>get_initializer_args：获取表的初始化参数</li><li>optstate_dim：获取优化器状态的总维度</li><li>get_emb_cols：获取嵌入表的列数</li><li>load：将键值对加载到嵌入表中</li><li>update：更新表中的键值对</li><li>export_batch：从表中导出批量键值对</li></ul>
|InitializerArgs|动态嵌入表的初始化参数类|<ul><li>mode：初始化模式</li><li>mean：正态分布的均值</li><li>std_dev：正态分布的标准差</li><li>lower：均匀分布的下界</li><li>upper：均匀分布的上界</li><li>value：常量初始化的固定值</li></ul>|
|DynamicEmbDataType|标识动态嵌入数据类型的枚举类|<ul><li>Float32：32位单精度浮点数</li><li>BFloat16：16位脑浮点数</li><li>Float16：16位半精度浮点数</li><li>Int64：64位有符号整数</li><li>UInt64：64位无符号整数</li><li>Int32：32位有符号整数</li><li>UInt32：32位无符号整数</li><li>Size_t：无符号整数</li></ul>|
|EvictStrategy|动态嵌入表的淘汰策略的枚举类|<ul><li>kLru：最近最少使用（Least Recently Used）</li><li>kLfu：最不经常使用（Least Frequently Used）</li><li>kEpochLru：周期化LRU（Epoch-based LRU）</li><li>kEpochLfu：周期化LFU（Epoch-based LFU）</li><li>kCustomized：自定义淘汰策略</li></ul>|
|OptimizerType|动态嵌入表优化器的枚举类|<ul><li>Null：无优化器</li><li>SGD：随机梯度下降（Stochastic Gradient Descent）</li><li>Adam：自适应矩估计</li><li>AdamW：带权重衰减的Adam</li><li>AdaGrad：自适应梯度算法</li></ul>|
