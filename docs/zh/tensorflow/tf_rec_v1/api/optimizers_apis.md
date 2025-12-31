# 优化器<a name="ZH-CN_TOPIC_0000001629887049"></a>

## Adagrad<a name="ZH-CN_TOPIC_0000001579847272"></a>

自定义Adagrad优化器。

**函数原型<a name="section1483104721911"></a>**

```bash
def create_hash_optimizer(learning_rate=0.001, initial_accumulator_value=0.9, use_locking=False, name="Adagrad")
```

**参数说明<a name="zh-cn_topic_0000001422098394_section698416142519"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|可选|学习率。默认值：0.001。取值范围：[0.0, 10.0]。|
|initial_accumulator_value|float|可选|累加器的初始值。取值范围：(0.0, 1.0]。默认值：0.9。|
|use_locking|bool|可选|优化器中防止对变量并发更新。默认值：False。取值范围：True、False。|
|name|string|可选|优化器名称。默认值：Adagrad。名称长度范围：[1, 200]。|


**返回值说明<a name="section420817634911"></a>**

CustomizedAdagrad（自定义Adagrad优化器）的一个实例对象。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.optimizers.adagrad import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer(0.001)
```


## Ftrl<a name="ZH-CN_TOPIC_0000001629887073"></a>

自定义Ftrl优化器。

**函数原型<a name="section1483104721911"></a>**

```bash
def create_hash_optimizer(learning_rate, use_locking=False, name="Ftrl", **kwargs)
```

**参数说明<a name="zh-cn_topic_0000001511059429_section869711351111"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|必选|学习率。取值范围：[0.0, 10.0]。|
|use_locking|bool|可选|优化器中防止对变量并发更新。默认值：False。取值范围：True、False。|
|name|string|可选|优化器名称。默认值：Ftrl。名称长度范围：[1, 200]。|


**\*\*kwargs参数说明<a name="section1643017411155"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate_power|float|可选|控制训练期间学习率的下降。默认值：-0.5。取值范围：[-2147483647.0, 0.0]。|
|initial_accumulator_value|float|可选|累积器的初始值。默认值：0.1。取值范围：(0.0, 1.0]。|
|l1_regularization_strength|float|可选|L1正则化惩罚。默认值：0.0。取值范围：[0.0, 10000.0]。|
|l2_regularization_strength|float|可选|L2正则化惩罚。默认值：0.0。取值范围：[0.0, 10000.0]。|
|accum_name|string|可选|保存梯度平方累加器的变量的后缀。默认值：None。长度范围：[1, 255]。|
|linear_name|string|可选|保存线性梯度累加器的变量的后缀。默认值：None。长度范围：[1, 255]。|
|l2_shrinkage_regularization_strength|float|可选|L2正则化幅度惩罚。默认值：0.0。取值范围：[0.0, 10000.0]。|


>[!NOTE] 说明 
>如果通过kwargs传递其他未说明参数，则Rec SDK TensorFlow内部不会使用到该参数。

**返回值说明<a name="section55541314104916"></a>**

CustomizedFtrl（自定义Ftrl优化器）的一个实例对象。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.optimizers.ftrl import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer(0.001)
```


## SGD<a name="ZH-CN_TOPIC_0000001580326436"></a>

自定义SGD优化器。

**函数原型<a name="section1483104721911"></a>**

```bash

def create_hash_optimizer(learning_rate, use_locking=False, name="GradientDescent", use_fusion_optim=False, weight_decay=None)
```

**参数说明<a name="zh-cn_topic_0000001422257698_section1411215366717"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|必选|学习率。取值范围：[0.0, 10.0]。|
|use_locking|bool|可选|优化器中防止对变量并发更新。默认值：False。取值范围：True、False。|
|name|string|可选|优化器名称。默认值：GradientDescent。名称长度范围：[1, 200]。|
|use_fusion_optim|bool|可选|是否使能算子加速。默认值：False。<br>取值范围：<li>True：使能融合算子加速。</li><li>False：不使能。</li>|
|weight_decay|float|可选|权重衰减系数。默认值：None，即不使能权重衰减。取值范围：[1e-5, 1e-2]。|


**返回值说明<a name="section58446553505"></a>**

CustomizedGradientDescent（自定义SGD优化器）的一个实例对象。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.optimizers.gradient_descent import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer(0.001)
```


## SGDByAddr<a name="ZH-CN_TOPIC_0000001579847296"></a>

自定义SGDByAddr地址优化器。

**函数原型<a name="section1483104721911"></a>**

```bash
def create_hash_optimizer_by_addr(learning_rate, weight_decay=0.0001, use_locking=False, name="GradientDescentByAddr")
```

**参数说明<a name="zh-cn_topic_0000001422257698_section1411215366717"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|必选|学习率。取值范围：[0.0, 10.0]。|
|weight_decay|float|可选|权重衰减。默认值：0.0001。取值范围：[0.0, 1.0]。|
|use_locking|bool|可选|优化器中防止对变量并发更新。默认值：False。取值范围：True、False。|
|name|string|可选|优化器名称。默认值：GradientDescentByAddr。名称长度范围：[1, 200]。|


**返回值说明<a name="section1652515816519"></a>**

CustomizedGradientDescentByAddr（自定义SGD地址优化器）的一个实例对象。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.optimizers.gradient_descent_by_addr import create_hash_optimizer_by_addr
hashtable_optimizer = create_hash_optimizer_by_addr(0.001)
```


## LazyAdam<a name="ZH-CN_TOPIC_0000001580166520"></a>

自定义LazyAdam优化器。

**函数原型<a name="section1483104721911"></a>**

```bash
def create_hash_optimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdam", use_fusion_optim=False)
```

**参数说明<a name="zh-cn_topic_0000001422098394_section698416142519"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|可选|学习率。默认值：0.001。取值范围：[0.0, 10.0]。|
|beta1|float|可选|第一矩的指数衰减率估计。默认值：0.9。取值范围：(0.0, 1.0)。|
|beta2|float|可选|第二矩的指数衰减率估计。默认值：0.999。取值范围：[0.0, 1.0]。|
|epsilon|float|可选|加入此值到分母中，提高数据稳定性。默认值：1e-8。取值范围：(0.0, 1.0]。|
|name|string|可选|优化器名称。默认值：LazyAdam。名称长度范围：[1, 200]。|
|use_fusion_optim|bool|可选|是否使用LazyAdam融合算子进行slot_m, slot_v, variable数据计算和更新。默认值：False。<br>取值范围：<li>True：表示使用融合算子，且需要手动编译和部署LazyAdam融合算子，使用方法和约束请参考[README](https://gitcode.com/Ascend/RecSDK/blob/branch_v7.2.0-RC1/cust_op/ascendc_op/ai_core_op/fused_lazy_adam/v220/README.md)。</li><li>False：表示不使用LazyAdam融合算子。</li>|


**返回值说明<a name="section31441728175119"></a>**

CustomizedLazyAdam（自定义LazyAdam优化器）的一个实例对象。

**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from mx_rec.optimizers.lazy_adam import create_hash_optimizer
hashtable_optimizer = create_hash_optimizer()
```


## LazyAdamByAddress<a name="ZH-CN_TOPIC_0000001580166496"></a>

自定义LazyAdamByAddress地址优化器。

**函数原型<a name="section1483104721911"></a>**

```bash
def create_hash_optimizer_by_address(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="LazyAdamByAddress")
```

**参数说明<a name="zh-cn_topic_0000001422098394_section698416142519"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|可选|学习率。默认值：0.001。取值范围：[0.0, 10.0]。|
|beta1|float|可选|第一矩的指数衰减率估计。默认值：0.9。取值范围：(0.0, 1.0)。|
|beta2|float|可选|第二矩的指数衰减率估计。默认值：0.999。取值范围：[0.0, 1.0]。|
|epsilon|float|可选|加入此值到分母中，提高数据稳定性。默认值：1e-8。取值范围：(0.0, 1.0]。|
|name|string|可选|优化器名称。默认值：LazyAdamByAddress。名称长度范围：[1, 200]。|


**返回值说明<a name="section666334110514"></a>**

CustomizedLazyAdamByAddress（自定义LazyAdam地址优化器）的一个实例对象。

**使用示例<a name="section172089497567"></a>**

```bash
from mx_rec.optimizers.lazy_adam_by_addr import create_hash_optimizer_by_address
hashtable_optimizer = create_hash_optimizer_by_address()
```


## AdagradByAddress<a name="ZH-CN_TOPIC_0000001898438414"></a>

自定义AdagradByAddress地址优化器。

**函数原型<a name="section19937194912307"></a>**

```bash
def create_hash_optimizer_by_address(learning_rate=0.001, initial_accumulator_value=0.9, name="Adagrad")
```

**参数说明<a name="section82884196321"></a>**

|**参数名**|**类型**|**必选/可选**|**说明**|
|--|--|--|--|
|learning_rate|float/tf.Tensor|可选|学习率。默认值：0.001。取值范围：[0.0, 10.0]。|
|initial_accumulator_value|float|可选|累加器的初始值。默认值：0.9。取值范围：(0.0, 1.0]。|
|name|string|可选|优化器名称。默认值：Adagrad。名称长度范围：[1, 200]。|


**返回值说明<a name="section64121143313"></a>**

CustomizedAdagradByAddress（自定义Adagrad地址优化器）的一个实例对象。

**使用示例<a name="section1311915426339"></a>**

```bash
from mx_rec.optimizers.adagrad_by_addr import create_hash_optimizer_by_address
hashtable_optimizer = create_hash_optimizer_by_address()
```


