# 准入淘汰管理接口<a name="ZH-CN_TOPIC_0000002428477736"></a>

## AdmitAndEvictConfig<a name="ZH-CN_TOPIC_0000002396563024"></a>

**功能描述<a name="section634582619155"></a>**

该接口表示单个嵌入表的准入和淘汰配置。该类提供了配置嵌入表特征准入和淘汰功能的参数，允许用户根据特定条件控制特征的准入和淘汰行为。

**函数原型<a name="section1483104721911"></a>**

```bash
class AdmitAndEvictConfig: 
     admit_threshold: Optional[int] = _DEFAULT_ADMIT_THRESHOLD 
     not_admitted_default_value: Optional[float] = 0.0 
     evict_threshold: Optional[int] = _DEFAULT_EVICT_THRESHOLD  # unit: seconds 
     evict_step_interval: Optional[int] = 0
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|admit_threshold|int|可选|特征准入阈值。特征（在输入分布后）将在重复次数大于admit_threshold时被准入。默认值为-1，表示特征准入功能未启用。|
|not_admitted_default_value|float|可选|未准入特征ID的嵌入值。默认值为0.0，仅在admit_threshold为非默认值时生效。|
|evict_threshold|int|可选|特征淘汰阈值，单位为秒。默认值为0，表示特征淘汰功能未启用。|
|evict_step_interval|int|可选|特征淘汰功能的步长间隔。默认值为0，仅在evict_threshold为非默认值时生效。|



## JaggedTensorWithTimestamp<a name="ZH-CN_TOPIC_0000002461958569"></a>

**功能描述<a name="section634582619155"></a>**

该接口是一个扩展自JaggedTensor的类，用于表示带有时间戳信息的Jagged Tensor。该类在JaggedTensor的基础上增加了一个\_timestamps属性，存储与values对应的时间戳信息。用于特征淘汰时计算时间。

**函数原型<a name="section1483104721911"></a>**

```bash
 class JaggedTensorWithTimestamp(ExtendedJaggedTensor):
    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> None: 
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|values|torch.Tensor|必选|表示Jagged Tensor的值。|
|weights|torch.Tensor|可选|表示每个值的权重。默认为None。|
|lengths|torch.Tensor|可选|表示每个样本的长度。默认为None。|
|offsets|torch.Tensor|可选|表示每个样本的起始偏移量。默认为None。|
|timestamps|torch.Tensor|可选|表示与values对应的时间戳信息。默认为None。|



## KeyedJaggedTensorWithTimestamp<a name="ZH-CN_TOPIC_0000002428320084"></a>

**功能描述<a name="section634582619155"></a>**

该接口是一个扩展自KeyedJaggedTensor的类，用于表示带有时间戳信息的Keyed Jagged Tensor。该类在KeyedJaggedTensor的基础上增加了一个\_timestamps属性，存储与values对应的时间戳信息。用于特征淘汰时计算时间。

**函数原型<a name="section1483104721911"></a>**

```bash
def from_jt_dict(jt_dict: Dict[str, JaggedTensorWithTimestamp],) -> "KeyedJaggedTensorWithTimestamp"
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|jt_dict|Dict[str, JaggedTensorWithTimestamp]|必选|特征名称和对应的JaggedTensorWithTimestamp组成的字典。长度不能为0。其中JaggedTensorWithTimestamp的取值范围参考[JaggedTensor（TorchRec）](./data_apis.md)。|



