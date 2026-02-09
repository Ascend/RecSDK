# 准入淘汰管理接口<a name="ZH-CN_TOPIC_0000002428477736"></a>

## AdmitAndEvictPolicyType<a name="ZH-CN_TOPIC_0000002430202770"></a>

**功能描述<a name="section634582619155"></a>**

准入淘汰策略类型枚举，定义嵌入表特征准入和淘汰的策略类型。

**函数原型<a name="section1483104721911"></a>**

```bash
class AdmitAndEvictPolicyType(Enum):
    NONE = 0
    POLICY_COUNT = 1
    POLICY_SHOWCLICK = 2
```

**参数说明<a name="section888634319218"></a>**

|参数名|说明|
|--|--|
|NONE|无准入淘汰策略。|
|POLICY_COUNT|基于计数（count）/时间（timestamp）的准入淘汰策略。特征准入基于重复次数，特征淘汰基于时间阈值。|
|POLICY_SHOWCLICK|基于展示点击（show/click）的准入淘汰策略。特征准入和淘汰基于展示次数和点击次数的加权分数。|

**返回值说明<a name="section651195312311"></a>**

-   成功：返回AdmitAndEvictPolicyType枚举值。
-   失败：抛出异常。


## ShowClickParams<a name="ZH-CN_TOPIC_0000002430202771"></a>

**功能描述<a name="section634582619155"></a>**

该接口表示基于展示点击（show/click）策略的参数配置。该类提供了配置展示点击准入和淘汰功能的参数，允许用户根据展示次数和点击次数控制特征的准入和淘汰行为。

**函数原型<a name="section1483104721911"></a>**

```bash
@dataclass
class ShowClickParams:
    alpha: float = 1.0
    beta: float = 0.0
    admit_threshold: float = 0.0
    evict_percentage: float = 0.0
    score_decay: float = 1.0
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|alpha|float|可选|展示次数的权重系数，用于计算准入和淘汰分数。默认值为1.0。准入分数计算公式：score = alpha * showCount + beta * clickCount。|
|beta|float|可选|点击次数的权重系数，用于计算准入和淘汰分数。默认值为0.0。准入分数计算公式：score = alpha * showCount + beta * clickCount。|
|admit_threshold|float|可选|特征准入阈值。开启准入时，小于此分数的特征将被丢弃。当此值大于0时表示开启准入功能。默认值为0.0，表示特征准入功能未启用。|
|evict_percentage|float|可选|特征淘汰比例。开启淘汰时，分数较小且在此比例中的特征将被淘汰。当此值大于0时表示开启淘汰功能。默认值为0.0，表示特征淘汰功能未启用。取值范围：[0.0, 1.0]。|
|score_decay|float|可选|分数衰减系数，用于淘汰分数计算和更新。取值范围：[0.0, 1.0]。默认值为1.0，表示不衰减；0.0表示全衰减。淘汰分数计算公式：newScore = (oldScore + alpha * showCount + beta * clickCount) * scoreDecay。|

**返回值说明<a name="section651195312311"></a>**

-   成功：返回ShowClickParams对象。
-   失败：抛出异常。


## AdmitAndEvictConfig<a name="ZH-CN_TOPIC_0000002396563024"></a>

**功能描述<a name="section634582619155"></a>**

该接口表示单个嵌入表的准入和淘汰配置。该类提供了配置嵌入表特征准入和淘汰功能的参数，允许用户根据特定条件控制特征的准入和淘汰行为。支持两种策略类型：基于计数的策略（POLICY_COUNT）和基于展示点击的策略（POLICY_SHOWCLICK）。

**函数原型<a name="section1483104721911"></a>**

```bash
@dataclass
class AdmitAndEvictConfig: 
     admit_threshold: Optional[int] = _DEFAULT_ADMIT_THRESHOLD 
     not_admitted_default_value: Optional[float] = 0.0 
     evict_threshold: Optional[int] = _DEFAULT_EVICT_THRESHOLD  # unit: seconds 
     evict_step_interval: Optional[int] = 0
     showclick_params: ShowClickParams = field(default_factory=lambda: ShowClickParams())
     policy_type: AdmitAndEvictPolicyType = AdmitAndEvictPolicyType.POLICY_COUNT
```

**参数说明<a name="section888634319218"></a>**

policy_type为POLICY_COUNT时：
|参数名|类型|可选/必选|说明|
|--|--|--|--|
|admit_threshold|int|可选|特征准入阈值。特征（在输入分布后）将在重复次数大于admit_threshold时被准入。默认值为-1，表示特征准入功能未启用。|
|not_admitted_default_value|float|可选|未准入特征ID的嵌入值。默认值为0.0，仅在admit_threshold为非默认值时生效。|
|evict_threshold|int|可选|特征淘汰阈值，单位为秒。默认值为0，表示特征淘汰功能未启用。|
|evict_step_interval|int|可选|特征淘汰功能的步长间隔。默认值为0，仅在evict_threshold为非默认值时生效。|

policy_type为POLICY_SHOWCLICK时：
|参数名|类型|可选/必选|说明|
|--|--|--|--|
|not_admitted_default_value|float|可选|未准入特征ID的嵌入值。默认值为0.0。|
|evict_step_interval|int|可选|特征淘汰功能的步长间隔。默认值为0。|
|showclick_params|ShowClickParams|可选|基于展示点击策略的参数配置。参数说明参考[ShowClickParams](#showclickparams)。|

**约束<a name="section888634319218"></a>**

POLICY_COUNT和POLICY_SHOWCLICK策略不能一起使用。


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



