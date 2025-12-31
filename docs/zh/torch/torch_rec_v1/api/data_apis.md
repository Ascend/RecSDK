# 数据接口<a name="ZH-CN_TOPIC_0000002302229576"></a>

## JaggedTensor（TorchRec）<a name="ZH-CN_TOPIC_0000002336148837"></a>

>[!NOTICE]须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

持有稀疏id和特征长度的类，用于查表。例如，values为\[id1, id2, id3, id4\]，length为\[1, 2, 1\]。表示为id2和id3查表后的Embedding应该被pooling。

**函数原型<a name="section1483104721911"></a>**

```cpp
class JaggedTensor:
 def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|values|torch.Tensor[int64]|必选|稀疏表查表ID。取值范围：[0，2^31]。|
|weights|torch.Tensor|可选|仅支持默认值为None，不支持用户自定义。|
|lengths|torch.Tensor[int64]|必选|每一个样本中的特征序列的长度。取值范围：[1，10000]。需保证lengths的总和与values的长度相等。Rec SDK Torch目前不支持可变batchsize，一个训练任务中的所有JaggedTensor的lengths的长度必须一致。|
|offsets|torch.Tensor[int64]|可选|offsets是lengths累加的结果。offsets的第一位为0，后续位数为lengths的累加。默认值为None。offsets的合法性由用户自行保证。|


**使用示例<a name="zh-cn_topic_0000001422098394_section653575124718"></a>**

```bash
from torchrec import JaggedTensor
JaggedTensor(values=[1, 3, 4], lengths=[1, 1, 1], offsets=[0, 1, 2, 3])
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。


## KeyedJaggedTensor（TorchRec）<a name="ZH-CN_TOPIC_0000002302229696"></a>

### from\_jt\_dict<a name="ZH-CN_TOPIC_0000002508691797"></a>

>[!NOTICE]须知
>此类下的接口为TorchRec开源接口，非Rec SDK Torch对外接口。本章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

通过JaggedTensor，创建KeyedJaggedTensor。

**函数原型<a name="section1483104721911"></a>**

```bash
def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor"
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|jt_dict|Dict[str, JaggedTensor]|必选|特征名称和对应的JaggedTensor组成的字典。长度不能为0。其中JaggedTensor的取值范围参考[JaggedTensor（TorchRec）](#jaggedtensortorchrec)。|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回KeyedJaggedTensor。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
from torchrec import KeyedJaggedTensor, JaggedTensor
jt = JaggedTensor(values=[1, 3, 4], lengths=[1, 1, 1])
kjt = KeyedJaggedTensor.from_jt_dict({"feat0": jt})
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例，参见[迁移与训练](../migration_and_training.md)。



