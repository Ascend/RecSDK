# 优化器接口<a name="ZH-CN_TOPIC_0000002336148957"></a>

## apply\_optimizer\_in\_backward（TorchRec）<a name="ZH-CN_TOPIC_0000002302229708"></a>

>[!NOTICE]须知
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

指定表使用的优化器。

**函数原型<a name="section1483104721911"></a>**

```cpp
def apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
) -> None:
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|optimizer_class|Type[torch.optim.Optimizer]|必选|优化器类型。取值范围：<li>torch.optim.Adagrad：Adagrad优化器。</li><li>torch.optim.Adam：Adam优化器。</li><li>torch.optim.SGD：SGD优化器。</li><li>torchrec.optim.AccumulateAdagrad：带梯度累积功能的Adagrad优化器。</li><li>torchrec.optim.AccumulateAdam：带梯度累积功能的Adam优化器。</li><li>torchrec.optim.AccumulateSGD：带梯度累积功能的SGD优化器。</li><div class="note"><span class="notetitle">说明</span><div class="notebody"><li>优化器的类型需要与创表接口时的配置保持一致。梯度累积功能当前只支持[EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection)的创表接口。</li><li>使用torchrec.optim.AccumulateSGD时，建议每个embedding的聚合数量不超过10000条。超过该限制时，计算精度有可能不满足双万分之一。</li></div></div>|
|params|Iterable[torch.nn.Parameter]|必选|设置优化器的torch.nn.Parameter对象。参考[步骤5](../quick_start.md#接口调用介绍)传入HashEmbeddingBagCollection对象的参数。<div class="notice"><span class="noticetitle">须知</span><div class="noticebody">基于性能考虑，params参数无法校验。用户需自行保证其类型正确性。</div></div>|
|optimizer_kwargs|Dict[str, Any]|必选|根据optimizer_class的参数和范围进行配置，用户需自行保证该参数的范围符合对应优化器限制。<p>如果使用梯度累积功能，optimizer_class需要传入带梯度累积功能的优化器，并传入取值为True的use_accumulate（bool类型）参数（默认为False）和指定梯度累积的步数的accumulate_step（int类型）参数（默认为1）。</p>|


**返回值说明<a name="section651195312311"></a>**

-   成功：None
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```cpp
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
import torch
embedding_optimizer = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.001, "eps": 0.1}
apply_optimizer_in_backward(
	embedding_optimizer,
	ebc.parameters(),
	optimizer_kwargs=optimizer_kwargs)

```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## in\_backward\_optimizer\_filter（TorchRec）<a name="ZH-CN_TOPIC_0000002336268629"></a>

>[!NOTICE]须知
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

过滤掉被指定为backward\_optimizer的参数。

**函数原型<a name="section1483104721911"></a>**

```cpp
def in_backward_optimizer_filter(
    named_parameters: Iterator[Tuple[str, nn.Parameter]], include: bool = False
) -> Iterator[Tuple[str, nn.Parameter]]:
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|named_parameters|Iterator[Tuple[str, nn.Parameter]]|必选|模型的参数列表。用户需自行保证该变量是通过torch.nn.Module的named_parameters()获得。|
|include|bool|可选|如果include为True，返回的结果包含backward_optimizer的参数名，否则不包含。默认值为False。<li>True：包含</li><li>False：不包含</li>|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回过滤后的参数。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```cpp
from torchrec.optim.optimizers import in_backward_optimizer_filter
parameter = in_backward_optimizer_filter(model.named_parameters())
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## KeyedOptimizerWrapper（TorchRec）<a name="ZH-CN_TOPIC_0000002302229608"></a>

>[!NOTICE]须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

用于封装过滤掉表的参数的优化器。

**函数原型<a name="section1483104721911"></a>**

```cpp
class KeyedOptimizerWrapper:
    def __init__(**kwargs)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|params|Mapping[str, Union[torch.Tensor, ShardedTensor]]|必选|参数列表。用户需自行保证通过[in_backward_optimizer_filter（TorchRec）](#in_backward_optimizer_filtertorchrec)获得。|
|optim_factory|Callable|必选|传入创建优化器的函数。用户需自行保证该函数满足输入为一个参数，输出一个torch.optim.Optimizer对象。|


**返回值说明<a name="section651195312311"></a>**

-   成功：参数过滤后的优化器。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```cpp
from torchrec.optim.keyed import KeyedOptimizerWrapper
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(model.named_parameters())),
    lambda params: torch.optim.Adam(params, lr=1e-1),
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


## CombinedOptimizer（TorchRec）<a name="ZH-CN_TOPIC_0000002302229544"></a>

>[!NOTICE]须知 
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

将多个优化器合并为一个。

**函数原型<a name="section1483104721911"></a>**

```cpp
class CombinedOptimizer:
    def __init__(**kwargs)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|optims|<li>List[KeyedOptimizer]</li><li>List[Tuple[str, KeyedOptimizer]]</li>|必选|需要合并的优化器对象列表，可以为空列表。<br>用户需要自行保证其输入值来源于CombinedOptimizer（TorchRec）或者[DistributedModelParallel（TorchRec）](./subtable_apis.md#distributedmodelparalleltorchrec)|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回合并后优化器。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```cpp
from torchrec.optim.keyed import CombinedOptimizer
optimizer = CombinedOptimizer([ddp_model.fused_optimizer])
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。


