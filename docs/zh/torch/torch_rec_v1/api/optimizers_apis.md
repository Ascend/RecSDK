# 优化器接口<a name="ZH-CN_TOPIC_0000002336148957"></a>

## apply\_optimizer\_in\_backward（TorchRec）<a id="TOPIC_0000002302229708"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

指定表使用的优化器。

**函数原型<a name="section1483104721911"></a>**

```python
def apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
) -> None:
```

**参数说明<a name="section888634319218"></a>**

| 参数名              | 类型                           | 可选/必选 | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|------------------|------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optimizer_class  | Type[torch.optim.Optimizer]  | 必选    | 优化器类型。取值范围：<ul><li>torch.optim.Adagrad：Adagrad优化器。</li><li>torch.optim.Adam：Adam优化器。</li><li>torch.optim.SGD：SGD优化器。</li><li>torchrec.optim.Adagrad：Adagrad优化器。</li><li>torchrec.optim.Adam：Adam优化器。</li><li>torchrec.optim.SGD：SGD优化器。</li><li>torchrec.optim.AccumulateAdagrad：带梯度累积功能的Adagrad优化器。</li><li>torchrec.optim.AccumulateAdam：带梯度累积功能的Adam优化器。</li><li>torchrec.optim.AccumulateSGD：带梯度累积功能的SGD优化器。</li></ul>说明：<ul><li>优化器类型需和创建稀疏表（仅[EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](./table_creation_apis.md#embcacheembeddingbagcollection)涉及）时的`embedding_optimizer_cls`参数保持一致。</li><li>梯度累积功能当前只支持[EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection)的创表接口，梯度累积优化器使用请参见[用例](https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_aggregation.py)。</li><li>使用torchrec.optim.AccumulateSGD时，建议每个embedding的聚合数量不超过10000条。超过该限制时，计算精度有可能不满足双万分之一。</li></ul> |
| params           | Iterable[torch.nn.Parameter] | 必选    | 设置优化器的torch.nn.Parameter对象。需传入[HashEmbeddingBagCollection](./table_creation_apis.md#hashembeddingbagcollection)/[EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](./table_creation_apis.md#embcacheembeddingbagcollection)对象的参数。<br>须知：<br>基于性能考虑，params参数未进行校验。用户需自行保证其类型正确性。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| optimizer_kwargs | Dict[str, Any]               | 必选    | 根据optimizer_class的参数和范围进行配置，用户需自行保证该参数的范围符合对应优化器限制。<p>如果使用梯度累积功能，optimizer_class需要传入带梯度累积功能的优化器，并传入取值为True的use_accumulate（bool类型）参数（默认为False）和指定梯度累积的步数的accumulate_step（int类型）参数（默认为1）。</p>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

**返回值说明**

- 成功：None
- 失败：抛出异常。

**使用示例**

```python
import torch
import torchrec
from hybrid_torchrec.modules.hash_embeddingbag import HashEmbeddingBagCollection, HashEmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward

def weight_init(param: torch.nn.Parameter):
    if len(param.shape) != 2:
        return
    torch.manual_seed(param.shape[1])
    result = torch.linspace(0, 1, steps=param.shape[1]).repeat(param.shape[0], 1)
    param.data.copy_(result)


embedding_dims: list[int] = [64, 64, 64]
num_embeddings: list[int] = [400, 4000, 400]
table_num: int = len(num_embeddings)
embedding_configs: list[HashEmbeddingBagConfig] = []
table_names: list[str] = [f"table{i}" for i in range(len(num_embeddings))]
feat_names: list[str] = [f"feat{i}" for i in range(table_num)]
for i in range(table_num):
    emb_config = HashEmbeddingBagConfig(
        name=table_names[i],
        embedding_dim=embedding_dims[i],
        num_embeddings=num_embeddings[i],
        feature_names=[feat_names[i]],
        pooling=torchrec.PoolingType.MEAN,
        init_fn=weight_init,  # type: ignore
    )
    embedding_configs.append(emb_config)
# 以HashEmbeddingBagCollection为例，获取其参数并传入apply_optimizer_in_backward接口
sparse_ebc: HashEmbeddingBagCollection = HashEmbeddingBagCollection(device="meta", tables=embedding_configs)
embedding_optimizer = torch.optim.Adagrad
optimizer_kwargs = {"lr": 0.001, "eps": 1e-5}
apply_optimizer_in_backward(
    embedding_optimizer,
    sparse_ebc.parameters(),
    optimizer_kwargs=optimizer_kwargs
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## in\_backward\_optimizer\_filter（TorchRec）<a name="ZH-CN_TOPIC_0000002336268629"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

过滤掉被指定为backward\_optimizer的参数。

**函数原型<a name="section1483104721911"></a>**

```python
def in_backward_optimizer_filter(
    named_parameters: Iterator[Tuple[str, nn.Parameter]], include: bool = False
) -> Iterator[Tuple[str, nn.Parameter]]:
```

**参数说明**

| 参数名              | 类型                                 | 可选/必选 | 说明                                                                                                       |
|------------------|------------------------------------|-------|----------------------------------------------------------------------------------------------------------|
| named_parameters | Iterator[Tuple[str, nn.Parameter]] | 必选    | 模型的参数列表。用户需自行保证该变量是通过torch.nn.Module的named_parameters()获得。                                               |
| include          | bool                               | 可选    | 如果include为True，返回的结果包含backward_optimizer的参数名，否则不包含。默认值为False。<ul><li>True：包含</li><li>False：不包含</li></ul> |

**返回值说明**

- 成功：返回过滤后的参数。
- 失败：抛出异常。

**使用示例**

注：样例中ddp_model为调用[DistributedModelParallel（TorchRec）](subtable_apis.md#TOPIC_0000002338384297)接口创建的模型对象，此处省略详细定义。

```python
from torchrec.optim.optimizers import in_backward_optimizer_filter

# ddp_model为DistributedModelParallel（TorchRec）创建的模型对象，此处省略详细定义。
ddp_model = ......
bw_optimizer_iter = in_backward_optimizer_filter(ddp_model.named_parameters())
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## KeyedOptimizerWrapper（TorchRec）<a name="ZH-CN_TOPIC_0000002302229608"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

用于封装过滤掉表的参数的优化器。

**函数原型<a name="section1483104721911"></a>**

```python
class KeyedOptimizerWrapper:
    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        optim_factory: OptimizerFactory,
    ) -> None:
```

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|params|Mapping[str, Union[torch.Tensor, ShardedTensor]]|必选|参数列表。用户需自行保证通过[in_backward_optimizer_filter（TorchRec）](#in_backward_optimizer_filtertorchrec)获得。|
|optim_factory|Callable|必选|传入创建优化器的函数。用户需自行保证该函数满足输入为一个参数，输出一个torch.optim.Optimizer对象。|

**返回值说明**

- 成功：参数过滤后的优化器。
- 失败：抛出异常。

**使用示例**

注：样例中ddp_model为调用[DistributedModelParallel（TorchRec）](subtable_apis.md#TOPIC_0000002338384297)接口创建的模型对象，此处省略详细定义。

```python
import torch
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

# ddp_model为DistributedModelParallel（TorchRec）创建的模型对象，此处省略详细定义。
ddp_model = ......
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adam(params, lr=1e-1),
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## CombinedOptimizer（TorchRec）<a name="ZH-CN_TOPIC_0000002302229544"></a>

>[!NOTICE]
>
>此接口为TorchRec开源接口，非Rec SDK Torch对外接口。此章节介绍使用Rec SDK Torch时调用的TorchRec接口支持的参数范围。

**功能描述<a name="section634582619155"></a>**

将多个优化器合并为一个。

**函数原型<a name="section1483104721911"></a>**

```python
class CombinedOptimizer(KeyedOptimizer):
    def __init__(
        self, optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]]
    ) -> None:
```

**参数说明**

| 参数名    | 类型                                                                              | 可选/必选 | 说明                                                                                                                                                            |
|--------|---------------------------------------------------------------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optims | <ul><li>List[KeyedOptimizer]</li><li>List[Tuple[str, KeyedOptimizer]]</li></ul> | 必选    | 需要合并的优化器对象列表，可以为空列表。<br>用户需要自行保证其输入值来源于CombinedOptimizer（TorchRec）或者[DistributedModelParallel（TorchRec）](./subtable_apis.md#distributedmodelparalleltorchrec)的fused_optimizer。 |

**返回值说明**

- 成功：返回合并后优化器。
- 失败：抛出异常。

**使用示例**

注：样例中ddp_model为调用[DistributedModelParallel（TorchRec）](subtable_apis.md#TOPIC_0000002338384297)接口创建的模型对象，此处省略详细定义。

```python
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


ddp_model = ......
dense_optimizer: KeyedOptimizerWrapper = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adagrad(params, lr=0.1),
)
optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。
