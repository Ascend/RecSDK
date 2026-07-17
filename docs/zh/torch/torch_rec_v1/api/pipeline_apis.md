# pipeline接口<a name="ZH-CN_TOPIC_0000002336268689"></a>

## HybridTrainPipelineSparseDist<a name="ZH-CN_TOPIC_0000002336149005"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002516109749"></a>

**功能描述<a name="section634582619155"></a>**

创建纯显存模式的分布式训练数据流水线调度器。

**函数原型<a name="section1483104721911"></a>**

```python
class HybridTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        return_loss: bool = False,
        pipe_n_batch: int = 6,
    ) -> None:
```

**参数说明**

| 参数名                 | 类型                    | 可选/必选 | 说明                                                               |
|---------------------|-----------------------|-------|------------------------------------------------------------------|
| model               | torch.nn.Module       | 必选    | 包含EmbeddingBagCollection/HashEmbeddingBagCollection的nn.Module类型。 |
| optimizer           | torch.optim.Optimizer | 必选    | 优化器。优化器的创建方式请参考当前API使用示例。          |
| device              | torch.device          | 必选    | 设备。取值为torch.device("npu")，即npu设备。                                |
| execute_all_batches | bool                  | 可选    | 默认值为True。仅支持默认值，不支持用户自定义。                                        |
| apply_jit           | bool                  | 可选    | 默认值为False。仅支持默认值，不支持用户自定义。                                       |
| return_loss         | bool                  | 可选    | 设置调用HybridTrainPipelineSparseDist对象的progress方法时，是否返回loss值。默认值为False，不返回loss值。                                              |
| pipe_n_batch        | int                   | 可选    | 预取n个batch做并行。取值范围：[1, 12]。默认值为6。                                 |

**返回值说明**

- 成功：返回pipeline。
- 失败：抛出异常。

**使用示例**

注：样例中ddp_model为调用[DistributedModelParallel（TorchRec）](subtable_apis.md#TOPIC_0000002338384297)接口创建的模型对象，此处省略详细定义。

```python
import torch
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist

npu_device = torch.device("npu")
# ddp_model为DistributedModelParallel（TorchRec）创建的模型对象，此处省略详细定义。
ddp_model = ......
dense_optimizer = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adagrad(params, lr=0.01),
)
optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
pipeline = HybridTrainPipelineSparseDist(ddp_model, optimizer, npu_device)
```

### progress<a name="ZH-CN_TOPIC_0000002336268757"></a>

**功能描述<a name="section1217131745816"></a>**

进行训练流水线查表，包括前向传播、反向传播和参数更新。

**函数原型<a name="section858517176587"></a>**

```python
def progress(self, dataloader_iter: Iterator[In]) -> Out:
```

**参数说明<a name="section65810189587"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|dataloader_iter|Iterator[In]|必选|数据集迭代器。该迭代器返回用于查表和训练的Batch类，详细实现请参考当前API使用示例。|

**返回值说明<a name="section10745722145816"></a>**

- 成功：返回模型的输出。
- 失败：抛出StopIteration异常或RuntimeError。

**使用示例<a name="section09971948135814"></a>**

注：样例中省略了Batch和RandomRecDataset的详细定义，详细定义请参见[dataset.py](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/little_demo/dataset.py)。

```python
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchrec.streamable import Pipelineable
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist

# 此处省略Batch和RandomRecDataset的详细定义。
@dataclass
class Batch(Pipelineable):
    pass

class RandomRecDataset(IterableDataset[Batch]):
    pass

BATCH_SIZE = 32
BATCH_NUM = 32
FEAT_NAMES = ["feat0", "feat1", "feat2"]
ID_RANGES = [400, 4000, 400]

dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
dataloader = DataLoader(
    dataset,
    batch_size=None,
    batch_sampler=None,
    pin_memory=True,
    prefetch_factor=32,
    pin_memory_device="npu",
    num_workers=4,
)
dataloader_iter = iter(dataloader)

# pipeline创建请参考前面的HybridTrainPipelineSparseDist初始化方法，此处省略详细定义。
pipeline: HybridTrainPipelineSparseDist = ......
# 若创建HybridTrainPipelineSparseDist对象时，传递return_loss为False，则：
output = pipeline.progress(dataloader_iter)
# 若创建HybridTrainPipelineSparseDist对象时，传递return_loss为True，则：
output, loss = pipeline.progress(dataloader_iter)
```

## EmbCacheTrainPipelineSparseDist<a name="ZH-CN_TOPIC_0000002396403128"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002483989806"></a>

**功能描述<a name="section634582619155"></a>**

创建多级缓存模式的分布式训练数据流水线调度器。

**函数原型<a name="section1483104721911"></a>**

```python
class EmbCacheTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cpu_device: torch.device,
        npu_device: torch.device,
        return_loss: bool = False,
        evict_step_interval: Optional[int] = None,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[EmbCacheTrainPipelineContext] = EmbCacheTrainPipelineContext,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        custom_model_zero_grad: Optional[Callable] = None,
        custom_model_bwd: Optional[Callable] = None,
    ) -> None:
```

**参数说明**

| 参数名                 | 类型                               | 可选/必选 | 说明                                                                                             |
| ---------------------- | ---------------------------------- | --------- | ------------------------------------------------------------------------------------------------ |
| model                  | torch.nn.Module                    | 必选      | 包含EmbCacheEmbeddingBagCollection/EmbCacheEmbeddingCollection的nn.Module对象。                  |
| optimizer              | torch.optim.Optimizer              | 必选      | 优化器。                                                                                         |
| cpu_device             | torch.device                       | 必选      | CPU设备。                                                                                        |
| npu_device             | torch.device                       | 必选      | NPU设备。                                                                                        |
| return_loss            | bool                               | 可选      | 设置调用EmbCacheTrainPipelineSparseDist对象的progress方法时，是否返回loss值。默认值为False，不返回loss值。       |
| evict_step_interval    | int                                | 可选      | 淘汰步数间隔，默认值为None。当创建稀疏表并开启淘汰功能时，该参数必传。参数非None时需大于等于10。 |
| execute_all_batches    | bool                               | 可选      | 是否执行所有批次，默认值为True，不支持用户自定义。                                               |
| apply_jit              | bool                               | 可选      | 是否应用JIT编译，默认值为False，不支持用户自定义。                                               |
| context_type           | Type[EmbCacheTrainPipelineContext] | 可选      | 上下文类型，默认值为EmbCacheTrainPipelineContext。                                               |
| pipeline_postproc      | bool                               | 可选      | 是否启用流水线后处理，默认值为False，与torchrec的TrainPipelineSparseDist一致。                   |
| custom_model_fwd       | Callable                           | 可选      | 自定义模型前向函数，默认值为None，与torchrec的TrainPipelineSparseDist一致。                      |
| custom_model_zero_grad | Callable                           | 可选      | zero_grad自定义函数，默认为None。                                                                |
| custom_model_bwd       | Callable                           | 可选      | 自定义模型反向函数，默认为None。                                                                 |

**返回值说明**

- 成功：返回EmbCacheTrainPipelineSparseDist对象。
- 失败：抛出异常。

**使用示例**

注：样例中ddp_model为调用[DistributedModelParallel（TorchRec）](subtable_apis.md#TOPIC_0000002338384297)接口创建的模型对象，此处省略详细定义。且当前接口为多级缓存模式的pipeline，调用DistributedModelParallel接口时需传入多级缓存模式对应的[EmbCacheEmbeddingCollection](./table_creation_apis.md#embcacheembeddingcollection)/[EmbCacheEmbeddingBagCollection](./table_creation_apis.md#embcacheembeddingbagcollection)接口创建的稀疏表。

```python
import torch
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist

cpu_device: torch.device = torch.device("cpu")
npu_device: torch.device = torch.device("npu")
# ddp_model为DistributedModelParallel（TorchRec）创建的模型对象，此处省略详细定义。
ddp_model = ......
dense_optimizer: KeyedOptimizerWrapper = KeyedOptimizerWrapper(
    dict(in_backward_optimizer_filter(ddp_model.named_parameters())),
    lambda params: torch.optim.Adagrad(params, lr=0.01),
)
optimizer = CombinedOptimizer([ddp_model.fused_optimizer, dense_optimizer])
pipeline = EmbCacheTrainPipelineSparseDist(
    ddp_model, optimizer, cpu_device=cpu_device, npu_device=npu_device, execute_all_batches=True
)
```

### progress<a name="ZH-CN_TOPIC_0000002396563016"></a>

**功能描述<a name="section634582619155"></a>**

进行训练流水线查表，包括前向传播、反向传播和参数更新。

**函数原型<a name="section1483104721911"></a>**

```python
def progress(self, dataloader_iter: Iterator[In]) -> Out:
```

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|dataloader_iter|Iterator[In]|必选|数据集迭代器。该迭代器返回用于查表和训练的Batch类，详细实现请参考当前API使用示例。|

**返回值说明**

- 成功：返回训练输出结果。

- 失败：抛出StopIteration异常或RuntimeError。

**使用示例<a name="section09971948135815"></a>**

注：样例中省略了Batch和RandomRecDataset的详细定义，详细定义请参见[dataset.py](https://gitcode.com/Ascend/RecSDK/blob/develop_examples_and_tools/torch_examples/little_demo/dataset.py)。pipeline的完整使用请参见[test](../../../../../training/torch_rec_v1/torchrec_embcache/tests/acc_test/test_embedding_ec_cache_pipeline.py)。

```python
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchrec.streamable import Pipelineable
from torchrec_embcache.distributed.train_pipeline import EmbCacheTrainPipelineSparseDist

# 此处省略Batch和RandomRecDataset的详细定义。
@dataclass
class Batch(Pipelineable):
    pass

class RandomRecDataset(IterableDataset[Batch]):
    pass

BATCH_SIZE = 32
BATCH_NUM = 32
FEAT_NAMES = ["feat0", "feat1", "feat2"]
ID_RANGES = [400, 4000, 400]

dataset = RandomRecDataset(BATCH_SIZE, BATCH_NUM, FEAT_NAMES, ID_RANGES)
dataloader = DataLoader(
    dataset,
    batch_size=None,
    batch_sampler=None,
    pin_memory=True,
    prefetch_factor=32,
    pin_memory_device="npu",
    num_workers=4,
)
dataloader_iter = iter(dataloader)

# pipeline创建请参考前面的EmbCacheTrainPipelineSparseDist初始化方法，此处省略详细定义。
pipeline: EmbCacheTrainPipelineSparseDist = ......
# 若创建EmbCacheTrainPipelineSparseDist对象时，传递return_loss为False，则：
output = pipeline.progress(dataloader_iter)
# 若创建EmbCacheTrainPipelineSparseDist对象时，传递return_loss为True，则：
output, loss = pipeline.progress(dataloader_iter)
```
