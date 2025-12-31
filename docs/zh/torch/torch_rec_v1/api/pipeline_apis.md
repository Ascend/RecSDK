# pipeline接口<a name="ZH-CN_TOPIC_0000002336268689"></a>

## HybridTrainPipelineSparseDist<a name="ZH-CN_TOPIC_0000002336149005"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002516109749"></a>

**功能描述<a name="section634582619155"></a>**

创建流水查表。

**函数原型<a name="section1483104721911"></a>**

```cpp
class HybridTrainPipelineSparseDist:
    def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|model|torch.nn.Module|必选|包含EmbeddingBagCollection的nn.Module类型。|
|optimizer|torch.optim.Optimizer|必选|优化器。优化器的创建方式请参考[步骤7](../quick_start.md#接口调用介绍 )中定义的优化器。|
|device|torch.device|必选|设备。取值为torch.device("npu")，即npu设备。|
|return_loss|bool|可选|是否返回loss。默认值为False。|
|pipe_n_batch|int|可选|预取n个batch做并行。取值范围：[1, 12]。默认值为6。|
|execute_all_batches|bool|可选|默认值为True。仅支持默认值，不支持用户自定义。|
|apply_jit|bool|可选|默认值为False。仅支持默认值，不支持用户自定义。|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回pipeline。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```cpp
from hybrid_torchrec.distributed.hybrid_train_pipeline import HybridTrainPipelineSparseDist
pipeline = HybridTrainPipelineSparseDist(model, optimizer, device)
```


### progress<a name="ZH-CN_TOPIC_0000002336268757"></a>

**功能描述<a name="section1217131745816"></a>**

进行流水训练。

**函数原型<a name="section858517176587"></a>**

```cpp
def progress(dataloader_iter: Iterator[In]) -> Out:
```

**参数说明<a name="section65810189587"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|dataloader_iter|Iterator[In]|必选|数据集迭代器。该迭代器返回用于查表和训练的Batch类，参考[步骤1](../quick_start.md#接口调用介绍)。|


**返回值说明<a name="section10745722145816"></a>**

-   成功：返回模型的输出。
-   失败：抛出异常。

**使用示例<a name="section09971948135814"></a>**

```cpp
pipeline.progress(dataloader_iter)
```



## EmbCacheTrainPipelineSparseDist<a name="ZH-CN_TOPIC_0000002396403128"></a>

### 初始化<a name="ZH-CN_TOPIC_0000002483989806"></a>

**功能描述<a name="section634582619155"></a>**

创建多级缓存流水查表。

**函数原型<a name="section1483104721911"></a>**

```cpp
class EmbCacheTrainPipelineSparseDist:
     def __init__(**kwargs):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|model|torch.nn.Module|必选|包含EmbCacheEmbeddingBagCollection和EmbCacheEmbeddingCollection的nn.Module对象|
|optimizer|torch.optim.Optimizer|必选|优化器|
|cpu_device|torch.device|必选|CPU设备|
|npu_device|torch.device|必选|NPU设备|
|return_loss|bool|可选|是否返回loss，默认值为False|
|execute*_*all_batches|bool|可选|是否执行所有批次，默认值为True，不支持用户自定义|
|apply_jit|bool|可选|是否应用JIT编译，默认值为False，不支持用户自定义|
|context_type|Type[EmbCacheTrainPipelineContext]|可选|上下文类型，默认值为EmbCacheTrainPipelineContext|
|pipeline_postproc|bool|可选|是否启用流水线后处理，默认值为False，与torchrec的TrainPipelineSparseDist一致|
|custom_model_fwd|Callable|可选|自定义模型前向函数，默认值为None，与torchrec的TrainPipelineSparseDist一致|
|custom_model_zero_grad|Callable|可选|zero_grad自定义函数，默认为None|
|custom_model_bwd|Callable|可选|自定义模型反向函数，默认为None|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回EmbCacheTrainPipelineSparseDist对象。
-   失败：抛出异常。


### progress<a name="ZH-CN_TOPIC_0000002396563016"></a>

**功能描述<a name="section634582619155"></a>**

执行训练流水线的一个步骤，包括前向传播、反向传播和参数更新。

**函数原型<a name="section1483104721911"></a>**

```cpp
def progress(self, dataloader_iter: Iterator[In]) -> Out:
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|dataloader_iter|Iterator[In]|必选|数据加载器迭代器|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回训练输出结果。

-   失败：抛出StopIteration异常或RuntimeError。



