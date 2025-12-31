# 多级缓存管理接口<a name="ZH-CN_TOPIC_0000002430082789"></a>

## InitializerType<a name="ZH-CN_TOPIC_0000002430202769"></a>

**功能描述<a name="section634582619155"></a>**

权重初始化类型枚举，定义嵌入表权重的初始化方式。

**函数原型<a name="section1483104721911"></a>**

```cpp
class InitializerType(Enum):
     LINEAR ="linear"
     TRUNCATED_NORMAL ="truncated_normal"
     UNIFORM = "uniform"
```

**参数说明<a name="section888634319218"></a>**

|参数名|说明|
|--|--|
|LINEAR|线性初始化。|
|TRUNCATED_NORMAL|截断正态分布初始化。|
|UNIFORM|均匀分布初始化。|


**返回值说明<a name="section651195312311"></a>**

-   成功：返回InitializerType枚举值。
-   失败：抛出异常。


## Saver<a name="ZH-CN_TOPIC_0000002420844874"></a>

**功能描述<a name="section634582619155"></a>**

多级缓存稀疏表保存加载功能类，提供多级缓存稀疏表数据（稀疏表Embedding，Embedding对应的优化器参数等）的保存、加载接口。

**函数原型<a name="section1483104721911"></a>**

```cpp
class Saver:
    def __init__(self, rank: int = None):
    ...
    def save(self, module: torch.nn.Module, path: str) -> None:
    ...
    def load(self, module: torch.nn.Module, path: str) -> None:
```

**使用约束<a name="section72467171850"></a>**

1.  保存/加载接口仅支持多级缓存保存/加载稀疏表相关数据（稀疏表Embedding，Embedding对应的优化器参数等）。
2.  不支持保存/加载Dense数据（需自行调用Torch原生接口）。
3.  不支持纯显存模式下稀疏表保存/加载。
4.  保存/加载接口不支持训练过程中调用/并发调用/异步调用，仅支持未执行训练/评估时调用接口。
5.  保存/加载接口仅支持保存/加载本地文件系统。

**参数说明<a name="section888634319218"></a>**

|**参数名**|**类型**|**可选/必选**|**说明**|
|--|--|--|--|
|rank|int|可选|当前进程在整个world_size中的rank。当torch分布式环境已初始化时，该参数为可选，此时将使用torch.distributed.get_rank()获取rank；否则该参数为必选。|
|module|torch.nn.Module|必选|模型对象实例。模型（或子模型）需包含类型为EmbCacheShardedEmbeddingBagCollection/EmbCacheShardedEmbeddingCollection的模型实例，且深度不能超过500。使用多级缓存支持的创表接口/分表接口进行模型创建和模型分片时即满足要求。|
|path|string|必选|保存/加载路径，长度取值范围：[1,1024]。<div><div>[!NOTICE]须知</div><div>保存/加载的路径中不能包含软连接和敏感字符（Key、password、privatekey），不能使用特殊路径（如/usr下的路径），且路径的权限不能高于750。</div></div>|


**返回值说明<a name="section651195312311"></a>**

-   成功：接口调用无报错，保存落盘/加载稀疏表数据。
-   失败：抛出异常。

**使用示例<a name="section2553042232"></a>**

```bash
from torchrec_embcache.saver import Saver
...
saver = Saver(rank=rank)
saver.save(model, "save_dir/sparse")  # 保存
saver.load(model, "save_dir/sparse")  # 加载
 
```


