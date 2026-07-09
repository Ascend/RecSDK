# 保存与加载接口 <a name="ZH-CN_TOPIC_0000002428477736"></a>

## DynamicEmbDump <a name="ZH-CN_TOPIC_0000002428320084"></a>

**功能描述<a name="section634582619155"></a>**

自动识别Torch模型中的动态嵌入表，并将其并行转储至文件系统，最终生成单一文件。

**函数原型<a name="section1483104721911"></a>**

```cpp
def DynamicEmbDump(
    path: str,
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: Optional[bool] = False,
    pg: dist.ProcessGroup = dist.group.WORLD,
    allow_overwrite: bool = False,
)
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|path|str|必选|存储文件的主目录路径|
|model|nn.Module|必选|包含动态嵌入表的模型对象|
|table_names|Optional[Dict[str, List[str]]]|可选|指定要导出的嵌入集合名称及对应的动态嵌入表列表。若为None，则导出所有动态嵌入表。|
|optim|Optional[bool]|可选|是否保存优化器状态。默认为False。|
|pg|dist.ProcessGroup|可选| 用于控制导出过程中通信范围的进程组。默认使用全局WORLD组。|
|allow_overwrite|bool|可选|是否覆盖已有文件夹。默认为False。|

**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.dump_load import DynamicEmbDump
DynamicEmbDump(
    path=path,
    model=model,
    optim=True,
    allow_overwrite=True
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## DynamicEmbLoad <a name="ZH-CN_TOPIC_0000002430202769"></a>

**功能描述<a name="section634582619155"></a>**

从DynamicEmbDump生成的二进制文件中加载数据，并将其载入Torch模型的动态嵌入表中。

**函数原型<a name="section1483104721911"></a>**

```cpp
def DynamicEmbLoad(
    path: str,
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: bool = False,
    pg: dist.ProcessGroup = dist.group.WORLD,
):
```

**参数说明<a name="section888634319218"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|path|str|必选|加载文件的主目录路径|
|model|nn.Module|必选|包含动态嵌入表的模型对象|
|table_names|Optional[Dict[str, List[str]]]|可选|指定要加载的嵌入集合名称及对应的动态嵌入表列表。若为None，则加载所有找到的动态嵌入表。|
|optim|Optional[bool]|可选|是否加载优化器状态。默认为False。|
|pg|dist.ProcessGroup|可选|用于控制加载过程中通信范围的进程组。默认使用全局WORLD组。|

**使用示例<a name="section193151694205"></a>**

```cpp
from dynamic_emb.distributed.dump_load import DynamicEmbLoad
DynamicEmbLoad(
    path=path,
    model=model,
    optim=True
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。
