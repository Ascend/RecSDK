# 保存与加载接口 <a name="ZH-CN_TOPIC_0000002428477736"></a>

## DynamicEmbDump <a name="ZH-CN_TOPIC_0000002428320084"></a>

**功能描述<a name="section634582619155"></a>**

自动识别Torch模型中的动态嵌入表，并将其并行转储至文件系统，最终生成单一文件。

**函数原型<a name="section1483104721911"></a>**

```python
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

```python
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

## set_score <a name="ZH-CN_TOPIC_0000002430202770"></a>

**功能描述<a name="section634582619156"></a>**

为模型中的动态嵌入表设置分数（score）。分数可用于控制后续增量导出等依赖分数阈值的处理逻辑。

**函数原型<a name="section1483104721912"></a>**

```cpp
def set_score(
    model: nn.Module,
    table_score: Union[int, Dict[str, Dict[str, int]]],
) -> None
```

**参数说明<a name="section888634319219"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|model|nn.Module|必选|包含动态嵌入表的模型对象。|
|table_score|Union[int, Dict[str, Dict[str, int]]]|必选|分数设置策略。传入`int`时表示对所有动态嵌入表统一设置；传入`Dict[str, Dict[str, int]]`时，外层key为embedding collection在模型中的路径，内层key为表名，value为分数。|

**使用示例<a name="section193151694206"></a>**

```cpp
from dynamic_emb import set_score

# 所有动态嵌入表设置为同一分数
set_score(model, 100)

# 按 collection 和表名分别设置分数
set_score(
    model,
    {
        "model.embedding": {
            "user_table": 100,
            "item_table": 200,
        }
    },
)
```

## get_score <a name="ZH-CN_TOPIC_0000002430202771"></a>

**功能描述<a name="section634582619157"></a>**

获取模型中动态嵌入表当前分数，返回按embedding collection和表名组织的分数字典。

**函数原型<a name="section1483104721913"></a>**

```cpp
def get_score(
    model: nn.Module,
) -> Optional[Dict[str, Dict[str, int]]]
```

**参数说明<a name="section888634319220"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|model|nn.Module|必选|包含动态嵌入表的模型对象。|

**返回值说明<a name="section888634319221"></a>**

|返回值类型|说明|
|--|--|
|Optional[Dict[str, Dict[str, int]]]|成功时返回分数字典；若模型中不存在动态嵌入集合或动态嵌入表，则返回`None`并给出告警。|

**使用示例<a name="section193151694207"></a>**

```cpp
from dynamic_emb import get_score

score_info = get_score(model)
if score_info is not None:
    print(score_info)
```

## DynamicEmbLoad <a name="ZH-CN_TOPIC_0000002430202769"></a>

**功能描述<a name="section634582619155"></a>**

从DynamicEmbDump生成的二进制文件中加载数据，并将其载入Torch模型的动态嵌入表中。

**函数原型<a name="section1483104721911"></a>**

```python
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

```python
from dynamic_emb.distributed.dump_load import DynamicEmbLoad
DynamicEmbLoad(
    path=path,
    model=model,
    optim=True
)
```

**参考资源<a name="section426664933312"></a>**

接口调用流程及示例可参见[迁移与训练](../migration_and_training.md)。

## incremental_dump <a name="ZH-CN_TOPIC_0000002430202869"></a>

**功能描述<a name="section634582619159"></a>**

根据分数阈值（score threshold）从模型中的动态嵌入表增量导出满足条件的 index-embedding 键值对。仅导出表中 **score 大于等于阈值** 的条目，并返回导出后的 keys、values 以及各表当前 score，供下一次增量导出或 `set_score` 使用。

接口会自动识别模型中的 `ShardedDynamicEmbeddingCollection`，在分布式场景下可通过 `pg` 指定通信进程组；多卡时各 rank 本地分片上的匹配结果会通过集合通信汇总后返回。

**函数原型<a name="section1483104721917"></a>**

```python
def incremental_dump(
    model: nn.Module,
    score_threshold: Union[int, Dict[str, Dict[str, int]]],
    pg: Optional[dist.ProcessGroup] = None,
) -> Union[
    Tuple[
        Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Dict[str, Dict[str, int]],
    ],
    None,
]:
```

**参数说明<a name="section888634319518"></a>**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|model|nn.Module|必选|包含动态嵌入表的 Torch 模型对象（通常为经 `DistributedModelParallel` 封装后的模型）。|
|score_threshold|Union[int, Dict[str, Dict[str, int]]]|必选|分数阈值策略。传入 `int` 时，模型中所有动态嵌入表均使用该阈值进行增量导出；传入 `Dict[str, Dict[str, int]]` 时，外层 key 为 embedding collection 在模型中的路径（`module_path`），内层 key 为动态嵌入表名，value 为该表的分数阈值，仅对字典中出现的表执行导出。|
|pg|Optional[dist.ProcessGroup]|可选|控制增量导出过程中通信范围的进程组。默认为 `None`；在分布式训练且 world_size 大于 1 时用于跨 rank 汇总匹配结果。|

**返回值说明<a name="section888634319221"></a>**

|返回值类型|说明|
|--|--|
|Tuple[Dict, Dict]|成功时返回二元组 `(ret_tensors, ret_scores)`。`ret_tensors[collection_path][table_name]` 为 `(keys, values)`：`keys` 为 host 侧匹配的索引张量，`values` 为对应的 embedding（含优化器状态时按表内布局展平）。`ret_scores[collection_path][table_name]` 为导出完成后的当前 score，可作为下一次 `incremental_dump` 的 `score_threshold` 输入，或通过 `set_score` 写回模型。当 `score_threshold` 为 `int` 时，`ret_scores` 包含所有动态嵌入表的 score；为字典时仅包含本次实际导出的表。|
|None|模型中不存在 `ShardedDynamicEmbeddingCollection`，或不存在动态嵌入表时，发出告警并返回 `None`。|

**异常说明<a name="section888634319522"></a>**

|异常|说明|
|--|--|
|ValueError|`model`、`score_threshold` 或 `pg` 类型/取值不合法时抛出。例如 `score_threshold` 既不是 `int` 也不是 `Dict[str, Dict[str, int]]`，或阈值超出允许的 int64 范围。|

**使用示例<a name="section193171694205"></a>**

```python
import torch.distributed as dist
from torchrec.distributed.comm import intra_and_cross_node_pg
from dynamic_emb.distributed.incremental_dump import (
    get_score,
    set_score,
    incremental_dump,
)

# 1. 读取当前各表 score（可选）
score_info = get_score(model)
assert score_info is not None
prefix_path = "model"  # embedding collection 在模型中的路径
undump_score = score_info[prefix_path]

# 2. 训练若干 step 后，按上次 score 作为阈值做增量导出
ret_tensors, ret_scores = incremental_dump(
    model,
    {prefix_path: undump_score},
    intra_and_cross_node_pg()[0],
)
undump_score = ret_scores[prefix_path]

# 3. 校验：本次 forward 访问过的索引应出现在导出 keys 中
for table_name, indices in unique_indices_per_table.items():
    dumped_keys = set(ret_tensors[prefix_path][table_name][0].tolist())
    assert indices.issubset(dumped_keys)
```
