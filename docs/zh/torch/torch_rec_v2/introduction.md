# 简介<a name="ZH-CN_TOPIC_0000002302229580"></a>

## 软件架构<a name="ZH-CN_TOPIC_0000002302229644"></a>

**图 1**  软件架构图<a name="fig6490183721815"></a>
![](../../figures/torch_rec_v1/软件架构图.png "软件架构图")

Rec SDK Torch基于TorchRec、推荐场景主流框架、CANN和各种硬件和网络，对于搜索、推荐、广告模型训练的应用场景需求，提供极简易用、高性能API，助力昇腾AI处理器完成搜索、推荐、广告等模型的高效训练。各个模块的介绍说明如表1所示：

**表 1**  结构图模块介绍

|Rec SDK Torch模块|说明|
|--|--|
|推荐接口层|提供易用性接口、简化用户接入和迁移成本。支持用户规模化上量。|
|推荐功能层|核心功能实现层，满足用户的使用要求。|
|推荐加速层|性能竞争力核心组件，为整机系统提供更优性能。|
|推荐存储层|支持稀疏表的分布式存储。|
|TorchRec-npu|开源TorchRec的昇腾适配层。|

## 功能特性介绍<a name="ZH-CN_TOPIC_0000002336268769"></a>

Rec SDK Torch涉及功能如下：

- 模型训练基础功能
- 支持单机单卡训练、单机多卡分布式训练。
- 支持基于Torch开发的模型。

- 推荐场景特有功能

    基于TorchRec与dynamic_emb动态稀疏表方案，Rec SDK Torch面向搜索、推荐、广告等场景提供完整嵌入能力：兼容`EmbeddingCollection`与 `EmbeddingBagCollection`，支持Row-wise分布式分片与哈希映射；提供HBM+Storage两级Caching加速大表访问；基于score_strategy实现TIMESTAMP、STEP、CUSTOMIZED等准入淘汰策略；并支持全量checkpoint（DynamicEmbDump/DynamicEmbLoad）与基于 score阈值的增量导出（incremental_dump）。底层由HKV（Hierarchical Key-Value）高性能存储库与自定义算子扩展`dynamic_emb_extensions`驱动，兼顾易用性与训练吞吐。

- 大规模稀疏表特有功能

    支持按照Row-wise的分布式稀疏表切分方式。

**关键功能特性<a name="section7262101710233"></a>**

Rec SDK Torch基于TorchRec与dynamic_emb动态稀疏表方案，在推荐训练场景下提供哈希映射、Row-wise分表、稀疏表动态扩容与淘汰、高性能自定义算子等基础能力。

以下介绍四类核心功能特性及其典型用法，接口细节可参见[接口说明](./api/README.md)与[迁移与训练](./migration_and_training.md)。

### EmbeddingCollection 与 EmbeddingBagCollection<a name="section_ec_ebc_intro"></a>

Rec SDK Torch兼容TorchRec原生嵌入模块，通过对应分片器将静态嵌入表替换为动态稀疏表，支持Sequence Embedding与Pooled Embedding两类典型推荐场景。

**表 2**  嵌入模块与分片器对应关系

|TorchRec 模块|分片器|Sharded 模块|适用场景|
|--|--|--|--|
|EmbeddingCollection|DynamicEmbeddingCollectionSharder|ShardedDynamicEmbeddingCollection|Sequence Embedding，输出变长序列embedding（pooling_mode=NONE）|
|EmbeddingBagCollection|DynamicEmbeddingBagCollectionSharder|ShardedDynamicEmbeddingBagCollection|Pooled Embedding，对特征内 ID 做sum/mean等池化后输出固定维度向量|

两类模块均通过`DynamicEmbTableOptions`配置动态表行为（容量、淘汰策略、score 策略、是否启用 caching 等），并配合`DynamicEmbeddingShardingPlanner`、`DistributedModelParallel`完成 Row-wise分布式分片。

**EmbeddingCollection** 适用于多值离散特征、序列推荐等需要保留每个ID对应embedding的场景。分片器`DynamicEmbeddingCollectionSharder`在TorchRec原生流程基础上适配动态表索引去重（`use_index_dedup`）。

**EmbeddingBagCollection** 适用于 CTR、召回等需要对特征内多个 ID 聚合后再送入 DNN 的场景。分片器`DynamicEmbeddingBagCollectionSharder`通过`RwPooledDynamicEmbeddingSharding`接入 `BatchedDynamicEmbeddingBag`，在row-wise 分片下完成分布式pooled lookup。

接口示例：

```python
from torchrec.modules.embedding_modules import EmbeddingCollection, EmbeddingBagCollection
from dynamic_emb import (
    DynamicEmbeddingCollectionSharder,
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbTableOptions,
    DynamicEmbScoreStrategy
)

# Sequence Embedding
ec = EmbeddingCollection(tables=table_configs, device="npu")
ec_sharder = DynamicEmbeddingCollectionSharder(fused_params=fused_params, use_index_dedup=True)

# Pooled Embedding
ebc = EmbeddingBagCollection(tables=embedding_bag_configs, device="npu")
ebc_sharder = DynamicEmbeddingBagCollectionSharder(fused_params=fused_params)

table_options = DynamicEmbTableOptions(
    max_capacity=max_capacity,
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    caching=False,
)
```

### 多级缓存<a name="section_caching_intro"></a>

多级缓存提供 **HBM 热数据层 + 全量 Storage 冷数据层** 的存储架构，用于在NPU HBM有限时加速高频key访问，并将全量数据存放在更大容量的Storage（HKV 或外部存储）中。

**启用条件与配置**

在`DynamicEmbTableOptions`中设置`caching=True`，并合理配置`local_hbm_for_values`（分配给 Cache 的 HBM 字节数）。启用caching时还可配置`external_storage`指定自定义Storage后端（如内存字典等），此时要求`caching=True`。

|配置项|说明|
|--|--|
|caching|是否启用两级存储，默认False|
|local_hbm_for_values|Cache可用 HBM 大小（字节），Planner中通常按设备内存自动推断|
|external_storage|可选，自定义Storage类；未指定时使用HKV作为Storage|

**开启缓存功能时的数据访问流程**

训练前向lookup时先在Cache（HBM上的 `KeyValueTable`）中查找；未命中则回源Storage查表或初始化，并将结果回填Cache。训练反向更新时优先在Cache更新梯度；Cache miss的key回写Storage。Cache满时按淘汰策略驱逐条目，被驱逐的key/value自动写入Storage，保证数据不丢失。

**配套能力**

- **flush / reset**：全量保存前自动flush，将Cache数据同步至Storage；`reset_cache_states()`可清空Cache热数据。
- **限制**：Caching与Prefetch当前仅支持`EmbeddingCollection`（pooling_mode=NONE）；`EmbeddingBagCollection`不支持caching。

接口示例：

```python
table_options = DynamicEmbTableOptions(
    max_capacity=1024 * 1024,
    caching=True,
    local_hbm_for_values=100 * 1024 * 1024,  # 为 Cache 预留 100MB HBM
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
)
```

### 基于 score_strategy 的准入淘汰策略<a name="section_score_strategy_intro"></a>

动态稀疏表在容量满时需要淘汰低频或不重要的key以容纳新特征。Rec SDK Torch通过 **score_strategy** 控制每条key的score生成与更新方式；score越小，表满时越优先被淘汰。score_strategy在`DynamicEmbTableOptions`中按表配置，并自动映射到底层HKV的evict_strategy。

**表 3**  score_strategy 策略说明

|策略|枚举值|Score 来源|用户是否需set_score|映射evict_strategy|典型场景|
|--|--|--|--|--|--|
|TIMESTAMP|0|NPU 设备时间戳，每次前向训练自动更新|否|LRU|默认策略，零配置近似LRU|
|STEP|1|每前向训练一次，表级step+1，同batch内所有key共用|否|CUSTOMIZED|按训练步数淘汰；配合Prefetch时score需单调递增|
|CUSTOMIZED|2|用户自定义，每次前向训练前手动设置|是|CUSTOMIZED|增量保存、自定义淘汰逻辑|
|LFU|3|每次前向训练传入score=1，HKV累加访问频率|否|LFU|最少使用优先淘汰|

**工作原理**

- **TIMESTAMP**：调用NPU`GetSystemCycle`获取时间戳作为score；底层kLru策略由HKV内部LRU机制处理，dump/load时对score做时间戳变换以持久化。
- **STEP / CUSTOMIZED**：find/insert 时向HKV传入显式score，按score大小决定淘汰顺序。
- **LFU**：每次访问传入增量score，HKV在key维度累加频率，淘汰频率最低的条目。

**set_score / get_score**

get_score可获取模型中动态嵌入表当前分数，返回分数字典；set_score为模型中的动态嵌入表设置分数，仅支持CUSTOMIZED策略。详情参见[set_score](./api/dump_load_apis.md#ZH-CN_TOPIC_0000002430202770)。

```python
from dynamic_emb import set_score, get_score

# 获取当前各表 score
score_info = get_score(model)

# 设置当前各表 score
set_score(model, {"model.embedding": {"user_table": 200}})
```

> **说明**：`set_score` 传入的 score 不能为 0，且建议单调递增；新 score 小于旧 score 时会发出告警。

### 全量保存与增量保存<a name="section_dump_load_intro"></a>

Rec SDK Torch提供两套互补的稀疏表持久化能力：**全量保存**用于checkpoint与断点续训，**增量保存**用于训练过程中按score阈值导出新增/更新的key-value。

**表 4**  全量保存与增量保存对比

|维度|全量保存|增量保存|
|--|--|--|
|入口 API|DynamicEmbDump / DynamicEmbLoad|incremental_dump + set_score / get_score|
|输出|写入文件系统|返回内存中的 (keys, values) 张量，需业务自行持久化|
|保存范围|表中全部key-value|score ≥ 阈值的条目（自上次保存以来新增/更新）|
|典型场景|checkpoint、断点续训、异卡加载|训练中周期性导出增量特征|

**全量保存（DynamicEmbDump / DynamicEmbLoad）**

识别DMP封装模型中的 `ShardedDynamicEmbeddingCollection` / `ShardedDynamicEmbeddingBagCollection`，将各rank分片数据并行写入文件系统。每张表生成keys、values、scores 二进制文件及meta JSON（rank 0 写入优化器参数与 evict_strategy）；可选保存优化器状态。启用caching时，保存前自动flush Cache至Storage。

```python
from dynamic_emb import DynamicEmbDump, DynamicEmbLoad

DynamicEmbDump(save_dir, model, optim=True, allow_overwrite=False)
DynamicEmbLoad(save_dir, model, optim=True)
```

**增量保存（incremental_dump）**

利用score单调递增特性，导出所有 **score ≥ 上次保存时 score 阈值** 的条目。训练若干step 后，以`get_score()`返回值为阈值调用`incremental_dump`，返回host侧keys/values及更新后的 score，供下次增量导出或`set_score`使用。多卡场景下各rank匹配结果通过集合通信汇总；启用caching时会合并Cache与Storage中的数据。

```python
from dynamic_emb import get_score, set_score
from dynamic_emb.distributed.incremental_dump import incremental_dump

undump_score = get_score(model)["model.embedding"]
ret_tensors, ret_scores = incremental_dump(
    model,
    score_threshold={"model.embedding": undump_score},
    pg=pg,
)
undump_score = ret_scores["model.embedding"]
```

保存与加载接口详情参见[保存与加载接口](./api/dump_load_apis.md)。
