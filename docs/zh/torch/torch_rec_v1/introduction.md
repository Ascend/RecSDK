# 简介<a name="ZH-CN_TOPIC_0000002302229580"></a>

Rec SDK Torch 是基于 PyTorch 和 TorchRec 构建的推荐系统训练框架，依托昇腾 NPU 和 CANN 异构计算架构，为搜索、推荐、广告等场景提供大规模稀疏特征 Embedding 表的高性能分布式训练能力。

## 核心术语<a id="core_terms"></a>

| 术语 | 说明 |
|------|------|
| NPU | Neural Processing Unit，神经网络处理器，昇腾 AI 处理器，用于执行深度学习相关的计算任务。 |
| CANN | Compute Architecture for Neural Networks，昇腾计算架构，是华为针对昇腾 AI 处理器开发的异构计算架构，为深度学习提供算子库和运行时支持。 |
| PyTorch | 一个开源的深度学习框架，提供动态计算图、自动微分和丰富的神经网络模块，广泛用于模型训练与推理。 |
| TorchRec | 基于 PyTorch 的推荐系统库，提供大规模稀疏特征 Embedding 表的分布式训练能力，包括稀疏表、优化器、流水线等核心组件。 |
| Embedding | 将离散型特征（如用户 ID、物品 ID）映射为低维稠密向量的技术，是推荐系统中表征特征语义的核心方法。 |
| 稀疏表（Embedding Table） | 用于存储大规模稀疏特征（如用户 ID、物品 ID）的 Embedding 向量的数据结构。 |
| embedding_dim | 稀疏表的列数，即每个特征的 Embedding 向量维度。 |
| num_embeddings | 稀疏表的行数，即最大特征数量。 |
| JaggedTensor | 持有稀疏 ID 和特征长度的数据结构，每个样本的特征 ID 数量可以不同。 |
| KeyedJaggedTensor | 在 JaggedTensor 基础上增加特征名称键（key），用于区分不同特征组。 |
| Pooling | 将同一特征的多个 Embedding 向量聚合为一个向量的操作，支持 SUM（求和）、MEAN（取平均）、NONE（不做 Pooling）。 |
| pipeline | 训练流水线，用于迭代数据集并进行训练。可将训练流程中部分不存在依赖关系的操作并行执行，提高训练效率。 |
| 纯显存模式 | 稀疏表数据全部存放在 NPU 显存中，通过 [HybridTrainPipelineSparseDist](./api/pipeline_apis.md#hybridtrainpipelinesparsedist) 作为pipeline进行训练。 |
| 多级缓存模式 | 稀疏表数据分布在 CPU 内存和 NPU 显存之间，通过 [EmbCacheTrainPipelineSparseDist](./api/pipeline_apis.md#embcachetrainpipelinesparsedist) 作为pipeline进行训练，支持更大规模的 Embedding 表。 |
| EBC（EmbeddingBagCollection） | 带 Pooling 的稀疏表，查表后自动对同一特征的多条 Embedding 做聚合。纯显存模式使用 [HashEmbeddingBagCollection](./api/table_creation_apis.md#hashembeddingbagcollection)，多级缓存模式使用 [EmbCacheEmbeddingBagCollection](./api/table_creation_apis.md#embcacheembeddingbagcollection)。 |
| EC（EmbeddingCollection） | 不带 Pooling 的稀疏表，查表后返回原始 Embedding 列表。多级缓存模式使用 [EmbCacheEmbeddingCollection](./api/table_creation_apis.md#embcacheembeddingcollection)。 |
| Row-wise/row_wise | 按行分表策略，将稀疏表的不同行分配到不同 NPU 卡上。 |
| Data-parallel/data_parallel | 数据并行分表策略，每张 NPU 卡保留完整的稀疏表副本。 |
| meta 设备 | PyTorch 的虚拟设备类型，用于延迟实际内存分配。在创建稀疏表时指定 "meta" 可先构建模型结构，待 DistributedModelParallel 分表后再实际分配内存。 |
| 准入（Admit） | 控制新特征 ID 是否被加入到稀疏表中，可通过重复次数阈值或展示/点击分数阈值来过滤低频特征。 |
| 淘汰（Evict） | 将长时间未访问或分数较低的特征 ID 从稀疏表中移除，以控制表的内存占用。 |

## 功能介绍<a name="ZH-CN_TOPIC_0000002336268769"></a>

Rec SDK Torch涉及功能如下：

- 模型训练基础功能
    - 支持单机单卡训练、单机多卡分布式训练。
    - 支持基于Torch开发的模型。

- 推荐场景特有功能

    基于Rec SDK Torch的稀疏表方案，Rec SDK Torch提供推荐的必备功能，如非亲和算子卸载、哈希映射功能等。

- 大规模稀疏表特有功能

    支持按照Row-wise的分布式稀疏表切分方式。

**关键功能特性<a name="section7262101710233"></a>**

Rec SDK Torch为用户提供了哈希映射、Row-wise分表、EC/EBC查表模式、流水线查表、查表融合算子、多级缓存等特性。

- 哈希映射

    Torch提供了用于稠密ID查表的nn.Embedding。但在推荐场景，大部分原始特征ID都是离散型，查表时不便于直接使用，常见的做法是将离散ID转为表的行号。为此Rec SDK Torch提供了哈希映射功能，用于将离散ID映射为Embedding表的行号，不需要用户提前做ID转换。

- Row-wise分表

    在将Embedding切分到不同表时，按行对Embedding进行分表，使用取余分桶策略，按照ID取余的余数确定Embedding在表上的分桶位置。

- EC/EBC查表模式

    EC查表对标原生Torch的nn.Embedding功能，对指定的多个ID，进行查表并直接输出查表结果。

    EBC查表对标原生Torch的nn.EmbeddingBag功能，对指定的多个ID，在查表时进行求和或者取平均的Pooling操作。

- 流水线查表

    Rec SDK Torch查表任务由通信、CPU、NPU计算等多个子任务构成。Rec SDK Torch提供了流水线查表方法让子任务之间可以并行以充分发挥硬件算力。

- 查表融合算子

    Rec SDK Torch提供了梯度计算和优化器融合的查表算子以优化查表性能。

- 多级缓存

    以Device Memory + Host Memory（DDR）结合的方式存储稀疏表数据，支撑更大的稀疏表规模。

详细内容请参见[功能特性介绍](./migration_and_training.md#functional_features_description)。

## 软件架构<a name="ZH-CN_TOPIC_0000002302229644"></a>

**图 1**  软件架构图<a name="fig6490183721815"></a>
![](../../figures/torch_rec_v1/软件架构图.png "软件架构图")

Rec SDK Torch基于TorchRec、推荐场景主流框架、CANN和各种硬件和网络，对于搜索、推荐、广告模型训练的应用场景需求，提供极简易用、高性能API，助力昇腾AI处理器完成搜索、推荐、广告等模型的高效训练。

**表 1**  结构图模块介绍

|Rec SDK Torch模块|说明|
|--|--|
|推荐接口层|提供易用性接口、简化用户接入流程，降低迁移成本。支持用户规模化上量。|
|推荐功能层|核心功能实现层，满足用户的使用要求。|
|推荐加速层|性能竞争力核心组件，为整机系统提供更优性能。|
|推荐存储层|支持稀疏表的分布式存储。|
|TorchRec-npu|开源TorchRec的昇腾适配层。|

## 支持的硬件和操作系统<a name="ZH-CN_TOPIC_0000002336268741"></a>

**表 2**  支持的产品列表
<table border="1">
  <tr>
    <th>产品型号</th>
    <th>产品架构</th>
    <th>操作系统版本</th>
  </tr>
  <tr>
    <td rowspan="2"><p>Atlas 800T A2 训练服务器</p><p>Atlas 200T A2 Box16 异构子框</p></td>
    <td>x86_64</td>
    <td>Debian版本：12<br>CentOS版本：7.6</td>
  </tr>
  <tr>
    <td>ARM</td>
    <td>openEuler版本：22.03</td>
  </tr>
  <tr>
    <td>Atlas 800T A3 超节点服务器</td>
    <td>ARM</td>
    <td>openEuler版本：22.03</td>
  </tr>
</table>
