# 推荐全栈介绍

## 简介

### 产品背景

随着人工智能技术的演进，电商、长短视频、社交等行业对搜索系统、推荐系统以及广告系统的效果诉求越发强烈。在如今互联网发达的时代，大量的用户数据、商品数据、视频资料，使信息剧烈爆炸，也使得搜索推荐广告系统的价值进一步凸显。搜索推荐广告系统的需求增长必然带来对算力的需求，如何部署更大算力并充分发挥算力成为从业人员重点关注的问题。

### 产品定义

Rec SDK 作为面向互联网市场搜索推荐广告场景的应用使能SDK产品，基于昇腾平台提供完整的推荐系统框架。联合 fbgemm-ascend（基于昇腾平台的高性能PyTorch NPU算子库）与 HierarchicalKV-ascend（基于昇腾平台的高性能key-value存储加速库），共同为搜索推荐广告模型训练提供从上层框架、高性能算子计算到大规模稀疏特征存储的全栈式解决方案，支撑超大规模推荐场景，助力高效完成昇腾平台上推荐模型的训练与部署。产品架构图参见图1。

图1 产品架构图

![推荐全栈架构图](./figures/rec_full_stack/全栈架构图.png)

**架构图说明：**

1.硬件层

- 昇腾硬件：包括昇腾A2、A3及950系列，提供底层算力支撑。

2.CANN软件栈

- 基础算子：支持Ascend C、CATLASS、Triton等算子开发语言与库。
- 自动编译优化：提供AutoFuse、Inductor等算子自动融合能力，提升计算效率。

3.AI框架与适配层

- AI框架：兼容PyTorch、TensorFlow业界主流框架。
- 框架适配：通过PyTorch-Adapter（PT-Adapter）、TensorFlow-Adapter（TF-Adapter）实现模型在昇腾平台的无缝迁移与运行。

4.应用层

- Rec SDK：面向推荐系统场景，提供torch_rec_v1/v2、tf_rec_v1/v2等推荐框架。
- 关键加速组件：
  - HierarchicalKV-ascend：面向推荐系统的高性能key-value存储加速库。
  - FBGEMM-ascend：高性能矩阵计算加速库。

5.开发工具

- MindStudio：提供全流程开发、调试与调优工具链。

### 产品价值

表1 产品价值说明

| 产品特性 | 产品价值                                           |
| -------- | -------------------------------------------------- |
| 易用     | 极简易用API，快速开发算法模型。                    |
| 精度     | 在标准模型验证精度误差小于万分之一。               |
| 性能     | 高效多级流水加速，高速集合通信加速，极致性能优化。 |

## 组件介绍

### 组件概览

推荐解决方案由以下三个核心组件构成（产品架构图参见图1）：

表2 推荐解决方案核心组件说明

| 组件                      | 定位         | 核心职责                                |
| :------------------------ | :----------- | :-------------------------------------- |
| Rec SDK               | 推荐框架     | 模型开发、TB级Embedding存储、全流程调度 |
| fbgemm-ascend         | 高性能算子库 | 嵌入表查询、融合算子、计算加速          |
| HierarchicalKV-ascend | KV存储加速库 | 大规模稀疏特征存储与低延迟访问          |

#### Rec SDK

[Rec SDK](https://gitcode.com/Ascend/RecSDK)作为面向互联网市场搜索推荐广告的应用使能SDK产品，对于搜索推荐广告模型训练的应用场景需求，提供基于昇腾平台的搜索推荐广告框架，支撑大规模搜索推荐广告场景，助力完成搜索推荐广告模型的高效训练。

Rec SDK的功能涉及：

- 模型训练基础功能。支持单机单卡训练、多机多卡分布式训练。

- 推荐场景特有功能。基于Rec SDK的稀疏表方案，Rec SDK提供必备功能，如特征保存和加载、特征准入、特征淘汰等。

- 大规模稀疏表特有功能。支持加速卡内存、主机内存、主机磁盘多级存储、支持多机存储、支持动态扩容。规模可超10TB。

Rec SDK由多个组件构成，包含`tf_rec_v1`、`tf_rec_v2`、`torch_rec_v1`、`torch_rec_v2`和`rec_ops`多个内部组件，参见表3：

表3 Rec SDK组件说明

| 组件名称     | 基础框架           | 适配状态 | 框架类型     | 功能描述                                                     |
| ------------ | ------------------ | -------- | ------------ | ------------------------------------------------------------ |
| tf_rec_v1    | TensorFlow         | 非全下沉 | 稀疏推荐框架 | 基于TensorFlow，适配NPU设备的非全下沉稀疏推荐框架，支持Atlas A2/A3/A5设备。 |
| tf_rec_v2    | TensorFlow         | 全下沉   | 稀疏推荐框架 | 基于TensorFlow，适配NPU设备的全下沉稀疏推荐框架，仅支持Atlas A5设备。 |
| torch_rec_v1 | PyTorch + TorchRec | 非全下沉 | 稀疏推荐框架 | 基于PyTorch、[TorchRec](https://github.com/meta-pytorch/torchrec/tree/release/v1.2.0)开源软件，适配NPU设备的非全下沉稀疏推荐框架，支持Atlas A2/A3/A5设备。 |
| torch_rec_v2 | PyTorch + TorchRec | 全下沉   | 稀疏推荐框架 | 基于PyTorch、[DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/v25.09)、[TorchRec](https://github.com/meta-pytorch/torchrec/tree/release/v1.2.0)开源软件，适配NPU设备的全下沉稀疏推荐框架，仅支持Atlas A5设备。 |
| rec_ops      | -                  | -        | 算子         | 基于Ascend C开发的推荐场景自定义算子集，支持Atlas A2/A3/A5设备。 |

关键术语说明：

- 非全下沉：稀疏表哈希映射、去重分桶相关操作在CPU上执行，其余计算任务在NPU上执行的混合模式

- 全下沉：指所有计算任务都下沉到NPU上执行，以获得更好的兼容性

#### fbgemm-ascend

[fbgemm-ascend](https://gitcode.com/Ascend/fbgemm-ascend)是 FBGEMM 算子在昇腾 NPU 平台上的算子实现，通过 `torch.ops.fbgemm.*` 提供高性能稀疏/稠密算子，帮助推荐、搜索等场景在 Ascend 设备上获得与 GPU 同步的训练体验。目标是承接社区 [FBGEMM](https://github.com/pytorch/FBGEMM) 的新能力，并针对 Ascend AI Core 进行深度调优。

核心功能：

- Ascend 定制算子：提供 AscendC 实现的核心推荐算子，并向上提供 Python 绑定。

- PyTorch 生态无缝集成：与 Torch、TorchRec 等组件协同，直接复用 `torch.ops.fbgemm.*` 接口。

- 多芯片自适应：自动探测 Atlas A2/A3/A5 训练系列产品芯片，区分编译目标。

#### HierarchicalKV-ascend

[HierarchicalKV-ascend](https://gitcode.com/Ascend/HierarchicalKV-ascend)（下称HKV）是 [开源HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV/commit/bbe2ee1858b6e54bccf9106e9f3c2d8c1c5d248c) 在昇腾 NPU 平台上的算子实现，是一个面向推荐系统的高性能key-value存储加速库。 在推荐系统中，HKV提供了大容量、高性能的动态Embedding表的增删改查能力。

核心功能：

- 支持分级存储

- 支持可定制的淘汰策略

- keys和values存储分离，keys仅存储于HBM

#### 组件协同

内部协同关系

- Rec SDK 内部组件：`tf_rec_v1`、`tf_rec_v2`、`torch_rec_v1`、`torch_rec_v2` 四个框架组件各自独立，分别适配 TensorFlow 和 PyTorch 生态，满足不同算法框架的诉求；`rec_ops` 作为公共算子集，为各框架组件提供基础算子能力。

- fbgemm-ascend 与 Rec SDK 的协同：`torch_rec_v1` 和 `torch_rec_v2` 基于 [TorchRec](https://github.com/meta-pytorch/torchrec/tree/release/v1.2.0) 开源框架，而 TorchRec 的 Embedding 算子实现复用 FBGEMM。因此，`fbgemm-ascend` 作为昇腾平台的算子实现，通过 `torch.ops.fbgemm.*` 接口为两个 PyTorch 框架组件提供高性能 Embedding 查询、融合算子等核心计算能力。

- HierarchicalKV-ascend 与 Rec SDK 的协同：`torch_rec_v2` 基于 [TorchRec](https://github.com/meta-pytorch/torchrec/tree/release/v1.2.0) 和 [DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/v25.09) 开源框架，而 `HierarchicalKV-ascend`作为 DynamicEmb 的底层 KV 存储引擎，为大规模稀疏特征提供高性能的增删改查能力，支持 HBM 内动态扩容与可定制淘汰策略。

协同关系示意

表4 组件协同关系说明

| 组件                    | 应用位置                       | 协同方式                                                     |
| ----------------------- | ------------------------------ | ------------------------------------------------------------ |
| `fbgemm-ascend`         | `torch_rec_v1`、`torch_rec_v2` | 作为底层算子库，通过 `torch.ops.fbgemm.*` 接口提供 Embedding 计算能力 |
| `HierarchicalKV-ascend` | `torch_rec_v2`                 | 作为 DynamicEmb 的底层 KV 存储层，提供大容量 Embedding 存储与访问能力 |

独立使用能力

除与 Rec SDK 集成外，`fbgemm-ascend` 和 `HierarchicalKV-ascend` 也支持独立使用：

- `fbgemm-ascend`：用户可在原生 PyTorch 环境中直接调用 `torch.ops.fbgemm.*` 算子，无缝复用现有 FBGEMM 生态代码。

- `HierarchicalKV-ascend`：可作为独立的 KV 存储加速库，集成到自定义的训练框架或推理服务中。

### 周边组件

除核心的 Rec SDK、fbgemm-ascend、HierarchicalKV-ascend 三大组件外，推荐解决方案还依托昇腾生态的周边组件，提供从算子开发、模型迁移到性能调优、精度分析的全流程能力。

#### 周边组件概览

表5 周边组件说明

| 类别         | 组件                                                                                                                              | 功能说明                                                                                                                                                   | 在推荐方案中的作用                                           |
| :----------- |:--------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------| :----------------------------------------------------------- |
| 算子开发 | [Ascend C](https://www.hiascend.com/document-scene/zh/devScene/operatordev/index.html)                                          | 面向昇腾AI处理器的算子编程语言，支持C/C++标准规范，提供多层接口抽象与自动并行计算                                                                                                           | 支撑`rec_ops`及自定义算子的高效开发                          |
|              | [CATLASS](https://gitcode.com/cann/catlass)                   | 高性能矩阵乘类算子模板库，提供GEMM类算子的模板化实现与性能优化模块；此外，在 Inductor-Ascend 遇到矩阵乘相关算子时，CATLASS 可通过模版生成高性能内核并配合 autotuning 机制自动选择最优的 tile 配置                               | 加速推荐模型中大量存在的矩阵运算，为矩阵乘类算子提供标杆性能模板 |
|              | [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)                                                                       | 基于昇腾平台的Triton编译框架，支持将Python编写的算子编译为高效NPU内核；此外，Triton-Ascend 是昇腾平台中 PyTorch 后端编译链的关键后端，负责接收 Inductor-Ascend 生成的 Triton DSL 代码，并最终编译、优化生成可在昇腾硬件上高效执行的机器码 | 支持动态shape场景下的算子编译优化，拓展推荐模型的灵活性      |
| 算子融合 | [AutoFuse](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse/autofuse_1_0001.html)                  | 基于Ascend C的自动融合框架，自动识别融合范围并生成融合算子代码                                                                                                                    | 减少推荐网络中Vector计算间的内存搬运，缓解Memory Bound问题，提升执行性能 |
|              | [Inductor-Ascend](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/torch_npu/_inductor/docs/overview/overview.md)                 | Inductor-Ascend在继承Pytorch社区Inductor能力的基础上，针对昇腾Ascend硬件，进行了亲和性改进和优化。其目标是：提供昇腾亲和的torch.compile图模式后端；生成昇腾亲和的Triton DSL，支持基于triton的算子自动融合；支持动态shape        | 与`torch_rec_v1/v2`协同，支持PyTorch模型的自动编译优化       |
| 框架适配 | [PyTorch-Adapter](https://www.hiascend.com/document/detail/zh/Pytorch/730/productoverview/docs/zh/overview/product_overview.md) | PyTorch框架的昇腾适配层，使PyTorch模型能运行在昇腾设备上                                                                                                                    | 支撑`torch_rec_v1/v2`在NPU上的运行                           |
|              | [TensorFlow-Adapter](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/index/index.html)                      | TensorFlow框架的昇腾适配层，使TensorFlow模型能运行在昇腾设备上                                                                                                              | 支撑`tf_rec_v1/v2`在NPU上的运行                              |
| 硬件使能 | [CANN](https://www.hiascend.com/cann)                                                                                           | 昇腾异构计算架构，提供模型推理与训练的基础能力                                                                                                                                | 所有上层组件的基础软件栈，提供芯片使能、算子库、图编译等核心能力 |

#### 周边组件能力支撑

依托上述周边组件，推荐解决方案对外提供以下关键能力：

- 算子开发：基于 [Ascend C](https://www.hiascend.com/document-scene/zh/devScene/operatordev/index.html)、[CATLASS](https://gitcode.com/cann/catlass)、[Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)，支持从标准算子到高性能矩阵乘算子的灵活开发与定制

- 算子自动融合：通过 [AutoFuse](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse/autofuse_1_0001.html) 与 [Inductor-Ascend](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/pytorch_compilation_mode.md)，自动识别融合机会并生成融合算子，减少内存搬运，释放昇腾算力

- 样例演示：提供完整的算子开发与模型迁移示例，帮助用户快速上手

- 模型迁移：借助 [PyTorch-Adapter](https://www.hiascend.com/document/detail/zh/Pytorch/730/productoverview/docs/zh/overview/product_overview.md)、[TensorFlow-Adapter](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/850/index/index.html)，将存量 TensorFlow/PyTorch 模型快速迁移至昇腾平台

- 模型开发：基于 [Rec SDK](https://gitcode.com/Ascend/RecSDK) 的高层 API，快速构建推荐模型训练任务

- 性能分析和调优：依托 CANN 提供的[性能分析工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/Profiling/atlasprofiling_16_0144.html)（如 msProf），定位模型执行瓶颈和性能调优

- 精度分析：利用 CANN [精度调试工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/ModelAccuracyAnalyzer/atlasaccuracy_16_1000.html)，验证模型迁移前后的精度一致性

#### 依赖关系说明

周边组件与核心组件（`Rec SDK`、`fbgemm-ascend` 和 `HierarchicalKV-ascend`）的关系如下：

- CANN 作为基础软件栈，为所有组件提供底层算力支持与运行时环境

- PyTorch-Adapter / TensorFlow-Adapter 分别支撑 `torch_rec_v1/v2` 与 `tf_rec_v1/v2` 在昇腾设备上的运行

- Inductor-Ascend 与 `torch_rec_v1/v2` 协同，实现 PyTorch 模型的自动编译与性能优化

- AutoFuse 在图编译阶段对推荐网络进行自动融合优化，用户无需感知融合细节

- Ascend C / CATLASS / Triton-Ascend 为 `rec_ops` 的自定义算子开发提供编程框架与模板库

## 术语表

| 术语      | 定义                                                         |
| --------- | ------------------------------------------------------------ |
| HBM       | HBM（High Bandwidth Memory，高带宽内存） 是一种高性能 DRAM。 |
| Embedding | Embedding指的是将离散的、高维的数据（如单词、句子、图像像素）映射到一个低维、稠密的连续向量空间的过程，这个生成的向量就被称为该数据的“嵌入向量”。 |
| 稀疏特征  | 稀疏特征是机器学习与数据挖掘中一个非常基础且重要的概念。它与稠密特征相对，指的是在大多数情况下取值为0（或为空、假），仅在少数情况下有非零（或非空）取值的特征。 |
| 特征准入  | 在模型训练或推理前，判断一个从未出现过的特征（如新上架的商品ID、新注册的用户ID）是否值得被加入模型的Embedding词典中的决策过程。 |
| 特征淘汰  | 在模型持续训练或定期调度中，将已经存在于Embedding表中但已“失效”的特征（如过季商品ID）删除，以释放存储和计算资源的机制。 |
| FBGEMM    | FBGEMM (Facebook GEneral Matrix Multiplication) 是一个由Meta（原Facebook）开源的高性能低精度数值计算库。 |
| HKV       | HKV（指开源HierarchicalKV）是一个由 NVIDIA 设计并开源的高性能 GPU 哈希表库。 |
