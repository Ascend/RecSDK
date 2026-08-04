# 推荐领域核心三方库DynamicEmb完善与支持

**状态 (Status):** Approved

**作者 (Authors):** @[pluto1314](https://gitcode.com/pluto1314) @[Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK)

**创建日期 (Created):** 2026-05-07

**更新日期 (Updated):** 2026-05-07

**相关 Issue/PR:** [#1111](https://gitcode.com/Ascend/RecSDK/issues/1111)

---

# 1. 概述

## 1.1 简介

本提案聚焦于推荐领域核心三方库 [DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/v25.09) 的完善与支持，核心目标是解决当前 DynamicEmb 库在功能完整性、性能表现、生态适配等方面的不足，提升其在推荐场景下的可用性和高效性，为 [Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK) 生态下 [torch_rec_v2](https://gitcode.com/Ascend/RecSDK/tree/develop/training/torch_rec_v2)  训练框架的推荐算法开发提供更稳定、高效的动态嵌入层能力支撑，降低开发者集成和使用成本。

## 1.2 动机

**使用场景 / 用例**

在电商、短视频、资讯等推荐业务场景中，动态嵌入层（DynamicEmb）是处理海量稀疏特征的核心组件，需支撑千万级以上特征维度的动态更新、低延迟查询、高并发访问等诉求。典型用户案例：某头部电商推荐团队基于 RecSDK 开发个性化推荐系统时，发现 DynamicEmb 存在特征更新延迟高、内存占用失控、与主流训练框架适配性差等问题，导致线上推荐模型效果波动、资源成本上升。

**当前痛点：**[Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK) 生态下的 [torch_rec_v2](https://gitcode.com/Ascend/RecSDK/tree/develop/training/torch_rec_v2) 当前版本的 API 与社区 [DynamicEmb v25.09](https://github.com/NVIDIA/recsys-examples/tree/v25.09) 版本API存在差异，需要完善API支持度。

**必要性与用户价值**

- 必要性：DynamicEmb 作为 RecSDK 推荐领域的核心依赖，其易用性已成为业务落地的关键瓶颈，若不解决将导致 RecSDK 在推荐场景的竞争力下降；
- 用户价值：完善后可降低推荐系统开发成本，提升特征处理效率，适配更多主流生态，同时降低资源占用，提升业务稳定性；

**不做此提案的影响**

若不推进本次完善工作，RecSDK 将无法满足主流推荐业务的使用诉求，开发者可能转向其他开源推荐框架，导致 RecSDK 生态用户流失，同时现有用户的业务问题无法得到解决，影响口碑。

## 1.3 目标

**目标**

当前版本 [torch_rec_v2](https://gitcode.com/Ascend/RecSDK/tree/develop/training/torch_rec_v2) 的 API 与社区 [DynamicEmb v25.09](https://github.com/NVIDIA/recsys-examples/tree/v25.09) 版本API存在差异，需对 DynamicEmb v25.09 版本 API 100%支持。

**非目标**

- 框架侧仅适配功能，无性能要求；算子侧性能挑战2xL20。
- 当前仅适配`v25.09`版本，其余版本暂未适配。

# 2. 用例分析

本提案使用[社区测试用例](https://github.com/NVIDIA/recsys-examples/tree/v25.09)作为场景用例：

- 功能点：在昇腾平台上兼容支持 DynamicEmb API，并基于社区测试用例进行功能验收。
- 关键性能指标：框架侧仅适配功能，无性能要求；算子侧性能挑战2xL20。
- 安全隐私要求：无敏感数据处理，无需额外安全隐私管控。
- DFX 要求：
  - 兼容性：支持 DynamicEmb v25.09 版本。
  - 可维护性：遵循 RecSDK 编码规范，接口注释覆盖率 100%，核心代码单元测试覆盖率≥80%。
  - 可测试性：基于社区测试用例进行测试，并对高优接口进行自动化测试。
  - 可靠性：涉及 API 在昇腾平台运行时功能与竞品一致。
- 使用限制 / 约束：当前仅适配`v25.09`版本，其余版本暂未适配。

# 3. 方案设计

## 3.1 总体方案

### 3.1.1 整体设计思路

[Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK) 生态下的 [torch_rec_v2](https://gitcode.com/Ascend/RecSDK/tree/develop/training/torch_rec_v2) 基于PyTorch、[DynamicEmb](https://github.com/NVIDIA/recsys-examples/tree/v25.09)、[TorchRec](https://github.com/meta-pytorch/torchrec/tree/release/v1.2.0) 开源软件，是适配 NPU 设备的全下沉稀疏推荐框架。其中 DynamicEmb 作为核心框架，对外提供用户 API，内部实现稀疏相关算子并扩展 HKV 作为稀疏表存储层。本提案 DynamicEmb 能力完善与支持的核心设计采用**框架适配层 + 稀疏算子适配层 + HKV 适配层**的三层分层架构，旨在实现推荐领域 DynamicEmb 能力与 Ascend 平台的深度适配，兼顾生态兼容性、算子性能与存储效率。

- **框架适配层**：对齐 DynamicEmb v25.09 原生 Python API，负责参数解析、Shard 规划、Dump/Load、Score 管理、增量导出等流程调度。
- **稀疏算子适配层**：负责 DynamicEmb 核心稀疏计算算子（Lookup/Update/Grad/Evict 等）在 NPU 上的等效实现与性能优化。
- **HKV 适配层**：对接昇腾 [HierarchicalKV-ascend](https://gitcode.com/Ascend/HierarchicalKV-ascend)（下称HKV） 存储，提供大容量、高性能的动态Embedding表的增删改查能力。

### 3.1.2 系统架构

整体架构采用分层设计，从上到下分为业务层、适配层、运行时层和硬件层：

**架构分层说明：**

1. 框架适配层：

   定位：与 DynamicEmb v25.09 完全对齐，提供 100% 兼容 API，上层业务**零修改迁移**。

   核心能力：

   - 完整支持：`DynamicEmbParameterConstraints` / `CheckMode` / `InitializerMode/Args`
   - 完整支持：`DynamicEmbeddingEnumerator` / `ShardingPlanner` / `CollectionSharder`
   - 新增支持：`DynamicEmbeddingBagCollectionSharder` / `ConstructTwinModule`
   - 新增支持：`incremental_dump` / `set_score` / `get_score` 全流程
   - 新增支持：`Optimizer` 适配（SGD/Adam/AdaGrad/RowWiseAdaGrad）

2. 稀疏算子适配层

   定位：NPU 侧稀疏计算核心，实现 DynamicEmb 所有底层算子等价功能。

   核心能力：

   - `BatchedDynamicEmbeddingTablesV2` 完整适配
   - Embedding 查找、更新、梯度、缓存、打分、淘汰
   - 对接 HKV 完成键值读写、score 下发

3. HKV 适配层

   定位：提供 DynamicEmb 所依赖的动态键值存储、缓存、外部存储、打分策略、增量导出等能力。

   核心能力：

   - 提供大容量、高性能的动态Embedding表的增删改查能力
   - 支持分级存储、支持可定制的淘汰策略
   - keys和values存储分离，keys仅存储于HBM

## 3.2 技术选型

| 对比项         | **方案 A：保持原生架构（三层分层，最终选定）**               | **方案 B：全量重构合并（一体化单模块）**                     |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **架构思路**   | 严格对齐社区 DynamicEmb v25.09 原生结构，保持**框架适配层 + 稀疏算子适配层 + HKV 适配层**三层边界不变，仅做 NPU 能力注入与接口补齐。 | 打破原有分层，将 API 解析、稀疏计算、存储管理、分布式逻辑全部重写，合并为一个自研一体化动态嵌入模块。 |
| **API 兼容性** | **100% 兼容**，上层业务代码无需修改，直接迁移。              | 上层接口兼容，但底层接口不一致。                             |
| **功能一致性** | 与社区行为完全一致，分布式规划、Shard 策略、Dump/Load、Score 策略完全对齐。 | 行为需重新实现，易出现精度、流程、边界条件不一致。           |
| **迭代成本**   | 低：社区升级时，只需同步适配新增接口，不改动架构。           | 高：社区每升级一次，都要重新对齐、重新自研，长期维护成本极高。 |
| **开发风险**   | 低：只补全缺失 API、适配 NPU 算子、对接 HKV，不破坏原有流程。 | 高：全链路重写，涉及分布式、存储、缓存、增量导出、优化器等复杂逻辑，极易出 Bug。 |
| **上线周期**   | 短：可在 2026-06-30 前完成 100% API 支持目标。               | 长：至少延期 6 个月以上，无法按期交付。                      |
| **可维护性**   | 高：模块职责清晰，符合社区习惯，便于多人协作与问题定位。     | 低：自研耦合重，无社区参考，问题定位困难，新人上手成本极高。 |
| **性能收益**   | 满足业务需求：NPU 算子 + HKV 存储已能提供足够加速。          | 理论上可能有小幅收益，但**收益远低于风险**。                 |
| **核心结论**   | **符合 RFC 目标、兼容、可控、可落地、可长期维护。**          | **破坏性大、成本高、风险不可控、不兼容业务、违背生态初衷。** |

## 3.3 功能与性能设计

DynamicEmb API 完善与支持主要分为**框架适配层 + 稀疏算子适配层 + HKV 适配层**的适配支持，本提案仅针对框架适配层和稀疏算子适配层进行设计与开发，而 HKV 则由其他提案进行说明。

### 3.3.1 框架侧API适配模块

**目标**：`torch_rec_v2`框架侧 DynamicEmb API 接口在 NPU 上适配，实现100% API 支持度。

截止`RecSDK 26.0.0`版本，对比原生[API 列表](https://github.com/NVIDIA/recsys-examples/blob/v25.09/corelib/dynamicemb/DynamicEmb_APIs.md)，`torch_rec_v2`支持度如下：

| class                             | API                                                          | 是否支持 |
| --------------------------------- | ------------------------------------------------------------ | -------- |
| DynamicEmbParameterConstraints    | use_dynamicemb                                               | 支持     |
|                                   | dynamicemb_options                                           | 支持     |
| DynamicEmbeddingEnumerator        | def enumerate(<br />self,<br />module: nn.Module,<br />sharders: List[ModuleSharder[nn.Module]],<br />) -> List[ShardingOption] | 支持     |
| DynamicEmbeddingShardingPlanner   | def collective_plan(self, <br />module: nn.Module,<br />sharders: List[ModuleSharder[nn.Module]],<br />pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,<br />) -> ShardingPlan: | 支持     |
| DynamicEmbeddingCollectionSharder | def shard(self,<br />module: EmbeddingCollection,<br />params: Dict[str, ParameterSharding],<br />env: ShardingEnv,<br />device: Optional[torch.device] = None,<br />module_fqn: Optional[str] = None,<br />) -> ShardedEmbeddingCollection: | 支持     |
| DynamicEmbCheckMode               | ERROR                                                        | 支持     |
|                                   | WARNING                                                      | 支持     |
|                                   | IGNORE                                                       | 支持     |
| DynamicEmbInitializerMode         | NORMAL                                                       | 支持     |
|                                   | TRUNCATED_NORMAL                                             | 支持     |
|                                   | UNIFORM                                                      | 支持     |
|                                   | CONSTANT                                                     | 支持     |
|                                   | DEBUG                                                        | 支持     |
| DynamicEmbInitializerArgs         | mode                                                         | 支持     |
|                                   | mean                                                         | 支持     |
|                                   | std_dev                                                      | 支持     |
|                                   | lower                                                        | 支持     |
|                                   | upper                                                        | 支持     |
|                                   | value                                                        | 支持     |
| DynamicEmbScoreStrategy           | TIMESTAMP                                                    | 不支持   |
|                                   | STEP                                                         | 不支持   |
|                                   | CUSTOMIZED                                                   | 不支持   |
|                                   | LFU                                                          | 不支持   |
| DynamicEmbTableOptions            | training                                                     | 支持     |
|                                   | initializer_args                                             | 支持     |
|                                   | eval_initializer_args                                        | 支持     |
|                                   | caching                                                      | 不支持   |
|                                   | init_capacity                                                | 支持     |
|                                   | max_load_factor                                              | 支持     |
|                                   | score_strategy                                               | 不支持   |
|                                   | bucket_capacity                                              | 支持     |
|                                   | safe_check_mode                                              | 支持     |
|                                   | global_hbm_for_values                                        | 支持     |
|                                   | external_storage                                             | 不支持   |
|                                   | index_type                                                   | 支持     |
| DynamicEmbDump                    | path                                                         | 支持     |
|                                   | model                                                        | 支持     |
|                                   | table_names                                                  | 支持     |
|                                   | optim                                                        | 支持     |
|                                   | pg                                                           | 支持     |
| DynamicEmbLoad                    | path                                                         | 支持     |
|                                   | model                                                        | 支持     |
|                                   | table_names                                                  | 支持     |
|                                   | optim                                                        | 支持     |
|                                   | pg                                                           | 支持     |
| incremental_dump                  | /                                                            | 不支持   |
| get_score                         | /                                                            | 不支持   |
| set_score                         | /                                                            | 不支持   |

除以上公开 API 外，还有`DynamicEmbeddingBagCollectionSharder`、`ConstructTwinModule`、`Optimizer`相关未公开 API 暂未支持。

本提案主要针对上述暂未支持的 API 进行完善和支持。

### 3.3.2 算子适配模块

**目标**：将 DynamicEmb 框架侧依赖的稀疏算子在 NPU 上实现等效功能。

涉及稀疏算子参考如下：

| 算子名                                  | 功能描述                          |
| --------------------------------------- | --------------------------------- |
| dynamic_emb_adam_fused                  | FUSED模式Adam优化器计算           |
| dynamic_emb_RowWiseAdaGrad_with_pointer | RowWiseAdaGrad优化器计算          |
| dynamic_emb_sgd_with_table              | 表级别SGD优化器计算               |
| dynamic_emb_AdaGrad_with_pointer        | AdaGrad优化器计算                 |
| dynamic_emb_sgd_fused                   | FUSED模式SGD优化器计算            |
| dynamic_emb_AdaGrad_fused               | FUSED模式AdaGrad优化器计算        |
| dynamic_emb_RowWiseAdaGrad_with_table   | 表级别RowWiseAdaGrad优化器计算    |
| dynamic_emb_RowWiseAdaGrad_with_pointer | FUSED模式RowWiseAdaGrad优化器计算 |
| dynamic_emb_AdaGrad_with_table          | 表级别AdaGrad优化器计算           |
| dynamic_emb_sgd_with_pointer            | SGD优化器计算                     |
| dynamic_emb_adam_with_table             | 表级别Adam优化器计算              |
| lookup_forward                          | Bag模式查表前向计算               |
| select_index                            | 基于条件标志的索引筛选            |
| select                                  | 基于条件标志的值筛选              |
| lookup_backward                         | Bag模式梯度规约                   |

### 3.3.3 验收标准

1. 验收方式：基于社区测试用例进行验证。
2. 性能要求：框架侧仅适配功能，无性能要求；算子侧性能挑战2xL20。
3. 验收硬件：950DT or 950PR。

## 3.4 安全隐私与DFX设计

### 3.4.1 安全隐私设计

本提案涉及的 API 无敏感数据处理逻辑，无需额外的安全隐私管控措施，仅需遵循 RecSDK 现有安全规范，确保接口无越权访问、数据泄露风险。

### 3.4.2 DFX 设计

#### 3.4.2.1 兼容性

1. 版本兼容：支持 DynamicEmb v25.09 版本，其余版本暂未适配。
2. 接口兼容：保证与 DynamicEmb 原生接口 1:1 兼容，开发者无需修改业务代码即可迁移。

#### 3.4.2.2 可维护性

1. 代码规范：遵循 RecSDK 编码规范，接口注释覆盖率 100%，核心代码单元测试覆盖率≥80%。
2. 模块化设计：将接口适配层、算子适配层解耦，便于独立迭代和维护。

#### 3.4.2.3 可测试性

1. 自动化测试：开发单元测试、集成测试脚本，接入 RecSDK 持续集成（CI）流程，每次代码提交自动执行。
2. 测试用例：涉及 API 社区测试用例100%支持。

#### 3.4.2.4 可靠性

1. 涉及 API 在昇腾平台运行时功能与竞品一致。

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

**开发环境设计**：

| 类别     | 具体内容                                  |
| -------- | ----------------------------------------- |
| 硬件环境 | 950DT or 950PR                            |
| 软件环境 | Torch 2.7.1，TorchRec 1.2.0，FBGEMM 1.2.0 |

**开发约束**：参考"开发环境设计"。

**可验收设计**：

- 基于社区测试用例进行功能验收
- 框架侧仅适配功能，无性能要求
- 算子侧性能挑战2xL20

### 3.5.2 接口定义与设计

对齐 [DynamicEmb 原生 API](https://github.com/NVIDIA/recsys-examples/blob/v25.09/corelib/dynamicemb/DynamicEmb_APIs.md)，不新增自定义接口，仅做 NPU 兼容实现，**接口与原生 API 完全一致**。

但根据不同应用场景，使用方法有些不同：

1. 已使用原生 DynamicEmb 进行开发

   此场景下，**仅需对`import`调用有些许修改**，示例如下：

   ```python
   # 典型调用（接口名不变，import包从dynamicemb变为dynamic_emb）
   from dynamic_emb import (
       DynamicEmbInitializerArgs,
       DynamicEmbInitializerMode,
       DynamicEmbTableOptions,
       DynamicEmbParameterConstraints,
   )
   from fbgemm_gpu.split_embedding_configs import SparseType

   # 接口使用处无需修改，直接运行在NPU
   const = DynamicEmbParameterConstraints(
       sharding_types=[ShardingType.ROW_WISE.value],
       compute_kernels=["fused"],
       dynamicemb_options=DynamicEmbTableOptions(
           initializer_args=DynamicEmbInitializerArgs(
               mode=DynamicEmbInitializerMode.NORMAL
           ),
           training=True,
       ),
   )
   ```

2. 未使用原生 DynamicEmb 进行开发，使用的原生 TorchRec 进行开发

   此场景下，**需将 TorchRec 接口改为 DynamicEmb 接口调用**，示例如下：

   ```python
   # 原生 TorchRec 开发示例
   from torchrec.distributed.planner.types import ParameterConstraints
   from fbgemm_gpu.split_embedding_configs import SparseType

   # 接口使用
   const = ParameterConstraints(
       sharding_types=[ShardingType.ROW_WISE.value],
       compute_kernels=["fused"],
   )

   #############################################################
   # TorchRec 接口改为 DynamicEmb 接口调用示例
   #############################################################

   # 典型调用（将 TorchRec 接口改为 DynamicEmb 接口调用）
   from dynamic_emb import DynamicEmbParameterConstraints
   from fbgemm_gpu.split_embedding_configs import SparseType

   # 接口使用处修改
   const = DynamicEmbParameterConstraints(
       sharding_types=[ShardingType.ROW_WISE.value],
       compute_kernels=["fused"],
   )
   ```

   **注：**此修改为将 TorchRec 接口改为 DynamicEmb 接口调用，原生 DynamicEmb 迁移时也是如此。若已使用原生 DynamicEmb 进行开发，在迁移至昇腾设备时仅需修改导包名称即可。

### 3.5.3 编程手册设计

- 章节参考：安装部署、快速上手、API 清单、FAQ、迁移指南等。
- 形式：随 RecSDK 社区发布，Markdown + 示例代码。

# 4. 缺点和风险

| 风险类型             | 具体描述                                                     | 应对措施                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Breaking Change 风险 | DynamicEmb 接口适配层的实现可能引入兼容性问题，导致现有基于 DynamicEmb API 开发的推荐模型代码执行报错，例如动态嵌入表的创建、更新、查询逻辑与原生版本不一致引发的业务代码异常 | 1. 提供版本兼容模式，保留旧版 DynamicEmb API 调用逻辑的兼容分支，支持存量代码无修改运行；2. 发布前基于主流推荐模型（如 DLRM、DCNV2、OneRec）进行全量兼容性回归测试；3. 输出详细的接口变更清单和适配指南，明确差异点及修改方法；4. 提供兼容性检测工具，自动扫描用户代码中潜在的兼容问题并给出修复建议 |
| 性能回退风险         | 动态嵌入特性（如动态扩容、稀疏更新、缓存策略）在昇腾 NPU 上的算子适配可能导致部分边缘场景（如高并发动态插入、低频特征查询）出现性能回退，无法充分发挥昇腾硬件算力优势 | 1. 建立 DynamicEmb 性能基准线，覆盖动态嵌入的核心场景（插入、查询、更新、淘汰），每次代码提交自动执行性能对比测试；2. 为边缘场景提供性能开关，允许用户根据业务需求关闭 / 开启特定优化逻辑；3. 定期复盘性能数据，针对性能回退点优化算子实现（如动态嵌入表的内存布局、NPU 访存策略）；4. 优先保障核心场景（如 CTR 预估中的动态特征查询）性能与原生版本持平 |
| 复杂度提升风险       | DynamicEmb 涉及动态嵌入表管理、稀疏算子适配、分布式动态同步等复杂逻辑，分层架构（框架适配层 + 算子适配层）会增加代码复杂度，提升后续维护、迭代成本，且模块间耦合可能导致问题定位困难 | 1. 严格遵循 RecSDK 模块化设计规范，将动态嵌入表管理、算子适配、分布式同步等模块解耦，明确模块边界和接口；2. 完善代码注释（注释覆盖率 100%）和文档，核心模块提供设计文档、流程图和关键逻辑说明；3. 核心代码单元测试覆盖率≥80%，针对复杂逻辑（如动态淘汰策略、分布式一致性）增加专项测试用例；4. 定期进行代码重构，简化冗余逻辑，降低维护成本；5. 建立模块责任人制度，明确各子模块的维护主体和迭代规范 |
| 安全风险             | DynamicEmb 涉及动态嵌入数据的内存操作、分布式通信传输，若实现不当可能存在内存越界、数据传输泄露、越权访问嵌入数据等风险 | 1. 遵循 RecSDK 安全规范，对动态嵌入表的内存操作增加边界校验，防止越界访问；2. 分布式通信环节采用昇腾 HCCS/RDMA 原生安全机制，保障数据传输加密和身份校验；3. 对嵌入数据的读写操作增加权限校验，避免越权访问；4. 引入内存检测工具，扫描动态内存分配 / 释放环节的潜在漏洞（如内存泄漏、野指针） |
| 负面影响             | 1. DynamicEmb 接口和功能变更可能增加用户迁移成本，尤其是存量项目从原生 DynamicEmb 迁移至昇腾适配版本时，需调整动态嵌入的配置、调用逻辑；2. 动态嵌入算子适配依赖高版本 CANN 提供的稀疏计算、动态内存管理能力，用户需升级 CANN 环境才能使用新功能，增加环境适配成本 | 1. 开发自动化代码转换工具，自动将原生 DynamicEmb 代码转换为昇腾适配版本，降低手动修改成本；2. 兼容主流 CANN 版本（如 7.0、8.0），为低版本 CANN 提供核心功能降级适配方案（如关闭部分高级动态特性，保障基础功能可用）；3. 提前发布变更公告，明确版本依赖和适配周期，提供环境升级指南和问题排查手册；4. 提供线下 / 线上技术支持，协助企业用户完成环境升级和代码迁移 |
| 实现成本             | DynamicEmb 完善涉及动态嵌入表核心逻辑适配、稀疏动态算子开发、分布式动态同步机制实现等，工作量预计 12.5 人月，整体投入较大，且需同步跟进 DynamicEmb 社区版本迭代，进一步增加适配成本 | 1. 分阶段实现：优先完成核心接口（动态嵌入表创建、查询、更新）和核心场景（单卡动态嵌入训练）的适配，后续迭代分布式同步、高级淘汰策略等功能；2. 建立自动化测试和维护流程，减少重复人工测试成本，核心接口接入 RecSDK CI/CD 流水线；3. 与 DynamicEmb 社区保持同步，提前规划版本适配路线，减少社区版本迭代带来的重复适配工作；4. 复用 RecSDK 现有组件（如分布式通信模块、内存管理模块），降低重复开发成本 |
| API / 版本兼容性     | 1. DynamicEmb 社区新版本（如 1.x → 2.x）可能调整核心接口（如动态嵌入表的初始化参数、淘汰策略枚举值），导致昇腾适配版本需频繁迭代，且存量用户无法直接使用新功能；2. RecSDK 旧版本用户无法直接使用 DynamicEmb 新适配功能，需升级 RecSDK 整体版本 | 1. 建立 DynamicEmb 版本适配清单，明确昇腾版本支持的社区版本范围（如 v1.0、v1.5、v2.0），并标注各版本的功能支持度；2. 为 RecSDK 旧版本提供 DynamicEmb 兼容补丁，无需全量升级 RecSDK 即可使用核心动态嵌入功能；3. 开发 RecSDK 版本升级工具，自动化处理版本升级过程中 DynamicEmb 相关的依赖和配置变更；4. 保留旧版 API 兼容层，支持用户在不修改业务代码的前提下升级至新版 RecSDK |
| 旧版本迁移方案       | 现有用户从旧版 DynamicEmbAPI（昇腾适配早期版本）迁移至新版时，需修改动态嵌入表的配置、缓存策略调用方式等代码，迁移成本较高且易引入业务逻辑错误 | 1. 开发旧版→新版代码迁移工具，自动识别旧版 API 调用并转换为新版格式，生成迁移报告；2. 输出详细的迁移指南，包含核心场景的迁移示例（如动态嵌入表初始化、特征插入、淘汰策略配置）、常见问题及解决方案；3. 提供迁移专项技术支持，针对企业级用户的定制化场景提供一对一迁移指导；4. 保留旧版 API 兼容层至少 2 个迭代周期，给用户充足的迁移时间 |
| 动态特性可靠性风险   | DynamicEmb 的核心动态特性（如动态扩容、自动淘汰、分布式动态同步）在昇腾平台上可能出现数据一致性问题（如分布式场景下嵌入值同步延迟）、内存泄漏（动态表未及时释放）、缓存击穿（淘汰策略失效）等可靠性问题 | 1. 基于主流动态嵌入业务场景设计长稳测试用例，持续运行 7*24 小时验证功能稳定性；2. 增加动态嵌入表的监控能力，实时监测内存使用、数据同步状态、缓存命中率等指标，异常时触发告警；3. 针对分布式动态同步场景，实现数据一致性校验机制，定期核对各节点嵌入表数据；4. 完善异常处理逻辑，对动态扩容失败、内存不足、通信异常等场景增加容错和降级策略 |

# 5. 现有技术

*参考其他项目/社区的类似设计，说明借鉴与差异。*

参考项目 / 社区设计：

**[NVIDIA DynamicEmb](https://github.com/NVIDIA/recsys-examples/blob/v25.09)**：原生 DynamicEmb 框架，提供分布式动态嵌入存储框架，支持动态特征管理、score 策略、external_storage、incremental_dump 等核心能力。本提案对齐其接口定义，但底层实现替换为昇腾优化算子。

借鉴与差异：

| 参考项目          | 借鉴点                                                       | 差异点                                                       |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| NVIDIA DynamicEmb | 1. API 定义、参数、枚举完全对齐<br />2. Score 策略（TIMESTAMP/STEP/LFU/CUSTOMIZED）<br />3. incremental_dump 增量导出流程<br />4. external_storage 插件化架构<br />5. Dump/Load 持久化格式 | 1. 底层存储替换为昇腾 HKV 实现<br />2. 稀疏算子与动态查表逻辑在 NPU 实现<br />3. 支持昇腾 NPU 设备管理与拓扑 |

# 6. 未解决问题

暂无

---

附录

**参考资料链接：**

1. DynamicEmb 社区官方说明：[https://github.com/NVIDIA/recsys-examples/blob/v25.09](https://github.com/NVIDIA/recsys-examples/blob/v25.09)
2. RecSDK 开源仓库：[https://gitcode.com/Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK)
3. 昇腾 CANN 开发文档：[https://www.hiascend.com/document/](https://www.hiascend.com/document/)
4. PyTorch TorchRec 分布式框架：[https://pytorch.org/torchrec/](https://pytorch.org/torchrec/)
5. Issue #1111：[https://gitcode.com/Ascend/RecSDK/issues/1111](https://gitcode.com/Ascend/RecSDK/issues/1111)

**术语表：**

| 术语     | 定义                                                         |
| -------- | ------------------------------------------------------------ |
| TorchRec | PyTorch 官方推出的推荐系统专用框架，提供了丰富的推荐模型开发接口和算子 |
| FBGEMM   | FBGEMM 在 CPU 与 GPU 上提供优化的稀疏算子，支持嵌入查找、锯齿张量操作及量化推理等场景 |
| RecSDK   | 昇腾推荐领域软件开发工具包，为推荐模型提供昇腾硬件适配能力   |
| CANN     | 昇腾异构计算架构，提供昇腾硬件的算子库和编程接口             |
| HKV      | [HierarchicalKV-ascend](https://gitcode.com/Ascend/HierarchicalKV-ascend) 是 [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV/commit/bbe2ee1858b6e54bccf9106e9f3c2d8c1c5d248c) 在昇腾 NPU 平台上的算子实现，是一个面向推荐系统的高性能key-value存储加速库 |

**文档更新计划**：暂无
