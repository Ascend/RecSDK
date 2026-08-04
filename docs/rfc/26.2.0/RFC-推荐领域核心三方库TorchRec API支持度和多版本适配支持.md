# 推荐领域核心三方库TorchRec API支持度和多版本适配支持

**状态 (Status):** Approved

**作者 (Authors):** @[pluto1314](https://gitcode.com/pluto1314) @[Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK)

**创建日期 (Created):** 2026-07-14

**更新日期 (Updated):** 2026-07-15

**相关 Issue/PR:** [#1262](https://gitcode.com/Ascend/RecSDK/issues/1262), [#1112](https://gitcode.com/Ascend/RecSDK/issues/1112)

---

# 1. 概述

## 1.1 简介

本提案旨在完善 Ascend RecSDK 中 TorchRec API 的功能支持，实现 TorchRec 1.2.0/1.5.0 100% API 支持度；并在上一版本（26.1.0，基于 TorchRec 1.2.0/1.5.0 实现 80% API 支持度）的基础上，将存量功能迁移适配至 TorchRec 1.6.0 / 1.7.0 双版本，解决 TorchRec API 在昇腾硬件适配、版本演进、功能完整性等方面的问题，为推荐领域开发者提供高效、稳定、贴合昇腾平台特性的 TorchRec 兼容 API，降低推荐模型在昇腾平台的迁移和开发成本。

## 1.2 动机

**使用场景 / 用例**

推荐领域大量模型开发依赖 TorchRec 生态，开发者在将基于 TorchRec 的推荐模型迁移至昇腾平台时，需通过 RecSDK 的 TorchRecAPI 完成适配。典型场景包括 CTR 预估、多目标推荐、个性化推荐等模型的训练与推理部署。

**当前痛点**

- 现有 TorchRec API 部分接口缺失，无法覆盖主流推荐模型的开发需求，开发者需自行适配底层接口，开发效率低。
- API 和 底层 FBGEMM 算子兼容性不足，部分 TorchRec 原生用法在昇腾平台下不兼容、算子缺失，无明确的异常处理和兼容方案。
- 版本演进滞后：TorchRec 社区持续发布新版本（1.6.0、1.7.0），旧版本适配的 API 无法覆盖新版本生态，开发者若需使用新版 TorchRec 则无法直接基于现有适配进行开发，需等待版本适配周期。
- 缺乏完善的编程手册和调用示例，开发者上手成本高。

**必要性与用户价值**

- 若不完善 TorchRec API，将导致基于 TorchRec 的推荐模型难以高效迁移至昇腾平台，无法充分利用昇腾硬件的算力优势，同时降低开发者使用意愿，影响 RecSDK 在推荐领域的生态落地。完善后可大幅提升模型迁移效率（预计降低 50% 适配成本），并降低开发者学习和使用成本。
- 若不进行多版本适配，TorchRec 1.5.0 存量功能的适配成果将锁定在单一版本，无法随社区版本演进持续服务用户，开发者被迫锁定在旧版本依赖上，影响生态扩展。完成多版本适配后，开发者可自由选择 TorchRec 1.6.0 或 1.7.0 版本进行开发，享受社区新版本带来的生态兼容性和稳定性提升。

**不做此提案的影响**

昇腾平台在推荐领域的 TorchRec 生态适配能力持续落后，无法满足企业级用户的规模化迁移需求，丧失推荐领域市场竞争力。存量适配成果将停留在 TorchRec 1.5.0 版本，无法跟随社区版本演进。

## 1.3 目标

**目标**

- 基于v1.2.0/v1.5.0版本，完成[官网API](https://meta-pytorch.org/torchrec/api.html)提供特性的100%支持，支持对外API涉及测试用例跑通。
- 存量功能（基于26.1.0 80%支持度版本）在 TorchRec 1.6.0 / 1.7.0 中功能正常，相关社区测试用例通过。

**非目标**

- 当前仅做功能适配，不涉及性能优化。
- TorchRec 1.6.0 / 1.7.0 版本新增功能暂不做适配，仅迁移存量功能。
- `SSDTableBatchedEmbeddingBags`由于开源版本 FBGEMM 暂不支持（见https://github.com/pytorch/FBGEMM/issues/5666），因此本 RFC 低优支持该特性。

**约束说明**

- 当前版本不支持SSD和UVM场景。
- 当前版本split/dense查表算子仅支持部分场景（weighted/unweighted+codegen+SGD/Adagrad/Adam/RowwiseAdagrad优化器）。
- 当前torch_npu2.12.0/2.11.0在社区为beta版本发布，若上述存量功能（基于26.1.0 80%支持度版本）存在功能缺失，则后续在其正式发布版本中进行功能补充，当前阶段以beta版本进行适配。

# 2. 用例分析

本提案使用社区测试用例作为场景用例：

- 功能点：在昇腾平台上兼容支持 TorchRec API，并基于社区测试用例进行功能验收。
- 关键性能指标：功能适配，无性能要求。
- 安全隐私要求：无敏感数据处理，无需额外安全隐私管控。
- DFX 要求：
  - 兼容性：支持 TorchRec 1.5.0 存量功能迁移至 TorchRec 1.6.0 / 1.7.0 版本。
  - 可维护性：遵循 RecSDK 编码规范，接口注释覆盖率 100%，核心代码单元测试覆盖率≥80%。
  - 可测试性：基于社区测试用例进行测试，并对高优接口进行自动化测试。
  - 可靠性：涉及 API 在昇腾平台运行时功能与竞品一致。
- 使用限制 / 约束：
  - 当前版本不支持SSD和UVM场景。
  - 当前版本split/dense查表算子仅支持部分场景（weighted/unweighted+codegen+SGD/Adagrad/Adam/RowwiseAdagrad优化器）。
  - 当前torch_npu2.12.0/2.11.0在社区为beta版本发布，若上述存量功能（基于26.1.0 80%支持度版本）存在功能缺失，则后续在其正式发布版本中进行功能补充，当前阶段以beta版本进行适配。

# 3. 方案设计

## 3.1 总体方案

### 3.1.1 整体设计思路

基于昇腾平台，采用 “框架适配层 + 算子适配层” 分层架构完善 TorchRec API：

1. 框架适配层：对齐 TorchRec 原生 API 定义，补齐缺失接口，封装昇腾硬件底层调用逻辑，对外提供与原生 TorchRec 一致的调用方式。
2. 算子适配层：针对 TorchRec 依赖的 FBGEMM 库中的核心算子在 NPU 上实现等效功能。

### 3.1.2 系统架构

整体架构采用分层设计，从上到下分为业务层、适配层、核心加速层和硬件层：

![image-20260601200058604.png](https://raw.gitcode.com/user-images/assets/7379929/98316b83-6500-4f3a-a21e-478cd4086a2e/image-20260601200058604.png 'image-20260601200058604.png')

各层职责说明：

| 层级           | 职责                 | 关键组件                                  |
| :------------- | :------------------- | :---------------------------------------- |
| 业务模型层     | 用户定义的推荐模型   | DLRM、DCNV2、GR等                         |
| TorchRec框架层 | 原生TorchRec核心逻辑 | EmbeddingBagCollection、Sharder、Planner  |
| NPU适配层      | 设计的核心           | 算子适配、Cache管理、流水线适配、通信适配 |
| 底层运行时     | NPU基础能力          | torch_npu、CANN、算子库                   |
| 硬件层         | 昇腾NPU硬件          | HBM、DDR、HCCS互联                        |

针对本提案，核心在于 TorchRec 框架适配与 FBGEMM 算子适配。

## 3.2 技术选型

| 备选方案                                    | 优势                                         | 劣势                                                         | 不选择理由                                             |
| ------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| 方案 1：全量接口 + 全量算子适配             | API 支持度高，完全支持社区公开 API 列表      | 开发成本高（预计人力投入增加 200%），周期长（延期 3 个月 +），兼容性风险高 | 投入产出比低，且易引入兼容性问题，不符合快速落地的需求 |
| 方案 2：核心接口 + 核心算子适配（选定方案） | 1. 开发成本适中<br />2. 核心场景兼容性有保障 | 需针对核心算子逐个优化，部分边缘场景覆盖不及时               | -                                                      |

## 3.3 功能与性能设计

采用 “框架适配 + 算子适配” 完善 TorchRec API：

1. 框架适配：对齐 TorchRec 原生 API 定义，补齐缺失接口，封装昇腾硬件底层调用逻辑，对外提供与原生 TorchRec 一致的调用方式。
2. 算子适配：针对 TorchRec 依赖的 FBGEMM 库中的核心算子在 NPU 上实现等效功能。

### 3.3.1 框架侧API适配模块

**目标**：将 TorchRec 框架侧 API 接口在 NPU 上适配修改，本次实现100%支持度。

根据官网[API 列表](https://meta-pytorch.org/torchrec/api.html)，本提案适配支持的 API 参考如下：

```python
torchrec.distributed.planner.planners.EmbeddingShardingPlanner
torchrec.distributed.model_parallel.DistributedModelParallel
torchrec.inference.modules.shard_quant_model
```

以下 API 在26.1.0版本已适配支持：

```python
torchrec.sparse.jagged_tensor.JaggedTensor
torchrec.sparse.jagged_tensor.KeyedJaggedTensor
torchrec.sparse.jagged_tensor.KeyedTensor
torchrec.modules.embedding_configs.EmbeddingBagConfig
torchrec.modules.embedding_configs.EmbeddingConfig
torchrec.modules.embedding_configs.BaseEmbeddingConfig
torchrec.modules.embedding_modules.EmbeddingBagCollection
torchrec.modules.embedding_modules.EmbeddingCollection
torchrec.distributed.types.ShardingPlan
torchrec.distributed.planner.enumerators.EmbeddingEnumerator
torchrec.distributed.planner.partitioners.GreedyPerfPartitioner
torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
torchrec.distributed.planner.proposers.GreedyProposer
torchrec.distributed.planner.shard_estimators.EmbeddingPerfEstimator
torchrec.distributed.planner.shard_estimators.EmbeddingStorageEstimator
torchrec.inference.modules.quantize_inference_model
```

### 3.3.2 算子适配模块

**目标**：将 TorchRec 依赖的 FBGEMM 库中的核心算子在 NPU 上实现等效功能。

涉及 FBGEMM 算子参考如下：

```python
batch_index_select_dim0
block_bucketize_sparse_features_inference
jagged_index_select_2d_forward_v2
jagged_unique_indices
prune_embedding_tables
remap_indices_update_utils
sum_reduce_to_one
block_bucketize_sparse_features_2d_weights
SplitTableBatchedEmbeddingBagsCodegen
DenseTableBatchedEmbeddingBagsCodegen
lxu_cache_locking_counter_decrement
get_infos_metadata
generate_vbe_metadata
check_feature_gate_key
```

### 3.3.3 验收标准

1. 验收方式：基于社区测试用例进行验证。
2. 性能要求：本提案仅适配功能，无性能要求。
3. 验收硬件：950DT or 950PR。

## 3.4 安全隐私与DFX设计

### 3.4.1 安全隐私设计

本提案涉及的 API 无敏感数据处理逻辑，无需额外的安全隐私管控措施，仅需遵循 RecSDK 现有安全规范，确保接口无越权访问、数据泄露风险。

### 3.4.2 DFX 设计

#### 3.4.2.1 兼容性

1. 版本兼容：支持 TorchRec 1.5.0 存量功能迁移至 TorchRec 1.6.0 / 1.7.0 版本。
2. 接口兼容：保证与 TorchRec 原生接口 1:1 兼容，开发者无需修改业务代码即可迁移。

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

| 类别     | 具体内容                                                     |
| -------- | ------------------------------------------------------------ |
| 硬件环境 | 950DT or 950PR                                               |
| 软件环境 | - Torch 2.11.0，TorchRec 1.6.0，FBGEMM-GPU 1.6.0，torch_npu 2.11.0 (beta)<br />- Torch 2.12.0，TorchRec 1.7.0，FBGEMM-GPU 1.7.0，torch_npu 2.12.0 (beta)<br />注：TorchRec 1.5.0 存量功能在上述环境功能正常 |

**开发约束**：参考"开发环境设计"。

**可验收设计**：基于社区测试用例进行功能验收，无性能要求。

### 3.5.2 接口定义与设计

本次为对齐 [TorchRec 原生 API](https://meta-pytorch.org/torchrec/api.html)，不新增自定义接口，仅做 NPU 兼容实现，**接口形态与原生 API 完全一致**，示例如下：

```python
# 典型调用（上层不变）
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_modules import EmbeddingBagCollection

# 业务代码无需修改，直接运行在NPU
kjt = KeyedJaggedTensor(keys=..., values=..., lengths=...)
ebc = EmbeddingBagCollection(...)
output = ebc(kjt)
```

### 3.5.3 编程手册设计

- 章节参考：安装部署、快速上手、API 清单、FAQ、迁移指南等。
- 形式：随 RecSDK 社区发布，Markdown + 示例代码。

# 4. 缺点和风险

| 风险类型             | 具体描述                                                     | 应对措施                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Breaking Change 风险 | 接口适配层可能引入兼容性问题，导致现有基于 TorchRec API 的代码报错 | 1. 提供版本兼容模式，支持旧版接口调用；<br />2. 发布前进行全量兼容性测试；<br />3. 提供接口迁移指南，明确变更点和适配方法 |
| 性能回退风险         | 部分边缘场景下算子优化可能导致性能回退                       | 1. 建立性能基准线，每次代码提交执行性能测试；<br />2. 为边缘场景提供性能开关，可关闭优化逻辑；<br />3. 定期复盘性能数据，及时修复性能回退问题 |
| 复杂度提升风险       | 分层架构设计增加代码复杂度，维护成本上升                     | 1. 严格遵循模块化设计，降低模块间耦合；<br />2. 完善代码注释和单元测试；<br />3. 定期进行代码重构，简化逻辑 |
| 安全风险             | 无                                                           | 无                                                           |
| 负面影响             | 1. 接口变更可能增加用户迁移成本；<br />2. 算子优化需依赖高版本 CANN，用户需升级环境 | 1. 提供自动化代码转换工具，降低迁移成本；<br />2. 兼容主流 CANN 版本，提供低版本适配方案；<br />3. 提前发布变更公告，告知用户适配周期 |
| 实现成本             | 工作量预计10.41人月，工作量较大                              | 1. 分阶段实现，优先完成核心接口和性能优化；<br />2. 建立自动化测试和维护流程，降低维护成本；<br />3. 与 TorchRec 社区保持同步，提前规划版本适配 |
| API / 版本兼容性     | 1. TorchRec 新版本发布可能导致接口不兼容；<br />2. RecSDK 旧版本用户无法使用新功能 | 1. 建立 TorchRec 版本适配清单，明确支持范围；<br />2. 为 RecSDK 旧版本提供兼容补丁；<br />3. 提供版本升级工具，简化用户升级流程 |
| 旧版本迁移方案       | 现有用户从旧版 TorchRecAPI 迁移至新版需修改代码              | 1. 提供迁移工具，自动转换旧版代码；<br />2. 输出详细的迁移指南，包含示例和常见问题；<br />3. 提供技术支持，协助用户完成迁移 |

# 5. 现有技术

参考项目 / 社区设计：

**[META TorchRec](https://github.com/meta-pytorch/torchrec)**：原生 TorchRec 框架，提供了推荐模型的核心接口和算子，本提案对齐其接口定义，但底层实现替换为昇腾优化算子。

借鉴与差异：

| 参考项目      | 借鉴点                                                       | 差异点                                                       |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| META TorchRec | 1. 接口定义和参数设计；<br />2. 推荐模型的核心数据结构（如 KeyedJaggedTensor）；<br />3. 场景化的用例设计 | 1. 底层算子实现：替换为昇腾硬件优化算子；<br />2. 性能优化策略：针对昇腾 AI Core 架构定制；<br />3. 设备支持：扩展支持昇腾硬件 |

# 6. 未解决问题

暂无

---

附录

**参考资料链接：**

1. TorchRec 官方文档：[https://pytorch.org/torchrec/](https://pytorch.org/torchrec/)
2. RecSDK 开源仓库：[https://gitcode.com/Ascend/RecSDK](https://gitcode.com/Ascend/RecSDK)
3. Issue #1112：[https://gitcode.com/Ascend/RecSDK/issues/1112](https://gitcode.com/Ascend/RecSDK/issues/1112)

**术语表：**

| 术语     | 定义                                                         |
| -------- | ------------------------------------------------------------ |
| TorchRec | PyTorch 官方推出的推荐系统专用框架，提供了丰富的推荐模型开发接口和算子 |
| FBGEMM   | FBGEMM 在 CPU 与 GPU 上提供优化的稀疏算子，支持嵌入查找、锯齿张量操作及量化推理等场景 |
| RecSDK   | 昇腾推荐领域软件开发工具包，为推荐模型提供昇腾硬件适配能力   |
| CANN     | 昇腾异构计算架构，提供昇腾硬件的算子库和编程接口             |

**文档更新计划：**暂无
