# 需求背景

当前rec的算子都放在RecSdk仓库，跟推荐模型代码放在一起，所有算子无层次划分（attention算子，通信类算子，优化器算子等杂糅在一起)，并且算子发布后其易用性较差(客户需要手动编译算子适配层，import library)，文档信息零散，因此从开源适配的角度需要进行算子仓库重构和建设。

对比其他社区的仓库比如CANN/ops-transformer，和NV的recsys的建设都包含比较清晰的算子领域划分，较为清晰的算子接口使用说明文档，比较完善的生态适配。

# 提议方案

构建新的ops-rec算子库，参考业界优秀的算子，并鼓励客户/研究机构/大学等积极贡献该算子库，提升客户易用性。

## 仓库顶层目录结构

```text
ops_recsys/
├── thirdPart/              三方库依赖
│   ├── catlass/            Catlass 三方库
│   └── googletest/         UT 测试框架依赖
├── cmake/                  算子构建脚本框架（统一 cmake 模块）
├── docs/                   仓库文档：构建说明 / 算子列表 / 安装部署
├── examples/               每个算子的使用示例（按算子领域划分）
├── ops/                    算子实现
│   ├── ascendC/            Ascend C 算子
│   │   ├── attention/      注意力类算子
│   │   ├── embedding/      嵌入类算子
│   │   ├── sequence/       序列处理类算子
│   │   ├── activation/     激活类算子
│   │   ├── mixing/         混合类算子
│   │   ├── communication/  通信类算子
│   │   ├── optimizer/      优化器类算子
│   │   └── common/         算子公共模块（Log / Utils 等）
│   └── triton/             Triton 算子
├── experimental/           社区贡献算子目录
├── tests/                  测试框架
└── torch_extension/        框架适配层
   ├── torch_plugin/       Torch Eager 模式适配插件
   └── torch_inductor/     Torch Inductor 模式适配插件
```

## 单算子内部模板

每个算子统一遵循以下目录结构：

```text
<op_name>/
├── CMakeLists.txt          该算子的构建脚本
├── op_api/                 Aclnn 接口定义
├── op_host/                算子 Host 侧 Tiling 计算
├── op_kernel/              算子 Kernel 核心逻辑
├── op_graph/               算子 GE 入图插件
├── tests/                  算子测试用例 (UT / ST)
└── README.md               算子说明文档
```

## 领域分层

✅ 算子按推荐领域组织：attention / embedding / sequence / activation / mixing / communication / optimizer
✅ 算子模板统一：每个算子包含 CMakeLists.txt + op_api/ + op_host/ + op_kernel/ + op_graph/ + tests/ + README.md
✅ 芯片命名规范：使用 AtlasA2 / AtlasA3 / Atlas950（对应 A2 / A3 / 950）
✅ 测试完整：包含完整的 ST/UT 测试
✅ 生态适配完整：eager 模式适配 (torch_plugin) + inductor 模式适配 (torch_inductor) + triton 算子适配
✅ 使用案例丰富：每个算子提供场景使用说明，指导客户在不同场景下的使用

## 芯片跨代隔离策略

芯片按代际隔离，支持两种方式灵活选择：

- **文件隔离**：两个芯片的代码完全无法复用时使用，参考 Catlass A2/950 方案
- **宏隔离**：文件中存在部分可复用、部分不可复用的逻辑且无法抽取时使用，参考 CANN 部分算子方案

| 芯片代号 | 芯片名称 |
|---------|---------|
| A2 | Atlas A2 |
| A3 | Atlas A3 |
| 950 | Atlas 950 |

## 算子迁移范围

| 算子名 | 归属领域 | 算子说明 |
|-------|---------|---------|
| Hstu_v1 | Attention | 基于 AscendC 版本，支持 A2/A3/950，后续不再演进 |
| Hstu_v2 | Attention | 基于 Catlass 版本，支持 950 及后续版本 |
| Fa_Infer_with_arbitrary_func | Attention | FA 在生成式推荐领域支持任意 mask |
| In_mul | Activation | — |
| In_linear_silu | Activation | — |
| Token_mixing | Mixing | — |

# 方案目标

①  商业交付诉求
   推荐算子需作为独立产品对外交付 → 独立版本号 / CHANGELOG / 兼容性矩阵 / 发布节奏
②  生态对齐
   NVIDIA recsys-examples & 昇腾 ops-transformer 均采用独立算子仓模式
③  研发效率
   SDK / 训练 / 算子三方耦合 → 改一个算子需全量构建 → CI 时间长 / PR 职责不清
④  专业形象
   独立仓库 → Operator Catalog → API 文档 → 性能 Benchmark → 用户快速上手

# 总体设计

## 构建系统

- 顶层 `cmake/` 目录统一管理所有构建模块，参考 ops-transformer 的 `cmake/` 架构
- 每个算子仅需一个 `CMakeLists.txt` 声明源文件和依赖，无需触碰构建基础设施
- 芯片版本通过 cmake `soc_version` 参数在编译期处理，目录结构不暴露芯片差异

## 测试体系

- **UT (单元测试)**：每个算子的 `tests/` 目录内，覆盖 Kernel 正确性
- **ST (系统测试)**：`tests/` 顶层框架，覆盖端到端算子调用链路
- 测试与算子同目录存放，修改即覆盖

## 生态适配

- **Torch Eager 模式**：通过 `torch_extension/torch_plugin/` 注册 `torch.library` / `aclnn` 接口
- **Torch Inductor 模式**：通过 `torch_extension/torch_inductor/` 提供 Inductor 后端适配
- **Triton 算子**：`ops/triton/` 目录独立维护 Triton DSL 实现

## 社区贡献

- `experimental/` 目录接收社区贡献算子，成熟后迁入 `ops/ascendC/<domain>/`
- 鼓励客户 / 研究机构 / 大学通过 PR 贡献新算子或现有算子优化

## 版本与发布

- 独立语义化版本号 (`MAJOR.MINOR.PATCH`)
- 独立 `CHANGELOG.md`，每版本记录新增 / 变更 / 废弃算子
- 独立 whl 发布：`ops_recsys-x.y.z-cp39-linux_aarch64.whl`

# 验收标准

## 算子文档验收

1. 每个算子必须包含 **算子功能说明**：描述算子作用、适用场景
2. 每个算子必须包含 **产品支持情况**：列出支持的芯片型号和 CANN 版本
3. 每个算子必须包含 **参数说明**：
   - `dtype`、`format`、`shape` 约束
   - 必选 / 可选参数标识
   - 值域范围与约束条件
4. 每个算子必须包含 **aclnn 接口说明**
5. 每个算子必须包含 **torch 接口说明**

## 算子样例验收

1. 每个算子必须补充 **场景使用说明**，覆盖该算子的典型使用场景
2. 每个新增 feature 必须补充对应的 **feature 使用文档**
3. 测试验收时，若无法按照 feature 使用样例跑通并复现代码结果，应作为 **问题单** 处理

## 测试覆盖率验收

- UT 覆盖率 ≥ 80%（核心计算逻辑）
- ST 覆盖每个算子的典型输入 shape 组合（≥ 5 种场景）
- 精度测试：与 PyTorch 原生实现相比误差 ≤ 1e-3 (fp16) / ≤ 1e-5 (fp32)

## 构建验收

- 仓库在目标芯片平台一键构建成功 (`bash build.sh`)
- 构建产物包含完整的 whl 包
- whl 包安装后可 `import ops_recsys` 直接使用（无需手动编译适配层）

# 意见征集周期

截止 2026-07-30

# 抄送名单

yukunQin

# 其他补充说明

欢迎加入社区，感谢您对社区的贡献 🎉!
