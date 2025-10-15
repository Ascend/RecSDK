# NGO - NPU Graph Optimizer

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**NGO (NPU Graph Optimizer)** 是一个专门为昇腾硬件平台设计的图优化框架，与 PyTorch 的 TorchInductor 编译生态系统深度集成，为搜广推（搜索、广告、推荐）业务模型提供极致的性能优化能力。

## ✨ 核心特性

### 🔧 高性能图优化
- **Pass 系统**: 支持死代码消除、常量折叠、公共子表达式消除等经典图优化
- **Pattern 匹配**: 智能识别算子融合机会，如 Add+LayerNorm 融合为单算子
- **自动优化**: 与 `torch.compile` 无缝集成，一键启用所有优化

### 🚀 昇腾硬件加速
- **硬件感知**: 针对昇腾 Atlas 系列硬件特性优化
- **算子融合**: 将多个小算子融合为高性能的昇腾专用算子
- **内存优化**: 减少内存访问开销，提升计算效率

### 🎯 搜广推场景优化
- **模型适配**: 专门针对搜广推业务模型的图结构优化
- **性能提升**: 在典型场景下可获得显著的性能提升
- **精度保持**: 优化过程中保持模型精度不下降

### 🔧 易用性设计
- **零代码修改**: 现有模型无需修改即可使用
- **配置灵活**: 支持细粒度的优化配置
- **可视化监控**: 提供性能指标和优化效果的可视化展示

## 🏗️ 系统架构

NGO 采用分层架构设计，确保各模块的独立性和可扩展性：

```
┌─────────────────────────────────────────────────────────────┐
│                     用户应用层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  搜广推模型  │  │  其他模型    │  │  Benchmark  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   PyTorch 集成层                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           torch.compile(backend="ngo")                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    NGO 后端层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 优化引擎    │  │ 配置管理    │  │ 注册中心    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    核心功能层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Pass 系统   │  │ Pattern系统 │  │ 相似度匹配  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    基础框架层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 基础抽象    │  │ 日志系统    │  │ 工具库      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块

#### 1. 基础框架层
- **基础抽象**: 定义统一的组件接口和生命周期管理
- **日志系统**: 提供结构化的日志记录和性能监控
- **工具库**: 提供通用的辅助功能

#### 2. 核心功能层
- **Pass 系统**: 实现各种图优化算法
  - 死代码消除 (Dead Code Elimination)
  - 常量折叠 (Constant Folding)
  - 公共子表达式消除 (Common Subexpression Elimination)
- **Pattern 系统**: 实现算子模式匹配和替换
  - Add+LayerNorm 融合
  - 其他常见模式识别
- **相似度匹配**: 分析模型结构与典型子结构的相似度

#### 3. NGO 后端层
- **优化引擎**: 协调各个优化组件的执行
- **配置管理**: 提供灵活的配置管理机制
- **注册中心**: 统一管理所有优化组件

#### 4. PyTorch 集成层
- **TorchInductor 集成**: 作为 PyTorch 编译后端
- **API 兼容**: 完全兼容 `torch.compile` 接口

## 📁 代码目录结构

```
ngo/
├── ngo/                          # 主要源代码目录
│   ├── __init__.py              # 项目入口，注册后端
│   ├── core/                    # 核心功能模块
│   │   ├── __init__.py         # 核心模块初始化
│   │   ├── base.py             # 基础抽象类和接口定义
│   │   ├── unified_registry.py # 统一注册中心
│   │   ├── config/             # 配置管理
│   │   │   ├── __init__.py     # 配置模块初始化
│   │   │   ├── manager.py      # 通用配置管理器
│   │   │   └── optimization_config.py  # 优化配置管理
│   │   ├── engine/             # 优化引擎
│   │   │   ├── __init__.py     # 引擎模块初始化
│   │   │   └── engine.py       # 优化引擎核心实现
│   │   └── integration/        # 外部框架集成
│   │       ├── __init__.py     # 集成模块初始化
│   │       └── torch_backend.py # PyTorch 后端实现
│   ├── passes/                  # Pass 优化系统
│   │   ├── __init__.py         # Pass 模块初始化
│   │   ├── base.py             # Pass 基础抽象类
│   │   ├── manager.py          # Pass 管理器
│   │   ├── dead_code_elimination.py      # 死代码消除
│   │   ├── constant_folding.py           # 常量折叠
│   │   └── common_subexpression_elimination.py  # 公共子表达式消除
│   ├── patterns/               # Pattern 匹配系统
│   │   ├── __init__.py         # Pattern 模块初始化
│   │   ├── base.py             # Pattern 基础抽象类
│   │   ├── manager.py          # Pattern 管理器
│   │   └── add_layernorm.py    # Add+LayerNorm 融合模式
│   └── utils/                  # 工具模块
│       ├── __init__.py         # 工具模块初始化
│       ├── logger.py           # 统一日志管理系统
│       └── optional_deps.py    # 可选依赖管理
├── tests/                      # 测试代码目录
│   ├── __init__.py             # 测试模块初始化
│   ├── unit/                   # 单元测试
│   │   ├── test_*.py           # 各模块单元测试
│   │   └── test_*_fx.py        # FX 相关测试
│   ├── integration/            # 集成测试
│   │   ├── test_*.py           # 集成测试用例
│   │   └── graph_module_wrapper.py  # 图模块包装器
│   └── performance/            # 性能测试
│       └── test_performance_benchmarks.py  # 性能基准测试
├── config/                     # 配置文件目录
│   └── optimization.toml       # 优化配置文件
├── requirements.txt            # 项目依赖
├── pyproject.toml             # 项目配置文件
└── README.md                  # 项目说明文档
```

### 目录结构说明

#### 核心源代码 (ngo/)
- **core/**: 核心功能模块，包含基础抽象、注册中心、配置管理、优化引擎和外部集成
- **passes/**: Pass 优化系统，实现各种图优化算法
- **patterns/**: Pattern 匹配系统，实现算子模式识别和融合
- **utils/**: 工具模块，提供日志和依赖管理功能

#### 测试代码 (tests/)
- **unit/**: 单元测试，测试各个模块的独立功能
- **integration/**: 集成测试，测试模块间的协作和端到端功能
- **performance/**: 性能测试，验证优化效果和性能指标

#### 配置和文档
- **config/**: 配置文件，包含优化参数和系统设置

## 🚀 快速开始

### 环境要求

- **Python**: 3.11+
- **PyTorch**: 2.6.0+
- **硬件**: 昇腾 Atlas 系列设备（可选）
- **操作系统**: Linux / macOS

### 安装

#### 从源码安装

```bash
# 克隆项目
git clone <repository-url>
cd ngo

# 安装依赖
pip install -r requirements.txt

# 安装 NGO
pip install -e .
```

#### 使用 pip 安装

```bash
pip install ngo
```

### 基本使用

#### 1. 简单示例

```python
import torch
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.layer_norm = nn.LayerNorm(5)

    def forward(self, x):
        x = self.linear(x)
        # 残差连接 + LayerNorm - 这是 NGO 可以优化的模式
        residual = x
        x = torch.relu(x)
        x = x + residual
        x = self.layer_norm(x)
        return x

# 创建模型
model = SimpleModel()

# 使用 NGO 后端编译模型
optimized_model = torch.compile(model, backend="ngo")

# 使用优化后的模型
input_data = torch.randn(32, 10)
output = optimized_model(input_data)
```

#### 2. 配置优化选项

```python
import torch
from ngo.core.integration.torch_backend import NGOBackend, NGOBackendOptions

# 创建优化配置
options = NGOBackendOptions(
    enable_optimization=True,
    optimization_level=5,  # 0-优先级最低，1-优先级次低，2-优先级低，3-优先级中，4-优先级高，5-优先级最高
    enable_profiling=True,
    custom_config={
        "enable_pattern_matching": True,
        "enable_pass_optimization": True
    }
)

backend = NGOBackend(options=options)

# 应用配置
model = torch.compile(model, backend=backend)
```

#### 3. 查看优化效果

```python
import time

def benchmark(model, input_data, num_runs=100):
    model.eval()

    # 预热
    for _ in range(10):
        _ = model(input_data)

    # 测试原始模型
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_data)
    original_time = time.time() - start_time

    # 测试优化模型
    optimized_model = torch.compile(model, backend="ngo")
    for _ in range(10):
        _ = optimized_model(input_data)

    start_time = time.time()
    for _ in range(num_runs):
        _ = optimized_model(input_data)
    optimized_time = time.time() - start_time

    # 计算加速比
    speedup = original_time / optimized_time
    print(f"原始时间: {original_time:.4f}s")
    print(f"优化时间: {optimized_time:.4f}s")
    print(f"加速比: {speedup:.2f}x")

    return speedup
```

## 📚 详细使用指南

### 支持的优化类型

#### 1. Pass 系统优化

NGO 提供了多种经典的图优化 Pass：

- **死代码消除**: 移除不可达的计算节点
- **常量折叠**: 在编译时计算常量表达式
- **公共子表达式消除**: 合并重复的计算

```python
# 启用特定的 Pass
from ngo.core.config import OptimizationConfigManager

config_manager = OptimizationConfigManager()
config_manager.enable_pass("dead_code_elimination")
config_manager.enable_pass("constant_folding")
config_manager.enable_pass("common_subexpression_elimination")
```

#### 2. Pattern 匹配优化

NGO 可以识别并优化常见的计算模式：

- **Add+LayerNorm 融合**: 将加法和层归一化融合为单个算子
- **其他模式**: 持续扩展中的模式库

```python
# 查看可用的 Pattern
from ngo.core.unified_registry import list_patterns
patterns = list_patterns()
print(f"可用 Pattern: {patterns}")

# 启用特定 Pattern
config_manager.enable_pattern("add_layernorm")
```

### 配置管理

#### 1. 使用配置文件

创建 `config/optimization.toml`：

```toml
[logging]
level = "INFO"
file = "ngo.log"
enable_console = true

[passes.dead_code_elimination]
enabled = true
priority = "HIGHEST"

[passes.constant_folding]
enabled = true
priority = "HIGH"

[patterns.add_layernorm]
enabled = true
priority = "NORMAL"
```

#### 2. 动态配置

```python
from ngo.core.config import OptimizationConfigManager

config_manager = OptimizationConfigManager()

# 动态启用/禁用优化
config_manager.set_pass_enabled("dead_code_elimination", True)
config_manager.set_pattern_enabled("add_layernorm", True)

# 设置优化级别
config_manager.set_optimization_level(2)
```

### 性能监控

```python
import torch
from ngo.core.integration.torch_backend import NGOBackend, NGOBackendOptions

# 创建带性能监控的后端
backend = NGOBackend(options=NGOBackendOptions(enable_profiling=True))

# 编译模型
compiled_model = torch.compile(model, backend=backend)

# 执行后查看性能统计
stats = backend.get_compilation_stats()
print(f"优化时间: {stats['optimization_time']:.4f}s")
print(f"识别的模式数: {stats['patterns_found']}")
print(f"应用的 Pass 数: {stats['passes_applied']}")
```

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest

# 运行单元测试
python -m pytest tests/unit/

# 运行集成测试
python -m pytest tests/integration/

# 运行性能测试
python -m pytest tests/performance/

# 生成覆盖率报告
python -m pytest --cov=ngo --cov-report=html
```

### 测试覆盖率

项目保持高测试覆盖率：
- 单元测试覆盖率: > 80%
- 集成测试覆盖率: > 70%
- 性能测试: 覆盖核心场景

## 🛠️ 开发指南

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd ngo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -e ".[dev]"

# 安装 pre-commit hooks
pre-commit install
```

### 代码风格

项目使用严格的代码风格控制：

```bash
# 代码格式化
black ngo/
isort ngo/

# 类型检查
mypy ngo/

# 静态分析
flake8 ngo/
```

### 添加新的优化 Pass

1. **创建 Pass 类**:

```python
from ngo.passes.base import BasePass, PassResult
from ngo.core.base import ComponentPriority

class MyCustomPass(BasePass):
    def __init__(self):
        super().__init__(
            name="my_custom_pass",
            description="My custom optimization pass",
            pass_type=PassType.TRANSFORM,
            priority=ComponentPriority.HIGH
        )

    def analyze(self, graph: torch.fx.Graph) -> AnalysisResult:
        # 实现分析逻辑
        return AnalysisResult(can_optimize=True)

    def transform(self, graph: torch.fx.Graph) -> TransformResult:
        # 实现优化逻辑
        return TransformResult(success=True)
```

2. **注册 Pass**:

```python
from ngo.core.unified_registry import register_pass

@register_pass("my_custom_pass")
class MyCustomPass(BasePass):
    # 实现同上
    pass
```

### 添加新的 Pattern

1. **创建 Pattern 类**:

```python
from ngo.patterns.base import BasePattern, PatternMatchResult

class MyCustomPattern(BasePattern):
    def __init__(self):
        super().__init__(
            name="my_custom_pattern",
            description="My custom pattern"
        )

    def match(self, graph: torch.fx.Graph) -> List[PatternMatchResult]:
        # 实现匹配逻辑
        return matches

    def replace(self, graph: torch.fx.Graph, matches: List[PatternMatchResult]) -> bool:
        # 实现替换逻辑
        return True
```

2. **注册 Pattern**:

```python
from ngo.core.unified_registry import register_pattern

@register_pattern("my_custom_pattern")
class MyCustomPattern(BasePattern):
    # 实现同上
    pass
```

## 📊 性能基准

### 典型场景性能提升

（待补充）
添加表头对齐符号：| 模型类型 | 优化前 (ms) | 优化后 (ms) | 加速比 | 内存节省 |
|:--------|:-----------:|:-----------:|:------:|:--------:|
| Transformer | xx | xx | xx | xx% |
| 推荐模型 | xx | xx | xx | xx% |

### 硬件支持

- **Atlas 800T A2**: 完全支持，最佳性能
- **Atlas 200T A2 Box16**: 完全支持
- **其他昇腾设备**: 基础支持

## 🔍 故障排除

### 修改日志级别打印调试日志

```python
# 修改toml配置文件
[logging]
level = "DEBUG"
file = "ngo.log"
enable_console = true
```

### 常见问题

#### 1. NGO 后端不可用

```bash
# 检查可用后端
python -c "import torch; print(torch._dynamo.list_backends())"

# 如果没有 'ngo'，检查安装
pip install -e .
```

#### 2. 优化效果不明显

```python
# 检查优化配置
from ngo.core.config import OptimizationConfigManager
config = OptimizationConfigManager()
print(config.get_current_config())

# 尝试提高优化级别
config.set_optimization_level(3)
```

#### 3. 模型精度下降

```python
# 禁用可能影响精度的优化
config.set_pass_enabled("constant_folding", False)

# 使用保守的优化级别
config.set_optimization_level(1)
```

```python
# 使用torchinductor后端进行定界
torch.compile(model, backend="inductor")
```

## 📄 许可证

本项目采用 Apache License 2.0 许可证。

<div align="center">

**NGO - 为昇腾硬件而生的高性能图优化器**

</div>