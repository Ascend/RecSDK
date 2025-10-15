# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NGO (NPU Graph Optimizer) - NPU 图优化器

一个用于昇腾硬件上 PyTorch 模型的图优化框架。
"""

__version__ = "0.1.0"
__author__ = "NGO Development Team"
__email__ = "ngo-dev@example.com"

# 核心导入
from .core import OptimizationContext, OptimizationEngine, OptimizerComponent
from .core.config import ConfigManager

# Torch 集成
from .core.integration.torch_backend import (
    NGOBackend,
    NGOBackendOptions,
    create_ngo_backend,
)
from .core.unified_registry import UnifiedRegistry

# Pass 系统导入
from .passes import BasePass, PassManager
from .passes.common_subexpression_elimination import CommonSubexpressionEliminationPass
from .passes.constant_folding import ConstantFoldingPass
from .passes.dead_code_elimination import DeadCodeEliminationPass

# Pattern 系统导入
from .patterns import BasePattern, BaseStrategy


__all__ = [
    # 核心
    "OptimizerComponent",
    "OptimizationEngine",
    "OptimizationContext",
    "ConfigManager",
    "UnifiedRegistry",
    # Torch 集成
    "NGOBackend",
    "NGOBackendOptions",
    "create_ngo_backend",
    # Pass 系统
    "BasePass",
    "PassManager",
    "DeadCodeEliminationPass",
    "ConstantFoldingPass",
    "CommonSubexpressionEliminationPass",
    # Pattern 系统
    "BasePattern",
    "BaseStrategy",
]
