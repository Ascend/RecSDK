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
集成测试的通用导入。

此模块为集成测试提供集中的导入，支持入口点注册和显式
NGO 后端初始化。
"""

import sys
import os

# 将项目根目录添加到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# 导入后端组件
from ngo.core.integration.torch_backend import NGOBackend, NGOBackendOptions, create_ngo_backend
from ngo.core.engine import OptimizationEngine
from ngo.core.unified_registry import UnifiedRegistry

# 注册 NGO 后端到 PyTorch 用于测试
import torch
try:
    @torch._dynamo.register_backend
    def ngo_backend(*args, **kwargs):
        return create_ngo_backend()(*args, **kwargs)
    print("✅ NGO 后端已成功注册到 PyTorch")
except Exception as e:
    print(f"⚠️  注册 NGO 后端失败: {e}")
    # 这对大多数测试来说不是关键错误

# 导入测试工具
from tests.integration.graph_module_wrapper import (
    create_graph_module_wrapper,
    ensure_graph_module,
    GraphModuleWrapper,
)

# Export all necessary components for integration tests
__all__ = [
    "NGOBackend",
    "NGOBackendOptions",
    "create_ngo_backend",
    "OptimizationEngine",
    "UnifiedRegistry",
    "create_graph_module_wrapper",
    "ensure_graph_module",
    "GraphModuleWrapper",
]
