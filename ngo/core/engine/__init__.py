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
优化引擎模块

为NGO（NPU图优化器）提供核心执行引擎，协调组件执行、处理错误
并管理优化工作流。
"""

from .engine import (
    EngineError,
    ExecutionError,
    ExecutionResult,
    OptimizationEngine,
)

__all__ = [
    "EngineError",
    "ExecutionError",
    "ExecutionResult",
    "OptimizationEngine",
]
