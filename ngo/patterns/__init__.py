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
NGO Pattern系统。

该模块为图优化提供Pattern匹配和策略执行功能。
"""

# 导入所有可用的Pattern模块，确保装饰器被执行
from .add_layernorm import AddLayerNormPattern
from .base import (
    BasePattern,
    BaseStrategy,
    PatternMatchResult,
    StrategyExecutionResult,
    StrategyPriority,
    StrategyStatistics,
    clear_pattern_registry,
    get_global_pattern_manager,
    get_pattern,
    list_patterns,
    register_pattern,
    register_pattern_class,
    register_strategy,
)

__all__ = [
    'BasePattern',
    'BaseStrategy',
    'PatternMatchResult',
    'StrategyExecutionResult',
    'StrategyPriority',
    'StrategyStatistics',
    'register_strategy',
    'register_pattern',
    'register_pattern_class',
    'get_pattern',
    'list_patterns',
    'clear_pattern_registry',
    'get_global_pattern_manager',
    'AddLayerNormPattern'
]