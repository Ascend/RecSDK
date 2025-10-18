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
NGO的配置管理模块。

该模块提供简化的配置管理功能，包括TOML文件支持和动态更新。
"""

from .manager import ConfigError, ConfigManager, ConfigValidationError, ConfigSecurityError
from .optimization_config import (
    OptimizationConfigManager,
    PassConfigEntry,
    PatternConfigEntry,
    get_global_optimization_config_manager,
)

__all__ = [
    "ConfigManager",
    "ConfigError",
    "ConfigValidationError",
    "ConfigSecurityError",
    "OptimizationConfigManager",
    "PassConfigEntry",
    "PatternConfigEntry",
    "get_global_optimization_config_manager"
]
