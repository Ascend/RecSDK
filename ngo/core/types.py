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
NGO核心类型定义

该模块定义了NGO系统中使用的核心类型，避免循环导入问题。
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from .base import OptimizerComponent


class ComponentType(Enum):
    """组件类型枚举"""

    PASS = "pass"
    PATTERN = "pattern"


class RegistrationPhase(Enum):
    """注册组件的生效阶段"""

    BEFORE_DECOMP = "before_decomp"
    AFTER_DECOMP = "after_decomp"
    BOTH = "both"


# 类型别名，用于避免循环导入
BasePassType = Type[OptimizerComponent]
BasePatternType = Type[OptimizerComponent]
ComponentInstanceType = OptimizerComponent


def is_pass_class(component_class: Type[OptimizerComponent]) -> bool:
    """
    检查类是否为Pass类型

    Args:
        component_class: 组件类

    Returns:
        是否为Pass类型
    """
    module_name = component_class.__module__
    class_name = component_class.__name__

    # 通过模块路径判断
    if "passes" in module_name:
        return True

    # 通过类名判断
    if "Pass" in class_name and "Pattern" not in class_name:
        return True

    return False


def is_pattern_class(component_class: Type[OptimizerComponent]) -> bool:
    """
    检查类是否为Pattern类型

    Args:
        component_class: 组件类

    Returns:
        是否为Pattern类型
    """
    module_name = component_class.__module__
    class_name = component_class.__name__

    # 通过模块路径判断
    if "patterns" in module_name:
        return True

    # 通过类名判断
    if "Pattern" in class_name:
        return True

    return False


def get_component_type(component_class: Type[OptimizerComponent]) -> ComponentType:
    """
    获取组件类型

    Args:
        component_class: 组件类

    Returns:
        组件类型

    Raises:
        ValueError: 无法确定组件类型时
    """
    if is_pass_class(component_class):
        return ComponentType.PASS
    elif is_pattern_class(component_class):
        return ComponentType.PATTERN
    else:
        raise ValueError(f"无法确定组件类型: {component_class}")
