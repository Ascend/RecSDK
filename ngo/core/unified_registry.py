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
统一的NGO优化组件注册器

该模块提供了一个统一的注册机制，用于管理优化Pass和Pattern。
支持装饰器模式注册，并提供查询和管理功能。
"""

import inspect
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from ngo.utils.logger import get_logger
from ngo.core.base import ComponentMetadata, OptimizerComponent, ComponentPriority
from ngo.core.types import RegistrationPhase, get_component_type, ComponentInstanceType


@dataclass
class RegistrationInfo:
    """组件注册信息"""

    name: str
    component_class: Type
    component_type: str  # "pass" 或 "pattern" 组件类型
    enabled: bool = True
    priority: int = 5
    phase: RegistrationPhase = RegistrationPhase.BOTH
    metadata: Optional[ComponentMetadata] = None
    instance: Optional[OptimizerComponent] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = ComponentMetadata(
                name=self.name,
                version="1.0.0",
                description=f"Auto-generated {self.component_type}: {self.name}",
            )


class UnifiedRegistry:
    """
    统一的NGO组件注册器

    提供Pass和Pattern的统一注册、查询和管理功能。
    实现全局单例模式，线程安全。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化注册器"""
        if self._initialized:
            return

        self._registrations: Dict[str, RegistrationInfo] = {}
        self._pass_registrations: Dict[str, RegistrationInfo] = {}
        self._pattern_registrations: Dict[str, RegistrationInfo] = {}
        self._lock = threading.RLock()
        self._logger = get_logger(__name__)
        self._initialized = True

    def register(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        priority: int = 5,
        phase: RegistrationPhase = RegistrationPhase.BOTH,
    ) -> Callable:
        """
        装饰器：注册Pass或Pattern类

        Args:
            name: 组件名称（默认为类名）
            enabled: 是否启用
            priority: 优先级（0-5，5为最高）
            phase: 生效阶段

        Returns:
            装饰器函数
        """

        def decorator(component_class: Type[OptimizerComponent]):
            # 使用类型定义模块确定组件类型
            component_type_enum = get_component_type(component_class)
            component_type = component_type_enum.value

            # 使用类名作为默认名称
            component_name = name or component_class.__name__

            # 创建注册信息
            registration_info = RegistrationInfo(
                name=component_name,
                component_class=component_class,
                component_type=component_type,
                enabled=enabled,
                priority=priority,
                phase=phase,
            )

            # 执行注册
            with self._lock:
                self._registrations[component_name] = registration_info

                if component_type == "pass":
                    self._pass_registrations[component_name] = registration_info
                else:
                    self._pattern_registrations[component_name] = registration_info

            return component_class

        return decorator

    def unregister(self, name: str) -> bool:
        """
        注销组件

        Args:
            name: 组件名称

        Returns:
            是否成功注销
        """
        with self._lock:
            if name not in self._registrations:
                return False

            registration_info = self._registrations[name]

            del self._registrations[name]

            if registration_info.component_type == "pass":
                del self._pass_registrations[name]
            else:
                del self._pattern_registrations[name]

            self._logger.info(f"注销组件: {name}")

            return True

    def get_registration(self, name: str) -> Optional[RegistrationInfo]:
        """
        获取注册信息

        Args:
            name: 组件名称

        Returns:
            注册信息或None
        """
        return self._registrations.get(name)

    def get_component(self, name: str, **kwargs) -> Optional[ComponentInstanceType]:
        """
        获取组件实例（兼容旧接口）

        Args:
            name: 组件名称
            **kwargs: 实例化参数

        Returns:
            组件实例或None
        """
        # 首先检查是否已有实例
        registration_info = self.get_registration(name)
        if registration_info is None:
            return None

        # 如果已有实例，直接返回
        if registration_info.instance is not None:
            return registration_info.instance

        # 否则创建新实例
        return self.create_instance(name, **kwargs)

    def create_instance(self, name: str, **kwargs) -> Optional[ComponentInstanceType]:
        """
        创建组件实例

        Args:
            name: 组件名称
            **kwargs: 实例化参数

        Returns:
            组件实例或None
        """
        registration_info = self.get_registration(name)
        if registration_info is None:
            return None

        try:
            # 更新元数据名称
            if registration_info.metadata:
                registration_info.metadata.name = name

            # 创建实例
            if registration_info.component_type == "pass":
                # 对于Pass，需要特殊处理构造函数
                if hasattr(registration_info.component_class, "__init__"):
                    init_signature = inspect.signature(
                        registration_info.component_class.__init__
                    )
                    if "metadata" in init_signature.parameters:
                        instance = registration_info.component_class(
                            metadata=registration_info.metadata, **kwargs
                        )
                    else:
                        instance = registration_info.component_class(**kwargs)
                else:
                    instance = registration_info.component_class()
            else:
                # 对于Pattern，检查构造函数
                if hasattr(registration_info.component_class, "__init__"):
                    init_signature = inspect.signature(
                        registration_info.component_class.__init__
                    )
                    if len(init_signature.parameters) == 1:  # 只有 self 参数
                        instance = registration_info.component_class()
                    elif "metadata" in init_signature.parameters:
                        instance = registration_info.component_class(
                            registration_info.metadata
                        )
                    else:
                        instance = registration_info.component_class()
                else:
                    instance = registration_info.component_class()

            # 更新实例的metadata名称
            if hasattr(instance, "metadata") and instance.metadata:
                instance.metadata.name = name

            registration_info.instance = instance
            self._logger.info(f"创建组件实例: {name}")
            return instance

        except Exception as e:
            self._logger.error(f"创建组件实例失败 {name}: {e}")
            import traceback

            self._logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    def list_registrations(
        self,
        component_type: Optional[str] = None,
        enabled_only: bool = True,
        phase: Optional[RegistrationPhase] = None,
        sort_by_priority: bool = True,
    ) -> List[RegistrationInfo]:
        """
        列出注册信息

        Args:
            component_type: 组件类型过滤 ("pass" 或 "pattern")
            enabled_only: 只显示启用的组件
            phase: 生效阶段过滤
            sort_by_priority: 按优先级排序

        Returns:
            注册信息列表
        """
        with self._lock:
            if component_type == "pass":
                registrations = list(self._pass_registrations.values())
            elif component_type == "pattern":
                registrations = list(self._pattern_registrations.values())
            else:
                registrations = list(self._registrations.values())

            # 过滤
            if enabled_only:
                registrations = [r for r in registrations if r.enabled]

            if phase is not None:
                registrations = [r for r in registrations if r.phase == phase]

            # 排序
            if sort_by_priority:
                registrations.sort(key=lambda x: x.priority, reverse=True)

            return registrations

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取注册器统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {
                "total_registrations": len(self._registrations),
                "pass_registrations": len(self._pass_registrations),
                "pattern_registrations": len(self._pattern_registrations),
                "enabled_registrations": len(
                    [r for r in self._registrations.values() if r.enabled]
                ),
                "disabled_registrations": len(
                    [r for r in self._registrations.values() if not r.enabled]
                ),
                "phase_distribution": {
                    phase.value: len(
                        [r for r in self._registrations.values() if r.phase == phase]
                    )
                    for phase in RegistrationPhase
                },
                "priority_distribution": {},
            }

            # 优先级分布
            for priority_enum in ComponentPriority:
                count = len(
                    [r for r in self._registrations.values() if r.priority == priority_enum.value]
                )
                if count > 0:
                    stats["priority_distribution"][priority_enum.value] = count

            return stats

    def print_registrations(
        self,
        component_type: Optional[str] = None,
        enabled_only: bool = True,
        phase: Optional[RegistrationPhase] = None,
    ) -> None:
        """
        打印注册信息

        Args:
            component_type: 组件类型过滤
            enabled_only: 只显示启用的组件
            phase: 生效阶段过滤
        """
        registrations = self.list_registrations(
            component_type=component_type,
            enabled_only=enabled_only,
            phase=phase,
            sort_by_priority=True,
        )

        if not registrations:
            self._logger.info("没有找到注册的组件")
            return

        title = "注册的组件"
        if component_type:
            title += f" ({component_type})"
        if enabled_only:
            title += " [已启用]"
        if phase:
            title += f" [{phase.value}]"

        self._logger.info(f"\n{title}:")
        self._logger.info("=" * 60)

        for reg in registrations:
            status = "✓" if reg.enabled else "✗"
            self._logger.info(f"{status} {reg.name}")
            self._logger.info(f"   类型: {reg.component_type}")
            self._logger.info(f"   优先级: {reg.priority}/5")
            self._logger.info(f"   阶段: {reg.phase.value}")

    def clear(self) -> None:
        """清空所有注册"""
        with self._lock:
            self._registrations.clear()
            self._pass_registrations.clear()
            self._pattern_registrations.clear()
            self._logger.info("清空所有注册")

    def reset(self) -> None:
        """重置注册器"""
        self.clear()
        self._logger.info("重置注册器")


# 全局注册器实例
def get_registry() -> UnifiedRegistry:
    """获取全局注册器实例"""
    return UnifiedRegistry()


# 便捷的装饰器函数
def register_pass(
    name: Optional[str] = None,
    enabled: bool = True,
    priority: int = 5,
    phase: RegistrationPhase = RegistrationPhase.BOTH,
) -> Callable:
    """
    注册Pass的装饰器

    Args:
        name: Pass名称（默认为类名）
        enabled: 是否启用
        priority: 优先级（0-5，5为最高）
        phase: 生效阶段

    Returns:
        装饰器函数
    """
    return get_registry().register(name, enabled, priority, phase)


def register_pattern(
    name: Optional[str] = None,
    enabled: bool = True,
    priority: int = 5,
    phase: RegistrationPhase = RegistrationPhase.BOTH,
) -> Callable:
    """
    注册Pattern的装饰器

    Args:
        name: Pattern名称（默认为类名）
        enabled: 是否启用
        priority: 优先级（0-5，5为最高）
        phase: 生效阶段

    Returns:
        装饰器函数
    """
    return get_registry().register(name, enabled, priority, phase)


# 便捷的查询函数
def get_registration(name: str) -> Optional[RegistrationInfo]:
    """获取注册信息"""
    return get_registry().get_registration(name)


def create_instance(name: str, **kwargs) -> Optional[ComponentInstanceType]:
    """创建组件实例"""
    return get_registry().create_instance(name, **kwargs)


def list_passes(
    enabled_only: bool = True,
    phase: Optional[RegistrationPhase] = None,
    sort_by_priority: bool = True,
) -> List[RegistrationInfo]:
    """列出注册的Pass"""
    return get_registry().list_registrations(
        component_type="pass",
        enabled_only=enabled_only,
        phase=phase,
        sort_by_priority=sort_by_priority,
    )


def list_patterns(
    enabled_only: bool = True,
    phase: Optional[RegistrationPhase] = None,
    sort_by_priority: bool = True,
) -> List[RegistrationInfo]:
    """列出注册的Pattern"""
    return get_registry().list_registrations(
        component_type="pattern",
        enabled_only=enabled_only,
        phase=phase,
        sort_by_priority=sort_by_priority,
    )


def print_registrations(
    component_type: Optional[str] = None,
    enabled_only: bool = True,
    phase: Optional[RegistrationPhase] = None,
) -> None:
    """打印注册信息"""
    get_registry().print_registrations(
        component_type=component_type,
        enabled_only=enabled_only,
        phase=phase,
    )


def get_registry_statistics() -> Dict[str, Any]:
    """获取注册器统计信息"""
    return get_registry().get_statistics()


def clear_registry() -> None:
    """清空注册器"""
    get_registry().clear()
