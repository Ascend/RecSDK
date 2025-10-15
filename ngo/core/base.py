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
NGO 优化框架的基础组件。

该模块为 NGO (NPU 图优化器) 框架提供基础的抽象基类和上下文管理。
"""

import dataclasses
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union

from torch.fx import GraphModule

from ngo.utils.logger import get_logger


class ComponentState(Enum):
    """组件生命周期状态。"""

    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    ERROR = "error"
    DESTROYED = "destroyed"


class ComponentPriority(Enum):
    """组件执行优先级。"""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHER = 4
    HIGHEST = 5


@dataclass
class ComponentMetadata:
    """优化器组件的元数据。"""

    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    config_schema: Optional[Dict[str, Any]] = None


class OptimizerComponent(ABC):
    """
    所有 NGO 优化器组件的抽象基类。

    该类为 NGO 框架中的所有优化器组件定义接口和生命周期管理。
    """

    def __init__(self, metadata: ComponentMetadata):
        """
        初始化优化器组件。

        Args:
            metadata: 组件元数据，包括名称、版本等
        """
        self._metadata = metadata
        self._state = ComponentState.CREATED
        self._priority = ComponentPriority.NORMAL
        self._config: Dict[str, Any] = {}
        self._logger = get_logger()
        self._id = str(uuid.uuid4())
        self._created_time = time.time()
        self._last_execution_time: Optional[float] = None
        self._execution_count = 0
        self._error_count = 0

    @property
    def metadata(self) -> ComponentMetadata:
        """获取组件元数据。"""
        return self._metadata

    @property
    def state(self) -> ComponentState:
        """获取当前组件状态。"""
        return self._state

    @property
    def priority(self) -> ComponentPriority:
        """获取组件优先级。"""
        return self._priority

    @priority.setter
    def priority(self, value: ComponentPriority) -> None:
        """设置组件优先级。"""
        self._priority = value

    @property
    def id(self) -> str:
        """获取唯一组件标识符。"""
        return self._id

    @property
    def config(self) -> Dict[str, Any]:
        """获取组件配置。"""
        return self._config.copy()

    @property
    def execution_stats(self) -> Dict[str, Any]:
        """获取组件执行统计。"""
        return {
            "created_time": self._created_time,
            "last_execution_time": self._last_execution_time,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "uptime": (
                time.time() - self._created_time
                if self._state != ComponentState.DESTROYED
                else 0
            ),
        }

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        设置组件配置。

        Args:
            config: 配置字典

        Raises:
            ValueError: 如果配置无效
        """
        if self._state in [ComponentState.RUNNING, ComponentState.DESTROYED]:
            raise ValueError(f"Cannot set config in {self._state} state")

        # 基础验证 - 确保配置可 JSON 序列化
        if self._metadata.config_schema:
            try:
                json.dumps(config)  # 确保配置可 JSON 序列化
            except (TypeError, ValueError) as e:
                self._logger.error(f"配置验证失败: {e}")
                raise ValueError("Invalid configuration")

        self._config = config.copy()
        self._logger.info(f"Configuration updated for {self._metadata.name}")

    @abstractmethod
    def initialize(self) -> None:
        """
        初始化组件。

        该方法在组件使用前被调用一次。
        子类应该重写此方法以执行初始化。

        Raises:
            RuntimeError: 如果初始化失败
        """
        if self._state != ComponentState.CREATED:
            raise RuntimeError(
                f"Component already initialized or in invalid state: {self._state}"
            )

        try:
            self._logger.info(f"Initializing {self._metadata.name}")
            # 子类应该实现特定的初始化逻辑
            self._state = ComponentState.INITIALIZED
            self._logger.info(f"Successfully initialized {self._metadata.name}")
        except Exception as e:
            self._state = ComponentState.ERROR
            self._error_count += 1
            self._logger.error(f"Failed to initialize {self._metadata.name}: {e}")
            raise RuntimeError(f"Initialization failed: {e}") from e

    @abstractmethod
    def execute(self, context: "OptimizationContext") -> Any:
        """
        执行组件的优化逻辑。

        Args:
            context: 包含执行环境的优化上下文

        Returns:
            执行结果

        Raises:
            RuntimeError: 如果执行失败
        """
        if self._state != ComponentState.INITIALIZED:
            raise RuntimeError(f"Component not ready for execution: {self._state}")

        try:
            self._state = ComponentState.RUNNING
            self._logger.info(f"Executing {self._metadata.name}")

            # 更新执行统计
            self._last_execution_time = time.time()
            self._execution_count += 1

            # 子类应该实现特定的执行逻辑
            result = self._execute_impl(context)

            self._logger.info(f"Successfully executed {self._metadata.name}")
            return result

        except Exception as e:
            self._state = ComponentState.ERROR
            self._error_count += 1
            self._logger.error(f"Execution failed for {self._metadata.name}: {e}")
            raise RuntimeError(f"Execution failed: {e}") from e
        finally:
            if self._state == ComponentState.RUNNING:
                self._state = ComponentState.INITIALIZED

    @abstractmethod
    def _execute_impl(self, context: "OptimizationContext") -> Any:
        """
        组件执行的内部实现。

        子类必须实现此方法以提供特定的优化逻辑。

        Args:
            context: 优化上下文

        Returns:
            执行结果
        """
        pass

    def reset(self) -> None:
        """将组件重置为初始状态。"""
        if self._state == ComponentState.DESTROYED:
            raise RuntimeError("Cannot reset destroyed component")

        self._state = ComponentState.CREATED
        self._config.clear()
        self._execution_count = 0
        self._error_count = 0
        self._logger.info(f"Reset {self._metadata.name}")

    def destroy(self) -> None:
        """销毁组件并释放资源。"""
        try:
            self._logger.info(f"Destroying {self._metadata.name}")
            # 子类可以重写以清理特定资源
            self._state = ComponentState.DESTROYED
            self._logger.info(f"Successfully destroyed {self._metadata.name}")
        except Exception as e:
            self._logger.error(f"Error destroying {self._metadata.name}: {e}")
            self._state = ComponentState.ERROR
            raise

    def __str__(self) -> str:
        """String representation of the component."""
        return f"{self._metadata.name} v{self._metadata.version} ({self._state.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}(name='{self._metadata.name}', "
            f"version='{self._metadata.version}', state='{self._state.value}', "
            f"id='{self._id[:8]}...')"
        )


class OptimizationContext:
    """
    优化操作的上下文。

    该类管理优化组件的执行环境和共享状态。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化优化上下文。

        Args:
            config: 优化上下文的全局配置
        """
        self._config = config or {}
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._logger = get_logger("ngo.OptimizationContext")
        self._start_time = time.time()
        self._components: Dict[str, OptimizerComponent] = {}
        self._graph_module: Optional[GraphModule] = None
        self._component_results: Dict[str, Any] = {}

    @property
    def config(self) -> Dict[str, Any]:
        """获取全局配置。"""
        return self._config.copy()

    @property
    def data(self) -> Dict[str, Any]:
        """获取上下文数据。"""
        return self._data.copy()

    @property
    def graph_module(self) -> Optional[GraphModule]:
        """获取图模块。"""
        return self._graph_module

    @property
    def graph_size(self) -> Optional[int]:
        """获取图大小。"""
        return self._data.get("graph_size", -1)

    @property
    def component_results(self) -> Dict[str, Any]:
        """获取组件结果。"""
        return self._component_results

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
        self._logger.debug(f"Set config {key} = {value}")

    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get data value by key.

        Args:
            key: Data key
            default: Default value if key not found

        Returns:
            Data value or default
        """
        return self._data.get(key, default)

    def set_data(self, key: str, value: Any) -> None:
        """
        Set data value.

        Args:
            key: Data key
            value: Data value
        """
        self._data[key] = value
        self._logger.debug(f"Set data {key} = {value}")


  
    def set_graph_module(self, graph_module: GraphModule) -> None:
        """
        Set graph module.

        Args:
            graph_module: Graph module
        """
        self._graph_module = graph_module
        self._logger.debug(f"Set graph module")
        
    def set_component_result(self, component_name: str, result: Any) -> None:
        """
        Set component result.
        """
        self._component_results[component_name] = result
        self._logger.debug(f"Set component result {component_name} = {result}")
    
    def register_component(self, component: OptimizerComponent) -> None:
        """
        Register a component with the context.

        Args:
            component: Component to register
        """
        self._components[component.id] = component
        self._logger.debug(f"Registered component {component.metadata.name}")

    
    def get_component(self, component_id: str) -> Optional[OptimizerComponent]:
        """
        Get component by ID.

        Args:
            component_id: Component ID

        Returns:
            Component or None if not found
        """
        return self._components.get(component_id)

    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context statistics.

        Returns:
            Context statistics dictionary
        """
        return {
            "uptime": time.time() - self._start_time,
            "start_time": self._start_time,
            "registered_components": len(self._components),
            "data_keys": len(self._data),
            "metadata_keys": len(self._metadata),
        }

    def clear(self) -> None:
        """Clear all context data and metadata."""
        self._data.clear()
        self._metadata.clear()
        self._components.clear()
        self._logger.info("Context cleared")

    def copy(self) -> "OptimizationContext":
        """
        Create a copy of the context.

        Returns:
            New context with copied data
        """
        new_context = OptimizationContext(self._config.copy())
        new_context._data = self._data.copy()
        new_context._metadata = self._metadata.copy()
        new_context._components = self._components.copy()
        return new_context

    def __enter__(self) -> "OptimizationContext":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self._logger.error(f"Context exited with error: {exc_val}")
        self._logger.info("Context exited")

    def __str__(self) -> str:
        """String representation of the context."""
        return f"OptimizationContext(components={len(self._components)}, data={len(self._data)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"OptimizationContext(config_keys={len(self._config)}, "
            f"data_keys={len(self._data)}, components={len(self._components)})"
        )
