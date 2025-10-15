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
核心优化引擎实现。

提供用于协调优化组件和管理优化工作流的主要执行引擎。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ngo.utils.logger import get_logger
from ..base import OptimizationContext
from ..unified_registry import UnifiedRegistry, get_registry


@dataclass
class ExecutionResult:
    """优化执行的结果。"""

    success: bool
    execution_time: float
    components_executed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class EngineError(Exception):
    """引擎相关错误的基础异常。"""

    pass


class ExecutionError(EngineError):
    """执行失败时抛出。"""

    pass


class OptimizationEngine:
    """
    简化的优化引擎，协调优化组件的执行。

    Features:
    - 组件生命周期管理
    - 错误处理
    - 性能跟踪
    """

    def __init__(self, registry: Optional[UnifiedRegistry] = None):
        """
        初始化优化引擎。

        Args:
            registry: 组件注册表，默认使用全局注册表
        """
        self._registry = registry or get_registry()
        self._context = OptimizationContext()
        self._logger = get_logger(__name__)
        self._performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
        }

    @property
    def registry(self) -> UnifiedRegistry:
        """组件注册表"""
        return self._registry

    @property
    def context(self) -> OptimizationContext:
        """优化上下文"""
        return self._context

    @property
    def performance_metrics(self) -> Dict[str, Any]:
        """性能指标"""
        return self._performance_metrics.copy()

    @contextmanager
    def _execution_context(self):
        """执行操作上下文管理器"""
        start_time = time.time()
        execution_result = ExecutionResult(
            success=False, execution_time=0.0, components_executed=0
        )

        try:
            yield execution_result

        except Exception as e:
            execution_result.errors.append(str(e))
            raise

        finally:
            execution_result.execution_time = time.time() - start_time
            self._update_performance_metrics(execution_result)

    def _update_performance_metrics(self, result: ExecutionResult) -> None:
        """更新性能指标"""
        self._performance_metrics["total_executions"] += 1
        self._performance_metrics["total_execution_time"] += result.execution_time

        if result.success:
            self._performance_metrics["successful_executions"] += 1
        else:
            self._performance_metrics["failed_executions"] += 1

        if self._performance_metrics["total_executions"] > 0:
            self._performance_metrics["average_execution_time"] = (
                self._performance_metrics["total_execution_time"]
                / self._performance_metrics["total_executions"]
            )

    def execute(
        self,
        component_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        执行优化工作流

        Args:
            component_names: 要执行的组件名称列表，None表示所有组件
            config: 执行配置

        Returns:
            执行结果详情

        Raises:
            ExecutionError: 如果执行失败
        """
        with self._execution_context() as result:
            try:
                self._logger.info("开始执行优化工作流")

                # 应用配置
                if config:
                    self._context.set_config("execution_config", config)

                # 获取要执行的组件
                if component_names is None:
                    # 使用统一注册表获取所有已注册的组件
                    try:
                        passes = list(self._registry._pass_registrations.keys())
                        patterns = list(self._registry._pattern_registrations.keys())
                        components_to_execute = passes + patterns
                    except Exception:
                        components_to_execute = []
                else:
                    components_to_execute = component_names

                # 执行组件
                executed_count = 0
                for component_name in components_to_execute:
                    try:
                        self._execute_component(component_name)
                        executed_count += 1

                    except Exception as e:
                        error_msg = f"组件 {component_name} 执行失败: {e}"
                        result.errors.append(error_msg)
                        self._logger.error(error_msg)
                        # 继续执行其他组件，不中断整个流程

                result.success = len(result.errors) == 0
                result.components_executed = executed_count
                result.metrics = self._collect_execution_metrics()

                if result.success:
                    self._logger.info("优化工作流执行成功")
                else:
                    self._logger.warning(f"优化工作流执行完成，但有 {len(result.errors)} 个错误")

            except Exception as e:
                result.errors.append(str(e))
                self._logger.error(f"执行失败: {e}")
                raise ExecutionError(f"执行失败: {e}")

        return result

    def _execute_component(self, component_name: str) -> None:
        """
        执行单个组件

        Args:
            component_name: 要执行的组件名称
        """
        self._logger.info(f"执行组件: {component_name}")

        # 从统一注册表获取组件实例
        component = self._registry.get_component(component_name)
        if component is None:
            raise ExecutionError(f"组件 {component_name} 未找到或创建失败")

        # 执行组件
        try:
            execution_result = component.execute(self._context)
            self._logger.info(f"组件 {component_name} 执行成功, 执行结果: {execution_result}")

            # 在上下文中存储执行结果
            self._context.set_data(
                f"component_result_{component_name}", execution_result
            )

        except Exception as e:
            self._logger.error(f"组件 {component_name} 执行失败: {e}")
            raise

    def reset(self) -> None:
        """重置引擎到初始状态"""
        try:
            # 重置上下文
            self._context.clear()
            self._logger.info("引擎重置完成")
        except Exception as e:
            self._logger.error(f"引擎重置失败: {e}")
            raise EngineError(f"引擎重置失败: {e}")
        
    def _get_component_count(self) -> int:
        """获取组件数量"""
        try:
            passes = len(self._registry.list_passes()) if hasattr(self._registry, 'list_passes') else 0
            patterns = len(self._registry.list_patterns()) if hasattr(self._registry, 'list_patterns') else 0
            component_count = passes + patterns
        except Exception:
            component_count = 0
        return component_count

    def _collect_execution_metrics(self) -> Dict[str, Any]:
        """收集执行指标"""
        try:
            registry_stats = self._registry.get_statistics() if hasattr(self._registry, 'get_statistics') else {}
            context_stats = self._context.get_stats() if hasattr(self._context, 'get_stats') else {}
        except Exception:
            registry_stats = {}
            context_stats = {}

        metrics = {
            "engine_metrics": self.performance_metrics,
            "registry_stats": registry_stats,
            "context_stats": context_stats,
        }
        return metrics

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态信息"""

        return {
            "total_executions": self._performance_metrics["total_executions"],
            "successful_executions": self._performance_metrics["successful_executions"],
            "failed_executions": self._performance_metrics["failed_executions"],
            "average_execution_time": self._performance_metrics["average_execution_time"],
            "registry_components": self._get_component_count(),
        }

    def __str__(self) -> str:
        """引擎字符串表示"""

        return f"OptimizationEngine(组件数={self._get_component_count()}, 执行次数={self._performance_metrics['total_executions']})"

    def __repr__(self) -> str:
        """引擎详细字符串表示"""

        return (
            f"OptimizationEngine(组件数={self._get_component_count()}, "
            f"执行次数={self._performance_metrics['total_executions']}, "
            f"成功率={self._performance_metrics['successful_executions']}/{self._performance_metrics['total_executions']})"
        )
