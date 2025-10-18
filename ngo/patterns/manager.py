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
NGO优化系统的Pattern管理器。

该模块提供了PatternManager类，用于注册、管理和执行优化Pattern。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from torch.fx import GraphModule, Node

from ngo.utils.logger import get_logger
from ..core.base import ComponentMetadata, OptimizationContext
from .base import BasePattern, PatternMatchResult, StrategyExecutionResult


class PatternManagerState(Enum):
    """Pattern管理器生命周期状态。"""

    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class PatternExecutionInfo:
    """Pattern执行信息。"""

    pattern_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    last_execution_time: Optional[float] = None
    enabled: bool = True

    @property
    def success_rate(self) -> float:
        """计算成功率。"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


class PatternManager:
    """
    NGO优化的Pattern系统管理器。

    该类管理优化Pattern及其策略的注册、调度和执行。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Pattern管理器。

        Args:
            config: Pattern管理器的配置字典
        """
        self._config = config or {}
        self._pattern_info: Dict[str, PatternExecutionInfo] = {}
        self._state = PatternManagerState.CREATED
        self._logger = get_logger("ngo.PatternManager")
        self._context: Optional[OptimizationContext] = None

    def initialize(self) -> None:
        """Initialize the pattern manager."""
        if self._state != PatternManagerState.CREATED:
            raise RuntimeError(
                f"PatternManager already initialized or in invalid state: {self._state}"
            )

        try:
            self._logger.info("Initializing PatternManager")
            self._context = OptimizationContext(self._config)
            self._state = PatternManagerState.INITIALIZED
            self._logger.info("Successfully initialized PatternManager")
        except Exception as e:
            self._state = PatternManagerState.ERROR
            self._logger.error(f"Failed to initialize PatternManager: {e}")
            raise RuntimeError(f"Initialization failed: {e}") from e

    def get_pattern(self, pattern_id: str) -> Optional[BasePattern]:
        """
        Get a registered pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern instance or None if not found
        """
        from ..core.unified_registry import create_instance
        return create_instance(pattern_id)

    def list_patterns(self) -> List[str]:
        """
        List all registered pattern IDs.

        Returns:
            List of pattern IDs in execution order
        """
        from ..core.unified_registry import list_patterns

        registrations = list_patterns(enabled_only=True, sort_by_priority=True)
        return [reg.name for reg in registrations]

    def enable_pattern(self, pattern_id: str) -> bool:
        """
        Enable a pattern.

        Args:
            pattern_id: Pattern ID to enable

        Returns:
            True if successful, False otherwise
        """
        from ..core.unified_registry import get_registration

        registration = get_registration(pattern_id)
        if registration is None:
            self._logger.warning(f"Pattern {pattern_id} not found in registry")
            return False

        if registration.enabled:
            return True  # Already enabled

        registration.enabled = True
        self._logger.info(f"Enabled pattern {pattern_id}")
        return True

    def disable_pattern(self, pattern_id: str) -> bool:
        """
        Disable a pattern.

        Args:
            pattern_id: Pattern ID to disable

        Returns:
            True if successful, False otherwise
        """
        from ..core.unified_registry import get_registration

        registration = get_registration(pattern_id)
        if registration is None:
            self._logger.warning(f"Pattern {pattern_id} not found in registry")
            return False

        if not registration.enabled:
            return True  # Already disabled

        registration.enabled = False
        self._logger.info(f"Disabled pattern {pattern_id}")
        return True

    def execute_pattern(
        self, pattern_id: str, graph_module: GraphModule
    ) -> List[StrategyExecutionResult]:
        """
        Execute a single pattern on a graph module.

        Args:
            pattern_id: Pattern ID to execute
            graph_module: Graph module to optimize

        Returns:
            List of strategy execution results
        """
        from ..core.unified_registry import create_instance, get_registration

        registration = get_registration(pattern_id)
        if registration is None:
            raise ValueError(f"Pattern {pattern_id} not found in registry")

        if not registration.enabled:
            self._logger.info(f"Pattern {pattern_id} is disabled, skipping")
            return []

        if self._state not in [
            PatternManagerState.INITIALIZED,
            PatternManagerState.RUNNING,
        ]:
            raise RuntimeError(f"PatternManager not ready for execution: {self._state}")

        # Create pattern instance
        pattern = create_instance(pattern_id)
        if pattern is None:
            raise RuntimeError(f"Failed to create pattern instance: {pattern_id}")

        # Get or create pattern info
        if pattern_id not in self._pattern_info:
            self._pattern_info[pattern_id] = PatternExecutionInfo(
                pattern_name=pattern_id, enabled=True
            )
        pattern_info = self._pattern_info[pattern_id]

        start_time = time.time()

        try:
            self._state = PatternManagerState.RUNNING
            self._logger.info(f"Executing pattern {pattern_id}")

            # Match pattern
            match_result = pattern.match(graph_module)
            if not match_result.matched:
                self._logger.info(f"Pattern {pattern_id} did not match")
                return []

            # Execute strategies
            results = pattern.execute_strategies(graph_module, match_result)
            execution_time = time.time() - start_time

            # Update statistics
            pattern_info.execution_count += 1
            pattern_info.total_execution_time += execution_time
            pattern_info.last_execution_time = execution_time

            if all(result.success for result in results):
                pattern_info.success_count += 1
                self._logger.info(
                    f"Pattern {pattern_id} executed successfully in {execution_time:.4f}s"
                )
            else:
                pattern_info.failure_count += 1
                self._logger.warning(f"Pattern {pattern_id} execution had failures")

            # Reset state back to INITIALIZED after successful execution
            self._state = PatternManagerState.INITIALIZED
            return results

        except Exception as e:
            execution_time = time.time() - start_time
            pattern_info.execution_count += 1
            pattern_info.failure_count += 1
            pattern_info.total_execution_time += execution_time
            pattern_info.last_execution_time = execution_time

            self._logger.error(f"Pattern {pattern_id} execution failed: {e}")

            # Reset state back to INITIALIZED even after exception
            self._state = PatternManagerState.INITIALIZED
            raise RuntimeError(f"Pattern execution failed: {e}") from e

    def execute_all_patterns(self, graph_module: GraphModule) -> Dict[str, List[StrategyExecutionResult]]:
        """
        Execute all registered patterns on a graph module.

        Args:
            graph_module: Graph module to optimize

        Returns:
            Dictionary mapping pattern IDs to execution results
        """
        if self._state not in [
            PatternManagerState.INITIALIZED,
            PatternManagerState.RUNNING,
        ]:
            raise RuntimeError(f"PatternManager not ready for execution: {self._state}")

        results = {}

        # Get patterns from registry
        pattern_ids = self.list_patterns()

        for pattern_id in pattern_ids:
            try:
                pattern_results = self.execute_pattern(pattern_id, graph_module)
                results[pattern_id] = pattern_results
            except (RuntimeError, ValueError) as e:
                self._logger.error(f"Failed to execute pattern {pattern_id}: {e}")
                results[pattern_id] = []

        return results

    def get_pattern_info(self, pattern_id: str) -> Optional[PatternExecutionInfo]:
        """
        Get execution information for a pattern.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern execution info or None if not found
        """
        from ..core.unified_registry import get_registration

        registration = get_registration(pattern_id)
        if registration is None:
            return None

        # Get or create pattern info
        if pattern_id not in self._pattern_info:
            self._pattern_info[pattern_id] = PatternExecutionInfo(
                pattern_name=pattern_id, enabled=registration.enabled
            )

        return self._pattern_info[pattern_id]

    def get_manager_statistics(self) -> Dict[str, Any]:
        """
        Get overall manager statistics.

        Returns:
            Dictionary containing manager statistics
        """
        from ..core.unified_registry import get_registry_statistics, list_patterns

        registry_stats = get_registry_statistics()
        pattern_registrations = list_patterns(enabled_only=True)

        total_patterns = registry_stats['pattern_registrations']
        enabled_patterns = len(pattern_registrations)
        
        total_executions = 0
        total_successes = 0
        total_failures = 0
        total_execution_time = 0.0
        
        for info in self._pattern_info.values():
            total_executions += info.execution_count
            total_successes += info.success_count
            total_failures += info.failure_count
            total_execution_time += info.total_execution_time

        overall_success_rate = total_successes / total_executions if total_executions > 0 else 0.0
        avg_execution_time = total_execution_time / total_executions if total_executions > 0 else 0.0

        return {
            "total_patterns": total_patterns,
            "enabled_patterns": enabled_patterns,
            "disabled_patterns": total_patterns - enabled_patterns,
            "total_executions": total_executions,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "overall_success_rate": overall_success_rate,
            "average_execution_time": avg_execution_time,
            "manager_state": self._state.value,
            "execution_order": [reg.name for reg in pattern_registrations],
        }

    def clear_statistics(self) -> None:
        """Clear all execution statistics."""
        for info in self._pattern_info.values():
            info.execution_count = 0
            info.success_count = 0
            info.failure_count = 0
            info.total_execution_time = 0.0
            info.last_execution_time = None

        self._logger.info("Cleared all pattern statistics")

    def __str__(self) -> str:
        """String representation of the pattern manager."""
        stats = self.get_manager_statistics()
        return f"PatternManager(patterns={stats['total_patterns']}, enabled={stats['enabled_patterns']}, state={self._state.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        stats = self.get_manager_statistics()
        return (
            f"PatternManager(total_patterns={stats['total_patterns']}, "
            f"enabled_patterns={stats['enabled_patterns']}, "
            f"state='{self._state.value}', "
            f"executions={stats['total_executions']})"
        )
