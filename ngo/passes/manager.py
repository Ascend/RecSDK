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
NGO优化系统的Pass管理器。

该模块提供了PassManager类，用于处理优化Pass的注册、调度和执行。
它维护Pass注册表、执行顺序，并提供监控功能。
"""

import datetime
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Type

from ngo.utils.logger import get_logger

from ngo.core.base import ComponentPriority, OptimizationContext
from ngo.passes.base import BasePass, PassResult, PassState, PassType


class PassExecutionInfo:
    """Pass执行信息。"""

    def __init__(self, pass_instance: BasePass, execution_order: int):
        self.pass_instance = pass_instance
        self.execution_order = execution_order
        self.last_execution_time: Optional[datetime] = None
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        self.last_result: Optional[PassResult] = None


class PassManager:
    """
    优化Pass管理器。

    处理优化Pass的注册、调度和执行，具有适当的依赖管理和执行监控功能。
    """

    def __init__(self):
        """初始化PassManager。"""
        self._passes: Dict[str, PassExecutionInfo] = {}
        self._execution_lock = Lock()
        self.logger = get_logger("ngo.pass.manager")

        # 执行统计信息
        self._total_executions = 0
        self._total_execution_time = 0.0
        self._execution_history: List[Dict[str, Any]] = []

        self.logger.info("PassManager已初始化")

    def get_pass(self, pass_name: str) -> Optional[BasePass]:
        """
        根据名称获取已注册的Pass。

        Args:
            pass_name: 要检索的Pass名称

        Returns:
            Pass实例，如果未找到则返回None
        """
        from ..core.unified_registry import create_instance
        return create_instance(pass_name)

    def list_passes(self, pass_type: Optional[PassType] = None,
                   enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        列出已注册的Pass，支持可选过滤。

        Args:
            pass_type: 按Pass类型过滤
            enabled_only: 仅过滤已启用的Pass

        Returns:
            Pass信息字典列表
        """
        from ..core.unified_registry import create_instance, list_passes

        passes_info = []
        registrations = list_passes(enabled_only=enabled_only, sort_by_priority=True)

        for reg in registrations:
            pass_instance = create_instance(reg.name)
            if pass_instance is None:
                continue

            # 应用Pass类型过滤器
            if pass_type and pass_instance.pass_type != pass_type:
                continue

            # 获取执行信息（如果可用）
            exec_info = self._passes.get(reg.name)
            execution_count = exec_info.execution_count if exec_info else 0
            success_count = exec_info.success_count if exec_info else 0
            total_execution_time = exec_info.total_execution_time if exec_info else 0.0

            pass_info = {
                'name': reg.name,
                'type': pass_instance.pass_type.name,
                'enabled': reg.enabled,
                'priority': reg.priority,
                'execution_order': reg.priority,  # 使用优先级作为执行顺序
                'execution_count': execution_count,
                'success_rate': success_count / max(1, execution_count),
                'avg_execution_time': total_execution_time / max(1, execution_count)
            }
            passes_info.append(pass_info)

        # 按优先级排序（降序）
        passes_info.sort(key=lambda x: x['priority'], reverse=True)
        return passes_info

    def enable_pass(self, pass_name: str) -> bool:
        """
        启用Pass。

        Args:
            pass_name: 要启用的Pass名称

        Returns:
            成功返回True，否则返回False
        """
        from ..core.unified_registry import get_registration

        registration = get_registration(pass_name)
        if registration is None:
            self.logger.warning(f"在注册表中未找到Pass {pass_name}")
            return False

        if registration.enabled:
            return True  # 已经启用

        registration.enabled = True
        self.logger.info(f"已启用Pass: {pass_name}")
        return True

    def disable_pass(self, pass_name: str) -> bool:
        """
        禁用Pass。

        Args:
            pass_name: 要禁用的Pass名称

        Returns:
            成功返回True，否则返回False
        """
        from ..core.unified_registry import get_registration

        registration = get_registration(pass_name)
        if registration is None:
            self.logger.warning(f"在注册表中未找到Pass {pass_name}")
            return False

        if not registration.enabled:
            return True  # 已经禁用

        registration.enabled = False
        self.logger.info(f"已禁用Pass: {pass_name}")
        return True

    def get_execution_order(self) -> List[str]:
        """
        获取Pass的当前执行顺序。

        Returns:
            按执行顺序排列的Pass名称列表
        """
        from ..core.unified_registry import list_passes

        # 从注册表获取按优先级排序的Pass
        registrations = list_passes(enabled_only=True, sort_by_priority=True)
        return [reg.name for reg in registrations]

    def execute_pass(self, pass_name: str, context: OptimizationContext) -> Optional[PassResult]:
        """
        执行单个Pass。

        Args:
            pass_name: 要执行的Pass名称
            context: 优化上下文

        Returns:
            Pass结果，如果未找到Pass则返回None
        """
        from ..core.unified_registry import create_instance, get_registration

        registration = get_registration(pass_name)
        if registration is None:
            self.logger.error(f"Pass {pass_name} not found in registry")
            return None

        if not registration.enabled:
            self.logger.warning(f"Pass {pass_name} is disabled")
            return None

        # Create or get pass instance
        pass_instance = create_instance(pass_name)
        if pass_instance is None:
            self.logger.error(f"Failed to create pass instance: {pass_name}")
            return None

        # Get or create execution info
        if pass_name not in self._passes:
            execution_order = len(self._passes)
            exec_info = PassExecutionInfo(pass_instance, execution_order)
            self._passes[pass_name] = exec_info
        else:
            exec_info = self._passes[pass_name]
            exec_info.pass_instance = pass_instance

        start_time = datetime.now()

        try:
            self.logger.info(f"Executing pass: {pass_name}")

            # Initialize pass if needed
            if pass_instance.state == PassState.CREATED:
                if not pass_instance.initialize(context):
                    self.logger.error(f"Failed to initialize pass {pass_name}")
                    return None

            # Execute pass
            result = pass_instance.execute(context)

            # Update execution statistics
            exec_info.execution_count += 1
            exec_info.last_execution_time = datetime.now()
            exec_info.last_result = result

            execution_time = (datetime.now() - start_time).total_seconds()
            exec_info.total_execution_time += execution_time

            if result.success:
                exec_info.success_count += 1
                self.logger.info(f"Pass {pass_name} executed successfully in {execution_time:.3f}s")
            else:
                self.logger.warning(f"Pass {pass_name} execution failed: {result.metrics.error_message}")

            # Update global statistics
            self._total_executions += 1
            self._total_execution_time += execution_time

            # Add to execution history
            self._execution_history.append({
                'pass_name': pass_name,
                'timestamp': datetime.now(),
                'execution_time': execution_time,
                'success': result.success,
                'modified_graph': result.modified_graph
            })

            return result

        except Exception as e:
            self.logger.exception(f"Error executing pass {pass_name}: {e}")
            return None

    def execute_passes(self, context: OptimizationContext,
                      pass_names: Optional[List[str]] = None) -> List[PassResult]:
        """
        Execute multiple passes in order.

        Args:
            context: Optimization context
            pass_names: List of pass names to execute (None for all enabled)

        Returns:
            List of pass results
        """
        if pass_names is None:
            # Execute all enabled passes in priority order
            execution_order = self.get_execution_order()
            pass_names = [name for name in execution_order
                         if self._passes[name].pass_instance.config.enabled]

        results = []

        for pass_name in pass_names:
            result = self.execute_pass(pass_name, context)
            if result is not None:
                results.append(result)

        return results

    def get_pass_info(self, pass_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a pass.

        Args:
            pass_name: Name of the pass

        Returns:
            Pass information dictionary or None if not found
        """
        from ..core.unified_registry import create_instance, get_registration

        registration = get_registration(pass_name)
        if registration is None:
            return None

        pass_instance = create_instance(pass_name)
        if pass_instance is None:
            return None

        # Get execution info if available
        exec_info = self._passes.get(pass_name)
        execution_count = exec_info.execution_count if exec_info else 0
        success_count = exec_info.success_count if exec_info else 0
        total_execution_time = exec_info.total_execution_time if exec_info else 0.0

        return {
            'name': pass_name,
            'id': pass_instance.pass_id,
            'type': pass_instance.pass_type.name,
            'state': pass_instance.state.name,
            'enabled': registration.enabled,
            'priority': registration.priority,
            'config': pass_instance.config.to_dict(),
            'execution_order': registration.priority,
            'execution_count': execution_count,
            'success_count': success_count,
            'success_rate': success_count / max(1, execution_count),
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / max(1, execution_count),
            'last_execution_time': exec_info.last_execution_time.isoformat() if exec_info and exec_info.last_execution_time else None,
            'statistics': pass_instance.get_statistics()
        }

    def get_manager_statistics(self) -> Dict[str, Any]:
        """
        Get overall manager statistics.

        Returns:
            Dictionary with manager statistics
        """
        from ..core.unified_registry import get_registry_statistics, list_passes

        registry_stats = get_registry_statistics()
        pass_registrations = list_passes(enabled_only=True)

        total_passes = registry_stats['pass_registrations']
        enabled_passes = len(pass_registrations)

        successful_executions = sum(info.success_count for info in self._passes.values())

        return {
            'total_passes': total_passes,
            'enabled_passes': enabled_passes,
            'disabled_passes': total_passes - enabled_passes,
            'total_executions': self._total_executions,
            'successful_executions': successful_executions,
            'failed_executions': self._total_executions - successful_executions,
            'overall_success_rate': successful_executions / max(1, self._total_executions),
            'total_execution_time': self._total_execution_time,
            'average_execution_time': self._total_execution_time / max(1, self._total_executions),
            'execution_history_size': len(self._execution_history)
        }

    def reset_statistics(self):
        """Reset all execution statistics."""
        with self._execution_lock:
            for exec_info in self._passes.values():
                exec_info.execution_count = 0
                exec_info.success_count = 0
                exec_info.total_execution_time = 0.0
                exec_info.last_execution_time = None
                exec_info.last_result = None

            self._total_executions = 0
            self._total_execution_time = 0.0
            self._execution_history.clear()

            self.logger.info("Reset all execution statistics")

    def validate_dependencies(self) -> List[Dict[str, Any]]:
        """
        Validate pass dependencies and detect issues.

        Returns:
            List of validation issues found
        """
        issues = []

        for pass_name, exec_info in self._passes.items():
            pass_instance = exec_info.pass_instance

            # Check dependencies
            for dep_pass in pass_instance.config.dependencies:
                if dep_pass not in self._passes:
                    issues.append({
                        'type': 'missing_dependency',
                        'pass': pass_name,
                        'dependency': dep_pass,
                        'message': f"Pass {pass_name} depends on missing pass {dep_pass}"
                    })

            # Check mutual exclusivity
            for exclusive_pass in pass_instance.config.mutually_exclusive:
                if exclusive_pass not in self._passes:
                    issues.append({
                        'type': 'missing_exclusive',
                        'pass': pass_name,
                        'exclusive': exclusive_pass,
                        'message': f"Pass {pass_name} has exclusive constraint on missing pass {exclusive_pass}"
                    })

        return issues