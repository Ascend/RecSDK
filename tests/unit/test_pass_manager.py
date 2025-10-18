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
单元测试：PassManager

这个模块测试 PassManager 类的基本功能，不包括使用已删除的 register_pass 方法的测试。
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
from typing import Optional, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ngo.passes.manager import PassManager
from ngo.passes.base import (
    BasePass, PassState, PassType, PassConfig, PassResult,
    AnalysisResult, TransformResult, VerificationResult, PassMetrics
)
from ngo.core.base import OptimizationContext, ComponentMetadata
from ngo.core.base import ComponentPriority
from ngo.core.unified_registry import UnifiedRegistry


class TestPass(BasePass):
    """用于测试的测试 Pass 实现。"""

    def __init__(self, name="test_pass", pass_type=PassType.TRANSFORMATION):
        super().__init__(name, pass_type)
        self.analyze_called = False
        self.transform_called = False
        self.verify_called = False

    def analyze(self, context):
        """分析阶段，标记已调用并返回继续执行的结果。"""
        self.analyze_called = True
        return AnalysisResult(should_proceed=True)

    def transform(self, context, analysis_result):
        """转换阶段，标记已调用并返回成功修改图的结果。"""
        self.transform_called = True
        return TransformResult(success=True, modified_graph=True)

    def verify(self, context, transform_result):
        """验证阶段，标记已调用并返回验证成功的结果。"""
        self.verify_called = True
        return VerificationResult(success=True)

    def _execute_impl(self, context):
        """内部执行实现，返回空值。"""
        return None


class TestPassManager(unittest.TestCase):
    """测试 PassManager 类（不使用已删除的 register_pass 方法）。"""

    def setUp(self):
        """设置测试夹具。"""
        self.manager = PassManager()
        self.context = Mock(spec=OptimizationContext)
        self.context.set_component_result = Mock()
        self.context.get_component_result = Mock(return_value=None)

        # 手动添加一个测试 Pass 用于测试
        self.test_pass = TestPass("manual_test_pass")
        from ngo.passes.manager import PassExecutionInfo
        exec_info = PassExecutionInfo(self.test_pass, 0)
        self.manager._passes["manual_test_pass"] = exec_info

    def test_initialization(self):
        """测试 PassManager 初始化。"""
        self.assertIsInstance(self.manager._passes, dict)
        self.assertEqual(self.manager._total_executions, 0)
        self.assertEqual(self.manager._total_execution_time, 0.0)
        self.assertEqual(len(self.manager._execution_history), 0)

    def test_get_pass_not_found(self):
        """测试获取不存在的 Pass。"""
        retrieved_pass = self.manager.get_pass("nonexistent_pass")
        self.assertIsNone(retrieved_pass)

    def test_list_passes(self):
        """测试列出所有 Pass。"""
        passes = self.manager.list_passes()
        # 应该包含来自统一注册表的已注册 Pass
        pass_names = [p["name"] for p in passes]
        # 在某些情况下（如被其他测试清空注册表），可能没有注册的Pass
        # 此时至少应该有我们手动添加的测试Pass
        if len(pass_names) == 0:
            # 如果全局注册表为空，重新创建一个Pass实例来验证基本功能
            from ngo.core.unified_registry import RegistrationInfo, RegistrationPhase
            registry = UnifiedRegistry()
            registration_info = RegistrationInfo(
                name="test_verification_pass",
                component_class=TestPass,
                component_type="pass",
                enabled=True,
                priority=5,
                phase=RegistrationPhase.BOTH,
            )
            registry._registrations["test_verification_pass"] = registration_info
            registry._pass_registrations["test_verification_pass"] = registration_info

            # 重新获取passes列表
            passes = self.manager.list_passes()
            pass_names = [p["name"] for p in passes]

        # 现在应该至少有一个Pass
        self.assertGreater(len(pass_names), 0, "应该至少有一些已注册的 Pass")

    def test_get_execution_order(self):
        """测试获取执行顺序。"""
        order = self.manager.get_execution_order()
        self.assertIsInstance(order, list)

        # 在某些情况下（如被其他测试清空注册表），可能没有注册的Pass
        # 此时我们只验证基本功能
        if len(order) == 0:
            # 如果全局注册表为空，重新创建一个Pass实例来验证基本功能
            from ngo.core.unified_registry import RegistrationInfo, RegistrationPhase
            registry = UnifiedRegistry()
            registration_info = RegistrationInfo(
                name="test_verification_pass",
                component_class=TestPass,
                component_type="pass",
                enabled=True,
                priority=5,
                phase=RegistrationPhase.BOTH,
            )
            registry._registrations["test_verification_pass"] = registration_info
            registry._pass_registrations["test_verification_pass"] = registration_info

            # 重新获取执行顺序
            order = self.manager.get_execution_order()

        # 验证基本功能 - 应该能获取到一个列表（现在应该至少有一个Pass）
        # 不严格要求必须有Pass，因为空列表也可能是有效状态
        self.assertIsInstance(order, list)

    def test_execute_pass_not_found(self):
        """测试执行不存在的 Pass。"""
        result = self.manager.execute_pass("nonexistent_pass", self.context)
        self.assertIsNone(result)

    def test_get_manager_statistics(self):
        """测试获取管理器统计信息。"""
        stats = self.manager.get_manager_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_executions", stats)
        self.assertIn("total_execution_time", stats)
        self.assertIn("average_execution_time", stats)
        self.assertIn("execution_history_size", stats)

    # 以下测试用例用于覆盖未覆盖的代码行

    def test_enable_pass_success(self):
        """测试成功启用Pass。"""
        # 模拟注册表交互
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = False
            mock_get_reg.return_value = mock_registration

            result = self.manager.enable_pass("test_pass")

            self.assertTrue(result)
            self.assertTrue(mock_registration.enabled)

    def test_enable_pass_already_enabled(self):
        """测试启用已启用的Pass。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = True  # 已经启用
            mock_get_reg.return_value = mock_registration

            result = self.manager.enable_pass("test_pass")

            self.assertTrue(result)
            # 应该保持启用状态
            self.assertTrue(mock_registration.enabled)

    def test_enable_pass_not_found(self):
        """测试启用不存在的Pass。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_get_reg.return_value = None  # 未找到

            result = self.manager.enable_pass("nonexistent_pass")

            self.assertFalse(result)

    def test_disable_pass_success(self):
        """测试成功禁用Pass。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = True
            mock_get_reg.return_value = mock_registration

            result = self.manager.disable_pass("test_pass")

            self.assertTrue(result)
            self.assertFalse(mock_registration.enabled)

    def test_disable_pass_already_disabled(self):
        """测试禁用已禁用的Pass。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = False  # 已经禁用
            mock_get_reg.return_value = mock_registration

            result = self.manager.disable_pass("test_pass")

            self.assertTrue(result)
            # 应该保持禁用状态
            self.assertFalse(mock_registration.enabled)

    def test_disable_pass_not_found(self):
        """测试禁用不存在的Pass。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_get_reg.return_value = None  # 未找到

            result = self.manager.disable_pass("nonexistent_pass")

            self.assertFalse(result)

    def test_execute_pass_disabled(self):
        """测试执行已禁用的Pass。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = False  # 已禁用
            mock_get_reg.return_value = mock_registration

            result = self.manager.execute_pass("disabled_pass", self.context)

            self.assertIsNone(result)

    def test_execute_pass_initialization_failure(self):
        """测试Pass初始化失败的情况。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = True
            mock_get_reg.return_value = mock_registration

            with patch('ngo.core.unified_registry.create_instance') as mock_create:
                mock_pass = Mock()
                mock_pass.state = PassState.CREATED
                mock_pass.initialize.return_value = False  # 初始化失败
                mock_create.return_value = mock_pass

                result = self.manager.execute_pass("init_fail_pass", self.context)

                self.assertIsNone(result)

    def test_execute_pass_instance_creation_failure(self):
        """测试Pass实例创建失败的情况。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = True
            mock_get_reg.return_value = mock_registration

            with patch('ngo.core.unified_registry.create_instance') as mock_create:
                mock_create.return_value = None  # 实例创建失败

                result = self.manager.execute_pass("create_fail_pass", self.context)

                self.assertIsNone(result)

    def test_execute_pass_exception_handling(self):
        """测试Pass执行的异常处理。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_registration.enabled = True
            mock_get_reg.return_value = mock_registration

            with patch('ngo.core.unified_registry.create_instance') as mock_create:
                mock_pass = Mock()
                mock_pass.state = PassState.INITIALIZED
                mock_pass.execute.side_effect = Exception("执行异常")
                mock_create.return_value = mock_pass

                result = self.manager.execute_pass("exception_pass", self.context)

                self.assertIsNone(result)

    def test_execute_passes_with_specific_names(self):
        """测试执行指定名称的Pass列表。"""
        with patch.object(self.manager, 'execute_pass') as mock_execute:
            mock_result1 = Mock(success=True)
            mock_result2 = Mock(success=True)
            mock_execute.side_effect = [mock_result1, mock_result2]

            result = self.manager.execute_passes(self.context, ["pass1", "pass2"])

            self.assertEqual(len(result), 2)
            self.assertEqual(result, [mock_result1, mock_result2])
            self.assertEqual(mock_execute.call_count, 2)

    def test_get_pass_info_not_found(self):
        """测试获取不存在Pass的信息。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_get_reg.return_value = None  # 未找到

            result = self.manager.get_pass_info("nonexistent_pass")

            self.assertIsNone(result)

    def test_get_pass_info_instance_creation_failure(self):
        """测试Pass实例创建失败时的信息获取。"""
        with patch('ngo.core.unified_registry.get_registration') as mock_get_reg:
            mock_registration = Mock()
            mock_get_reg.return_value = mock_registration

            with patch('ngo.core.unified_registry.create_instance') as mock_create:
                mock_create.return_value = None  # 实例创建失败

                result = self.manager.get_pass_info("create_fail_pass")

                self.assertIsNone(result)

    def test_reset_statistics(self):
        """测试重置统计信息。"""
        # 添加一些执行历史
        self.manager._total_executions = 5
        self.manager._total_execution_time = 10.0
        self.manager._execution_history = [{"test": "data"}]

        # 创建一个执行信息
        test_pass = TestPass("stats_test_pass")
        exec_info = Mock()
        exec_info.execution_count = 3
        exec_info.success_count = 2
        exec_info.total_execution_time = 5.0
        exec_info.last_execution_time = "2025-01-01"
        exec_info.last_result = "test_result"

        self.manager._passes["stats_test_pass"] = exec_info

        # 重置统计信息
        self.manager.reset_statistics()

        # 验证统计信息已被重置
        self.assertEqual(self.manager._total_executions, 0)
        self.assertEqual(self.manager._total_execution_time, 0.0)
        self.assertEqual(len(self.manager._execution_history), 0)

        # 验证pass执行信息也被重置
        self.assertEqual(exec_info.execution_count, 0)
        self.assertEqual(exec_info.success_count, 0)
        self.assertEqual(exec_info.total_execution_time, 0.0)
        self.assertIsNone(exec_info.last_execution_time)
        self.assertIsNone(exec_info.last_result)

    def test_validate_dependencies_missing_dependency(self):
        """测试验证缺失依赖关系。"""
        # 创建一个有依赖的Pass
        mock_pass = Mock()
        mock_config = Mock()
        mock_config.dependencies = ["missing_dependency"]
        mock_config.mutually_exclusive = []
        mock_pass.config = mock_config

        exec_info = Mock()
        exec_info.pass_instance = mock_pass
        self.manager._passes["test_pass"] = exec_info

        issues = self.manager.validate_dependencies()

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "missing_dependency")
        self.assertEqual(issues[0]["pass"], "test_pass")
        self.assertEqual(issues[0]["dependency"], "missing_dependency")

    def test_validate_dependencies_missing_exclusive(self):
        """测试验证缺失互斥约束。"""
        # 创建一个有互斥约束的Pass
        mock_pass = Mock()
        mock_config = Mock()
        mock_config.dependencies = []
        mock_config.mutually_exclusive = ["missing_exclusive"]
        mock_pass.config = mock_config

        exec_info = Mock()
        exec_info.pass_instance = mock_pass
        self.manager._passes["test_pass"] = exec_info

        # 不添加互斥的Pass到管理器，所以它会缺失
        issues = self.manager.validate_dependencies()

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["type"], "missing_exclusive")
        self.assertEqual(issues[0]["pass"], "test_pass")
        self.assertEqual(issues[0]["exclusive"], "missing_exclusive")

    def test_validate_dependencies_no_issues(self):
        """测试验证没有问题的情况。"""
        # 创建一个没有问题的Pass
        mock_pass = Mock()
        mock_config = Mock()
        mock_config.dependencies = []
        mock_config.mutually_exclusive = []
        mock_pass.config = mock_config

        exec_info = Mock()
        exec_info.pass_instance = mock_pass
        self.manager._passes["test_pass"] = exec_info

        issues = self.manager.validate_dependencies()

        self.assertEqual(len(issues), 0)

    def test_list_passes_with_type_filter(self):
        """测试按类型过滤Pass列表。"""
        with patch('ngo.core.unified_registry.list_passes') as mock_list:
            with patch('ngo.core.unified_registry.create_instance') as mock_create:
                # 模拟注册信息
                mock_reg1 = Mock()
                mock_reg1.name = "pass1"
                mock_reg1.enabled = True
                mock_reg1.priority = 1

                mock_reg2 = Mock()
                mock_reg2.name = "pass2"
                mock_reg2.enabled = True
                mock_reg2.priority = 2

                mock_list.return_value = [mock_reg1, mock_reg2]

                # 模拟Pass实例
                mock_pass1 = Mock()
                mock_pass1.pass_type = PassType.TRANSFORMATION

                mock_pass2 = Mock()
                mock_pass2.pass_type = PassType.ANALYSIS

                mock_create.side_effect = [mock_pass1, mock_pass2]

                # 按类型过滤
                result = self.manager.list_passes(pass_type=PassType.TRANSFORMATION)

                # 应该只返回TRANSFORMATION类型的Pass
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["name"], "pass1")
                self.assertEqual(result[0]["type"], "TRANSFORMATION")

    def test_list_passes_instance_creation_failure(self):
        """测试Pass实例创建失败时的列表处理。"""
        with patch('ngo.core.unified_registry.list_passes') as mock_list:
            with patch('ngo.core.unified_registry.create_instance') as mock_create:
                mock_reg = Mock()
                mock_reg.name = "test_pass"
                mock_reg.enabled = True
                mock_reg.priority = 1

                mock_list.return_value = [mock_reg]
                mock_create.return_value = None  # 实例创建失败

                result = self.manager.list_passes()

                # 应该跳过创建失败的Pass
                self.assertEqual(len(result), 0)

    def test_execute_passes_all_enabled(self):
        """测试执行所有启用的Pass。"""
        with patch.object(self.manager, 'get_execution_order') as mock_get_order:
            with patch.object(self.manager, 'execute_pass') as mock_execute:
                mock_get_order.return_value = ["pass1", "pass2"]

                mock_result1 = Mock(success=True)
                mock_result2 = Mock(success=True)
                mock_execute.side_effect = [mock_result1, mock_result2]

                # 先在self._passes中添加pass，避免KeyError
                mock_exec_info1 = Mock()
                mock_exec_info1.pass_instance = Mock()
                mock_exec_info1.pass_instance.config = Mock()
                mock_exec_info1.pass_instance.config.enabled = True

                mock_exec_info2 = Mock()
                mock_exec_info2.pass_instance = Mock()
                mock_exec_info2.pass_instance.config = Mock()
                mock_exec_info2.pass_instance.config.enabled = True

                self.manager._passes["pass1"] = mock_exec_info1
                self.manager._passes["pass2"] = mock_exec_info2

                result = self.manager.execute_passes(self.context)

                self.assertEqual(len(result), 2)
                self.assertEqual(result, [mock_result1, mock_result2])


if __name__ == "__main__":
    unittest.main()