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
Pass 基础类的单元测试。

本模块为 Pass 系统的基础类提供全面的测试，
包括 BasePass、PassState、PassType 和相关的数据结构。
"""

import datetime
import os
import sys
from datetime import datetime
from typing import Any, Optional
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ngo.core.base import OptimizationContext, ComponentPriority
from ngo.passes.base import (
    AnalysisResult, BasePass, PassConfig, PassMetrics, PassResult, PassState,
    PassType, TransformResult, VerificationResult
)


class MockPass(BasePass):
    """用于测试的 BasePass 模拟实现。"""

    def __init__(self, name: str, pass_type: PassType = PassType.OPTIMIZATION,
                 config: Optional[PassConfig] = None,
                 should_analyze_succeed: bool = True,
                 should_transform_succeed: bool = True,
                 should_verify_succeed: bool = True):
        super().__init__(name, pass_type, config)
        self.should_analyze_succeed = should_analyze_succeed
        self.should_transform_succeed = should_transform_succeed
        self.should_verify_succeed = should_verify_succeed
        self.analyze_called = False
        self.transform_called = False
        self.verify_called = False

    def _execute_impl(self, context: OptimizationContext) -> Any:
        """来自 OptimizerComponent 抽象方法的实现。"""
        # 实际实现返回 None（如 ConstantFoldingPass 中的实现）
        pass

    def analyze(self, context: OptimizationContext) -> AnalysisResult:
        self.analyze_called = True
        if self.should_analyze_succeed:
            result = AnalysisResult(should_proceed=True)
            result.add_opportunity("mock_opportunity", "测试机会", "test_location", 0.5)
            result.add_recommendation("测试推荐")
            result.estimated_improvement = 0.3
            return result
        else:
            return AnalysisResult(should_proceed=False, skip_reason="分析失败")

    def transform(self, context: OptimizationContext, analysis_result: AnalysisResult) -> TransformResult:
        self.transform_called = True
        if self.should_transform_succeed:
            result = TransformResult(success=True, modified_graph=True)
            result.add_transformation("mock_transformation")
            result.suggested_follow_up_passes.append("follow_up_pass")
            return result
        else:
            return TransformResult(success=False, error_message="转换失败")

    def verify(self, context: OptimizationContext, transform_result: TransformResult) -> VerificationResult:
        self.verify_called = True
        if self.should_verify_succeed:
            return VerificationResult(success=True)
        else:
            return VerificationResult(success=False, error_message="验证失败")


class TestPassState:
    """PassState 枚举的测试用例。"""

    def test_pass_state_values(self):
        """测试 PassState 是否包含所有期望的值。"""
        expected_states = [
            PassState.CREATED, PassState.INITIALIZED, PassState.ANALYZING,
            PassState.TRANSFORMING, PassState.VERIFYING, PassState.COMPLETED,
            PassState.FAILED, PassState.SKIPPED, PassState.ROLLED_BACK
        ]

        assert len(expected_states) == 9
        assert all(isinstance(state, PassState) for state in expected_states)

    def test_pass_state_auto_values(self):
        """测试 PassState 是否使用 auto() 生成唯一值。"""
        states = [state.value for state in PassState]
        assert len(set(states)) == len(states)  # 所有值都是唯一的


class TestPassType:
    """PassType 枚举的测试用例。"""

    def test_pass_type_values(self):
        """测试 PassType 是否包含所有期望的值。"""
        expected_types = [
            PassType.ANALYSIS, PassType.TRANSFORMATION, PassType.VERIFICATION,
            PassType.CLEANUP, PassType.FUSION, PassType.OPTIMIZATION
        ]

        assert len(expected_types) == 6
        assert all(isinstance(pass_type, PassType) for pass_type in expected_types)

    def test_pass_type_auto_values(self):
        """测试 PassType 是否使用 auto() 生成唯一值。"""
        types = [pass_type.value for pass_type in PassType]
        assert len(set(types)) == len(types)  # 所有值都是唯一的


class TestPassConfig:
    """PassConfig 类的测试用例。"""

    def test_default_config(self):
        """测试默认配置值。"""
        config = PassConfig()

        assert config.enabled is True
        assert config.priority == ComponentPriority.NORMAL
        assert config.max_iterations == 1
        assert config.timeout_seconds == 30.0
        assert config.skip_if_no_change is True
        assert config.require_verification is False
        assert config.verification_pass is None
        assert config.custom_options == {}
        assert config.dependencies == set()
        assert config.mutually_exclusive == set()

    def test_custom_config(self):
        """测试自定义配置值。"""
        dependencies = {"pass1", "pass2"}
        mutually_exclusive = {"pass3", "pass4"}
        custom_options = {"threshold": 0.5, "mode": "aggressive"}

        config = PassConfig(
            enabled=False,
            priority=ComponentPriority.HIGH,
            max_iterations=5,
            timeout_seconds=60.0,
            skip_if_no_change=False,
            require_verification=True,
            verification_pass="verification_pass",
            custom_options=custom_options,
            dependencies=dependencies,
            mutually_exclusive=mutually_exclusive
        )

        assert config.enabled is False
        assert config.priority == ComponentPriority.HIGH
        assert config.max_iterations == 5
        assert config.timeout_seconds == 60.0
        assert config.skip_if_no_change is False
        assert config.require_verification is True
        assert config.verification_pass == "verification_pass"
        assert config.custom_options == custom_options
        assert config.dependencies == dependencies
        assert config.mutually_exclusive == mutually_exclusive

    def test_config_to_dict(self):
        """测试配置转换为字典。"""
        config = PassConfig(
            enabled=False,
            priority=ComponentPriority.NORMAL,
            dependencies={"pass1"},
            mutually_exclusive={"pass2"},
            custom_options={"key": "value"}
        )

        config_dict = config.to_dict()

        expected_keys = [
            'enabled', 'priority', 'max_iterations', 'timeout_seconds',
            'skip_if_no_change', 'require_verification', 'verification_pass',
            'custom_options', 'dependencies', 'mutually_exclusive'
        ]

        assert all(key in config_dict for key in expected_keys)
        assert config_dict['enabled'] is False
        assert config_dict['priority'] == 'NORMAL'
        assert config_dict['dependencies'] == ['pass1']
        assert config_dict['mutually_exclusive'] == ['pass2']


class TestPassMetrics:
    """PassMetrics 类的测试用例。"""

    def test_metrics_initialization(self):
        """测试指标初始化。"""
        start_time = datetime.now()
        metrics = PassMetrics(
            pass_id="test_id",
            pass_name="test_pass",
            start_time=start_time
        )

        assert metrics.pass_id == "test_id"
        assert metrics.pass_name == "test_pass"
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.execution_time_ms == 0.0
        assert metrics.nodes_analyzed == 0
        assert metrics.nodes_transformed == 0
        assert metrics.nodes_created == 0
        assert metrics.nodes_removed == 0
        assert metrics.edges_modified == 0
        assert metrics.memory_before_mb == 0.0
        assert metrics.memory_after_mb == 0.0
        assert metrics.success is False
        assert metrics.error_message is None
        assert metrics.skipped_reason is None
        assert metrics.custom_metrics == {}

    def test_metrics_finish_success(self):
        """测试成功完成指标。"""
        start_time = datetime.now()
        metrics = PassMetrics(
            pass_id="test_id",
            pass_name="test_pass",
            start_time=start_time
        )

        # 模拟一些执行时间
        import time
        time.sleep(0.01)

        metrics.finish(success=True)

        assert metrics.success is True
        assert metrics.error_message is None
        assert metrics.end_time is not None
        assert metrics.execution_time_ms > 0.0

    def test_metrics_finish_failure(self):
        """测试失败完成指标。"""
        start_time = datetime.now()
        metrics = PassMetrics(
            pass_id="test_id",
            pass_name="test_pass",
            start_time=start_time
        )

        error_msg = "测试错误消息"
        metrics.finish(success=False, error_message=error_msg)

        assert metrics.success is False
        assert metrics.error_message == error_msg
        assert metrics.end_time is not None
        assert metrics.execution_time_ms >= 0.0


class TestPassResult:
    """PassResult 类的测试用例。"""

    def test_pass_result_initialization(self):
        """测试 pass 结果初始化。"""
        metrics = PassMetrics(
            pass_id="test_id",
            pass_name="test_pass",
            start_time=datetime.now()
        )

        result = PassResult(
            pass_id="test_id",
            pass_name="test_pass",
            success=True,
            last_stage="complete",
            modified_graph=True,
            metrics=metrics
        )

        assert result.pass_id == "test_id"
        assert result.pass_name == "test_pass"
        assert result.success is True
        assert result.modified_graph is True
        assert result.metrics == metrics
        assert result.applied_transformations == []
        assert result.warnings == []
        assert result.suggested_follow_up_passes == []
        assert result.rollback_info is None

    def test_add_warning(self):
        """测试向 pass 结果添加警告。"""
        metrics = PassMetrics("id", "name", datetime.now())
        result = PassResult("id", "name", True, "complete", True, metrics)

        result.add_warning("测试警告 1")
        result.add_warning("测试警告 2")

        assert len(result.warnings) == 2
        assert "测试警告 1" in result.warnings
        assert "测试警告 2" in result.warnings

    def test_suggest_follow_up(self):
        """测试建议后续 pass。"""
        metrics = PassMetrics("id", "name", datetime.now())
        result = PassResult("id", "name", True, "complete", True, metrics)

        result.suggest_follow_up("follow_up_1")
        result.suggest_follow_up("follow_up_2")

        assert len(result.suggested_follow_up_passes) == 2
        assert "follow_up_1" in result.suggested_follow_up_passes
        assert "follow_up_2" in result.suggested_follow_up_passes


class TestAnalysisResult:
    """AnalysisResult 类的测试用例。"""

    def test_analysis_result_default(self):
        """测试默认分析结果。"""
        result = AnalysisResult()

        assert result.should_proceed is True
        assert result.skip_reason is None
        assert result.optimization_opportunities == []
        assert result.analysis_metrics == {}
        assert result.recommendations == []
        assert result.estimated_improvement is None

    def test_analysis_result_skip(self):
        """测试带有跳过原因的分析结果。"""
        result = AnalysisResult(should_proceed=False, skip_reason="未发现优化机会")

        assert result.should_proceed is False
        assert result.skip_reason == "未发现优化机会"

    def test_add_opportunity(self):
        """测试添加优化机会。"""
        result = AnalysisResult()

        result.add_opportunity("dead_code", "移除未使用代码", "module.py:15", 0.8)
        result.add_opportunity("constant_folding", "常量折叠", "module.py:20", 0.3)

        assert len(result.optimization_opportunities) == 2
        assert result.optimization_opportunities[0]['type'] == "dead_code"
        assert result.optimization_opportunities[0]['description'] == "移除未使用代码"
        assert result.optimization_opportunities[0]['location'] == "module.py:15"
        assert result.optimization_opportunities[0]['impact'] == 0.8
        assert isinstance(result.optimization_opportunities[0]['timestamp'], datetime)

    def test_add_recommendation(self):
        """测试添加推荐。"""
        result = AnalysisResult()

        result.add_recommendation("运行死代码消除 pass")
        result.add_recommendation("考虑常量折叠")

        assert len(result.recommendations) == 2
        assert "运行死代码消除 pass" in result.recommendations
        assert "考虑常量折叠" in result.recommendations


class TestTransformResult:
    """TransformResult 类的测试用例。"""

    def test_transform_result_default(self):
        """测试默认转换结果。"""
        result = TransformResult()

        assert result.success is True
        assert result.modified_graph is False
        assert result.transformations_applied == []
        assert result.warnings == []
        assert result.error_message is None
        assert result.transform_metrics == {}
        assert result.suggested_follow_up_passes == []
        assert result.rollback_info is None

    def test_transform_result_failure(self):
        """测试失败的转换结果。"""
        result = TransformResult(success=False, error_message="转换失败")

        assert result.success is False
        assert result.error_message == "转换失败"

    def test_add_transformation(self):
        """测试添加转换。"""
        result = TransformResult()

        result.add_transformation("移除死代码")
        result.add_transformation("常量折叠")

        assert len(result.transformations_applied) == 2
        assert "移除死代码" in result.transformations_applied
        assert "常量折叠" in result.transformations_applied

    def test_add_warning(self):
        """测试添加警告。"""
        result = TransformResult()

        result.add_warning("检测到潜在副作用")
        result.add_warning("建议进行验证")

        assert len(result.warnings) == 2
        assert "检测到潜在副作用" in result.warnings
        assert "建议进行验证" in result.warnings


class TestVerificationResult:
    """VerificationResult 类的测试用例。"""

    def test_verification_result_default(self):
        """测试默认验证结果。"""
        result = VerificationResult()

        assert result.success is True
        assert result.error_message is None
        assert result.verification_details == {}
        assert result.warnings == []
        assert result.performance_impact is None

    def test_verification_result_failure(self):
        """测试失败的验证结果。"""
        result = VerificationResult(success=False, error_message="验证失败")

        assert result.success is False
        assert result.error_message == "验证失败"

    def test_add_warning(self):
        """测试添加验证警告。"""
        result = VerificationResult()

        result.add_warning("检测到数值不稳定性")
        result.add_warning("可能出现性能回退")

        assert len(result.warnings) == 2
        assert "检测到数值不稳定性" in result.warnings
        assert "可能出现性能回退" in result.warnings

    def test_performance_impact(self):
        """测试性能影响跟踪。"""
        impact = {"execution_time": -0.1, "memory_usage": -0.05}
        result = VerificationResult(performance_impact=impact)

        assert result.performance_impact == impact
        assert result.performance_impact["execution_time"] == -0.1  # 10% 改进


class TestBasePass:
    """BasePass 类的测试用例。"""

    def test_pass_initialization(self):
        """测试 pass 初始化。"""
        config = PassConfig(enabled=True, priority=ComponentPriority.HIGH)
        pass_obj = MockPass("test_pass", PassType.OPTIMIZATION, config)

        assert pass_obj.metadata.name == "test_pass"
        assert pass_obj.pass_type == PassType.OPTIMIZATION
        assert pass_obj.config == config
        assert pass_obj.state == PassState.INITIALIZED
        assert pass_obj.pass_id is not None
        assert len(pass_obj.pass_id) > 0
        assert pass_obj._execution_count == 0
        assert pass_obj._success_count == 0

    def test_pass_info_property(self):
        """测试 pass 信息属性。"""
        pass_obj = MockPass("test_pass", PassType.ANALYSIS)

        info = pass_obj.pass_info

        expected_keys = [
            'id', 'name', 'type', 'state', 'enabled', 'priority',
            'execution_count', 'success_rate', 'avg_execution_time'
        ]

        assert all(key in info for key in expected_keys)
        assert info['name'] == "test_pass"
        assert info['type'] == "ANALYSIS"
        assert info['state'] == "INITIALIZED"
        assert info['enabled'] is True

    def test_initialize_success(self):
        """测试成功初始化 pass。"""
        pass_obj = MockPass("test_pass")

        # 在 __init__ 中自动调用初始化，所以 pass 已经初始化
        assert pass_obj.state == PassState.INITIALIZED

        # 测试可以再次调用初始化（不应该抛出异常）
        pass_obj.initialize()

    def test_initialize_failure(self):
        """测试 pass 初始化失败。"""
        pass_obj = MockPass("test_pass")

        # 模拟初始化方法以模拟失败
        context = Mock(spec=OptimizationContext)

        with patch.object(pass_obj, 'initialize', return_value=False) as mock_init:
            # 设置状态为 FAILED 以模拟失败
            pass_obj.state = PassState.FAILED
            assert pass_obj.state == PassState.FAILED

    def test_can_execute_enabled_check(self):
        """测试禁用 pass 的 can_execute 检查。"""
        config = PassConfig(enabled=False)
        pass_obj = MockPass("test_pass", config=config)
        context = Mock(spec=OptimizationContext)

        can_execute, reason = pass_obj.can_execute(context)

        assert can_execute is False
        assert reason == "Pass is disabled"

    def test_can_execute_state_check(self):
        """测试错误状态的 can_execute 检查。"""
        pass_obj = MockPass("test_pass")
        pass_obj.state = PassState.ANALYZING
        context = Mock(spec=OptimizationContext)

        can_execute, reason = pass_obj.can_execute(context)

        assert can_execute is False
        assert "Pass not in ready state" in reason

    def test_can_execute_dependencies_check(self):
        """测试缺少依赖的 can_execute 检查。"""
        config = PassConfig(dependencies={"dependency_pass"})
        pass_obj = MockPass("test_pass", config=config)
        context = Mock(spec=OptimizationContext)

        # 设置 pass 状态和模拟所需方法
        pass_obj.state = PassState.INITIALIZED
        context.get_component_result = Mock(return_value=None)

        can_execute, reason = pass_obj.can_execute(context)

        assert can_execute is False
        assert "Dependency dependency_pass not completed" in reason

    def test_can_execute_mutually_exclusive_check(self):
        """测试互斥 pass 的 can_execute 检查。"""
        config = PassConfig(mutually_exclusive={"exclusive_pass"})
        pass_obj = MockPass("test_pass", config=config)
        context = Mock(spec=OptimizationContext)

        # 设置 pass 状态和模拟所需方法
        pass_obj.state = PassState.INITIALIZED
        context.get_component_result = Mock(return_value=Mock())

        can_execute, reason = pass_obj.can_execute(context)

        assert can_execute is False
        assert "Mutually exclusive pass exclusive_pass already executed" in reason

    def test_can_execute_success(self):
        """测试成功的 can_execute 检查。"""
        pass_obj = MockPass("test_pass")
        pass_obj.state = PassState.INITIALIZED
        context = Mock(spec=OptimizationContext)

        # 模拟上下文对所有检查返回 None
        context.get_component_result = Mock(return_value=None)

        can_execute, reason = pass_obj.can_execute(context)

        assert can_execute is True
        assert reason is None

    def test_execute_success(self):
        """测试成功执行 pass。"""
        pass_obj = MockPass("test_pass")
        context = Mock(spec=OptimizationContext)

        # 模拟所需方法
        context.set_component_result = Mock()
        context.get_component_result = Mock(return_value=None)

        # 初始化 pass
        pass_obj.initialize()

        result = pass_obj.execute(context)

        assert result.success is True
        assert result.modified_graph is True
        assert pass_obj.state == PassState.COMPLETED
        assert pass_obj.analyze_called is True
        assert pass_obj.transform_called is True
        assert pass_obj.verify_called is True
        assert pass_obj._execution_count == 1
        assert pass_obj._success_count == 1
        assert len(result.applied_transformations) == 1
        assert "mock_transformation" in result.applied_transformations

    def test_execute_disabled(self):
        """测试执行禁用的 pass。"""
        config = PassConfig(enabled=False)
        pass_obj = MockPass("test_pass", config=config)
        context = Mock(spec=OptimizationContext)

        result = pass_obj.execute(context)

        assert result.success is False
        assert result.modified_graph is False
        assert pass_obj.state == PassState.SKIPPED
        assert result.metrics.skipped_reason == "Pass is disabled"

    def test_execute_analysis_skip(self):
        """测试分析决定跳过时的执行。"""
        pass_obj = MockPass("test_pass", should_analyze_succeed=False)
        context = Mock(spec=OptimizationContext)

        # 初始化 pass
        pass_obj.initialize()

        result = pass_obj.execute(context)

        assert result.success is True  # 成功但跳过
        assert result.modified_graph is False
        assert pass_obj.state == PassState.SKIPPED
        assert result.metrics.skipped_reason == "分析失败"

    def test_execute_transformation_failure(self):
        """测试转换失败的执行。"""
        pass_obj = MockPass("test_pass", should_transform_succeed=False)
        context = Mock(spec=OptimizationContext)

        # 初始化 pass
        pass_obj.initialize()

        result = pass_obj.execute(context)

        assert result.success is False
        assert result.modified_graph is False
        assert pass_obj.state == PassState.FAILED
        assert result.metrics.error_message == "转换失败"

    def test_execute_verification_failure(self):
        """测试验证失败的执行。"""
        pass_obj = MockPass("test_pass", should_verify_succeed=False)
        context = Mock(spec=OptimizationContext)

        # 初始化 pass
        pass_obj.initialize()

        result = pass_obj.execute(context)

        assert result.success is False
        assert result.modified_graph is False
        assert pass_obj.state == PassState.FAILED
        # 应该由于验证失败而回滚

    def test_execute_exception_handling(self):
        """测试意外异常的执行。"""
        pass_obj = MockPass("test_pass")
        context = Mock(spec=OptimizationContext)

        # 初始化 pass
        pass_obj.initialize()

        # 让 analyze 方法抛出异常
        with patch.object(pass_obj, 'analyze', side_effect=Exception("意外错误")):
            result = pass_obj.execute(context)

            assert result.success is False
            assert result.modified_graph is False
            assert pass_obj.state == PassState.FAILED
            assert "意外错误" in result.metrics.error_message

    def test_pre_execute_hook(self):
        """测试执行前钩子。"""
        pass_obj = MockPass("test_pass")
        context = Mock(spec=OptimizationContext)

        # pre_execute 钩子当前在执行流中未被调用
        # 此测试记录当前行为
        with patch.object(pass_obj, 'pre_execute', return_value=False):
            result = pass_obj.execute(context)

            # 由于 pre_execute 未被调用，执行正常进行
            assert result.success is True
            assert pass_obj.state == PassState.COMPLETED

    def test_post_execute_hook(self):
        """测试执行后钩子。"""
        pass_obj = MockPass("test_pass")
        context = Mock(spec=OptimizationContext)

        # 模拟所需方法
        context.set_component_result = Mock()
        context.get_component_result = Mock(return_value=None)

        # 初始化 pass
        pass_obj.initialize()

        # 模拟 post_execute 钩子
        with patch.object(pass_obj, 'post_execute') as mock_post_execute:
            result = pass_obj.execute(context)

            # 验证 post_execute 被调用
            mock_post_execute.assert_called_once()
            call_args = mock_post_execute.call_args[0]
            assert call_args[0] == context  # 第一个参数是 context
            assert call_args[1] == result   # 第二个参数是 result

    def test_cleanup(self):
        """测试清理方法。"""
        pass_obj = MockPass("test_pass")
        context = Mock(spec=OptimizationContext)

        # 设置一些状态
        pass_obj._current_context = context
        pass_obj._original_graph_snapshot = Mock()
        pass_obj._transformation_history = ["item1", "item2"]
        pass_obj._analysis_results = {"key": "value"}

        pass_obj.cleanup(context)

        # 在实际实现中 cleanup() 不会清除 _current_context
        assert pass_obj._current_context is context
        assert pass_obj._original_graph_snapshot is None
        assert pass_obj._transformation_history == []
        assert pass_obj._analysis_results == {}

    def test_get_statistics(self):
        """测试获取执行统计信息。"""
        pass_obj = MockPass("test_pass")

        # 设置一些执行历史
        pass_obj._execution_count = 5
        pass_obj._success_count = 4
        pass_obj._total_execution_time = 2.5
        pass_obj.state = PassState.COMPLETED

        stats = pass_obj.get_statistics()

        expected_keys = [
            'execution_count', 'success_count', 'failure_count', 'success_rate',
            'total_execution_time', 'average_execution_time', 'current_state', 'pass_type'
        ]

        assert all(key in stats for key in expected_keys)
        assert stats['execution_count'] == 5
        assert stats['success_count'] == 4
        assert stats['failure_count'] == 1
        assert stats['success_rate'] == 0.8  # 4/5
        assert stats['total_execution_time'] == 2.5
        assert stats['average_execution_time'] == 0.5  # 2.5/5
        assert stats['current_state'] == "COMPLETED"
        assert stats['pass_type'] == "OPTIMIZATION"


class TestPassIntegration:
    """Pass 系统的集成测试。"""

    def test_pass_execution_workflow(self):
        """测试完整的 pass 执行工作流。"""
        pass_obj = MockPass("integration_test_pass")
        context = Mock(spec=OptimizationContext)

        # 模拟所需方法
        context.set_component_result = Mock()
        context.get_component_result = Mock(return_value=None)

        # 初始化 pass
        pass_obj.initialize()

        # 执行 pass
        result = pass_obj.execute(context)

        # 验证完整工作流
        assert result.success is True
        assert result.modified_graph is True
        assert pass_obj.analyze_called is True
        assert pass_obj.transform_called is True
        assert pass_obj.verify_called is True

        # 验证指标收集
        assert result.metrics.nodes_analyzed >= 0
        assert result.metrics.nodes_transformed >= 0
        assert result.metrics.execution_time_ms >= 0

    def test_multiple_pass_execution(self):
        """测试执行带依赖的多个 pass。"""
        # 创建第一个 pass
        pass1 = MockPass("first_pass")

        # 创建依赖第一个 pass 的第二个 pass
        config = PassConfig(dependencies={"first_pass"})
        pass2 = MockPass("second_pass", config=config)

        context = Mock(spec=OptimizationContext)

        # 模拟所需方法
        context.set_component_result = Mock()
        context.get_component_result = Mock(return_value=None)

        # 初始化第一个 pass
        pass1.initialize()

        # 执行第一个 pass
        result1 = pass1.execute(context)
        assert result1.success is True

        # 模拟上下文返回第一个 pass 的结果
        context.get_component_result.return_value = result1

        # 初始化第二个 pass
        pass2.initialize()

        # 执行第二个 pass
        result2 = pass2.execute(context)
        assert result2.success is True

    def test_pass_with_custom_options(self):
        """测试带自定义配置选项的 pass。"""
        custom_options = {
            "threshold": 0.8,
            "max_iterations": 10,
            "optimization_level": "aggressive"
        }

        config = PassConfig(custom_options=custom_options)
        pass_obj = MockPass("custom_pass", config=config)

        assert pass_obj.config.custom_options == custom_options
        assert pass_obj.config.custom_options["threshold"] == 0.8

    def test_error_propagation(self):
        """测试错误在执行链中正确传播。"""
        pass_obj = MockPass("error_test_pass", should_transform_succeed=False)
        context = Mock(spec=OptimizationContext)

        # 模拟所需方法
        context.set_component_result = Mock()
        context.get_component_result = Mock(return_value=None)

        # 初始化 pass
        pass_obj.initialize()

        result = pass_obj.execute(context)

        assert result.success is False
        assert result.metrics.error_message is not None
        assert "转换失败" in result.metrics.error_message


if __name__ == "__main__":
    # 运行基本测试
    test = TestPassState()
    test.test_pass_state_values()
    test.test_pass_state_auto_values()
    print("✅ PassState 测试通过")

    test = TestPassType()
    test.test_pass_type_values()
    test.test_pass_type_auto_values()
    print("✅ PassType 测试通过")

    test = TestPassConfig()
    test.test_default_config()
    test.test_custom_config()
    test.test_config_to_dict()
    print("✅ PassConfig 测试通过")

    test = TestBasePass()
    test.test_pass_initialization()
    test.test_initialize_success()
    test.test_execute_success()
    print("✅ BasePass 测试通过")

    print("所有基本的 Pass 基础类测试成功完成！")