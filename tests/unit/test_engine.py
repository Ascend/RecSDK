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
Unit tests for optimization engine.

Tests the simplified OptimizationEngine and ExecutionResult functionality.
"""

from typing import Any
from unittest.mock import Mock, patch

from ngo.core.base import (
    ComponentMetadata, ComponentPriority, OptimizerComponent
)
from ngo.core.engine import (
    EngineError, ExecutionError, ExecutionResult, OptimizationEngine
)
from ngo.core.unified_registry import UnifiedRegistry, get_registry, clear_registry


class MockOptimizerComponent(OptimizerComponent):
    """Mock optimizer component for testing."""

    def __init__(self, metadata: ComponentMetadata, **kwargs):
        super().__init__(metadata)
        self.executed = False
        self.should_fail = False

    def initialize(self) -> None:
        """Initialize the component."""
        # 简单的初始化实现
        pass

    def execute(self, context) -> Any:
        """Execute the component."""
        if self.should_fail:
            raise RuntimeError("Mock component failure")
        self.executed = True
        return {"success": True, "component": self.metadata.name}

    def _execute_impl(self, context) -> Any:
        """Internal implementation of component execution."""
        if self.should_fail:
            raise RuntimeError("Mock component failure")
        self.executed = True
        return {"success": True, "component": self.metadata.name}


class TestExecutionResult:
    """Test ExecutionResult functionality."""

    def test_execution_result_creation(self):
        """Test execution result creation."""
        result = ExecutionResult(
            success=True,
            execution_time=0.1,
            components_executed=2
        )

        assert result.success is True
        assert result.execution_time == 0.1
        assert result.components_executed == 2
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_execution_result_with_errors(self):
        """Test execution result with errors."""
        result = ExecutionResult(
            success=False,
            execution_time=0.2,
            components_executed=1,
            errors=["Component failed"]
        )

        assert result.success is False
        assert len(result.errors) == 1
        assert "Component failed" in result.errors


class TestOptimizationEngine:
    """Test OptimizationEngine functionality."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        registry = UnifiedRegistry()
        engine = OptimizationEngine(registry)

        assert engine.registry is registry
        assert engine.context is not None
        assert engine.performance_metrics["total_executions"] == 0

    def test_engine_with_default_registry(self):
        """Test engine with default registry."""
        engine = OptimizationEngine()

        assert engine.registry is not None
        assert isinstance(engine.registry, UnifiedRegistry)

    def test_engine_execute_no_components(self):
        """Test engine execution with no components."""
        # 清除全局注册表，然后创建干净的注册器
        clear_registry()
        engine = OptimizationEngine()

        result = engine.execute()

        assert result.success is True
        assert result.components_executed == 0
        assert result.execution_time > 0

    def test_engine_execute_with_mock_component(self):
        """Test engine execution with mock component."""
        # Create mock component
        metadata = ComponentMetadata(
            name="test_component",
            version="1.0.0",
            description="Test component"
        )
        mock_component = MockOptimizerComponent(metadata)

        # Register component directly in registrations
        registry = UnifiedRegistry()
        from ngo.core.unified_registry import RegistrationInfo, RegistrationPhase
        registration_info = RegistrationInfo(
            name="test_component",
            component_class=MockOptimizerComponent,
            component_type="pass",
            enabled=True,
            priority=5,
            phase=RegistrationPhase.BOTH,
            metadata=metadata,
            instance=mock_component
        )
        registry._registrations["test_component"] = registration_info
        registry._pass_registrations["test_component"] = registration_info

        # Create engine and execute
        engine = OptimizationEngine(registry)
        result = engine.execute(["test_component"])

        assert result.success is True
        assert result.components_executed == 1
        assert mock_component.executed is True

    def test_engine_execute_with_failing_component(self):
        """Test engine execution with failing component."""
        # Create failing mock component
        metadata = ComponentMetadata(
            name="failing_component",
            version="1.0.0",
            description="Failing test component"
        )
        mock_component = MockOptimizerComponent(metadata)
        mock_component.should_fail = True

        # Register component directly in registrations
        registry = UnifiedRegistry()
        from ngo.core.unified_registry import RegistrationInfo, RegistrationPhase
        registration_info = RegistrationInfo(
            name="failing_component",
            component_class=MockOptimizerComponent,
            component_type="pass",
            enabled=True,
            priority=5,
            phase=RegistrationPhase.BOTH,
            metadata=metadata,
            instance=mock_component
        )
        registry._registrations["failing_component"] = registration_info
        registry._pass_registrations["failing_component"] = registration_info

        # Create engine and execute
        engine = OptimizationEngine(registry)
        result = engine.execute(["failing_component"])

        assert result.success is False
        assert result.components_executed == 0
        assert len(result.errors) == 1
        assert "Mock component failure" in result.errors[0]

    def test_engine_execute_with_config(self):
        """Test engine execution with configuration."""
        # 清除全局注册表以避免其他测试的失败组件影响
        clear_registry()
        engine = OptimizationEngine()

        config = {"test_key": "test_value"}
        result = engine.execute(config=config)

        assert result.success is True
        # Check that config was applied to context
        assert engine.context.get_config("execution_config") == config

    def test_engine_reset(self):
        """Test engine reset."""
        engine = OptimizationEngine()

        # Set some context data
        engine.context.set_data("test", "value")

        # Reset engine
        engine.reset()

        # Context should be cleared
        assert engine.context.get_data("test") is None

    def test_engine_get_status(self):
        """Test engine status."""
        engine = OptimizationEngine()

        status = engine.get_status()

        assert "total_executions" in status
        assert "successful_executions" in status
        assert "failed_executions" in status
        assert "average_execution_time" in status
        assert "registry_components" in status

    def test_engine_str_representation(self):
        """Test engine string representation."""
        engine = OptimizationEngine()

        str_repr = str(engine)
        assert "OptimizationEngine" in str_repr
        assert "组件数" in str_repr

    def test_engine_repr(self):
        """Test engine repr."""
        engine = OptimizationEngine()

        repr_str = repr(engine)
        assert "OptimizationEngine" in repr_str
        assert "执行次数" in repr_str


class TestEnginePerformanceMetrics:
    """Test engine performance metrics."""

    def test_performance_metrics_update(self):
        """Test performance metrics update."""
        engine = OptimizationEngine()

        # Execute successful operation
        result = ExecutionResult(
            success=True,
            execution_time=0.1,
            components_executed=1
        )
        engine._update_performance_metrics(result)

        metrics = engine.performance_metrics
        assert metrics["total_executions"] == 1
        assert metrics["successful_executions"] == 1
        assert metrics["failed_executions"] == 0
        assert metrics["total_execution_time"] == 0.1
        assert metrics["average_execution_time"] == 0.1

    def test_performance_metrics_with_failure(self):
        """Test performance metrics with failure."""
        engine = OptimizationEngine()

        # Execute failed operation
        result = ExecutionResult(
            success=False,
            execution_time=0.2,
            components_executed=0
        )
        engine._update_performance_metrics(result)

        metrics = engine.performance_metrics
        assert metrics["total_executions"] == 1
        assert metrics["successful_executions"] == 0
        assert metrics["failed_executions"] == 1
        assert metrics["total_execution_time"] == 0.2
        assert metrics["average_execution_time"] == 0.2

    def test_performance_metrics_average_calculation(self):
        """Test average execution time calculation."""
        engine = OptimizationEngine()

        # Execute multiple operations
        for i in range(3):
            result = ExecutionResult(
                success=True,
                execution_time=0.1 * (i + 1),
                components_executed=1
            )
            engine._update_performance_metrics(result)

        metrics = engine.performance_metrics
        assert metrics["total_executions"] == 3
        assert abs(metrics["average_execution_time"] - 0.2) < 0.0001  # (0.1 + 0.2 + 0.3) / 3


class TestEngineErrorHandling:
    """Test engine error handling."""

    def test_execution_error_creation(self):
        """Test ExecutionError creation."""
        error = ExecutionError("Test error")

        assert str(error) == "Test error"
        assert isinstance(error, EngineError)

    def test_engine_error_creation(self):
        """Test EngineError creation."""
        error = EngineError("Test engine error")

        assert str(error) == "Test engine error"

    def test_collect_execution_metrics(self):
        """Test execution metrics collection."""
        engine = OptimizationEngine()

        metrics = engine._collect_execution_metrics()

        assert "engine_metrics" in metrics
        assert "registry_stats" in metrics
        assert "context_stats" in metrics
        assert metrics["engine_metrics"]["total_executions"] == 0