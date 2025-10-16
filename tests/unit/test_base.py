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
Unit tests for base optimization components.

Tests the OptimizerComponent abstract base class and OptimizationContext class.
"""

import logging
import time
from unittest.mock import Mock, patch

import pytest

from ngo.core.base import (
    ComponentMetadata,
    ComponentPriority,
    ComponentState,
    OptimizationContext,
    OptimizerComponent,
)


class MockOptimizerComponent(OptimizerComponent):
    """Mock implementation of OptimizerComponent for testing."""

    def __init__(self, metadata: ComponentMetadata):
        super().__init__(metadata)
        self.initialized = False
        self.executed = False

    def initialize(self) -> None:
        """Initialize the mock component."""
        super().initialize()
        self.initialized = True

    def execute(self, context: OptimizationContext) -> str:
        """Execute the mock component."""
        return super().execute(context)

    def _execute_impl(self, context: OptimizationContext) -> str:
        """Execute the mock component."""
        self.executed = True
        return f"Mock execution result for {self.metadata.name}"


class TestComponentMetadata:
    """Test cases for ComponentMetadata."""

    def test_metadata_creation(self):
        """Test creating ComponentMetadata with minimal arguments."""
        metadata = ComponentMetadata(name="test", version="1.0.0")
        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.dependencies == []
        assert metadata.tags == set()
        assert metadata.config_schema is None

    def test_metadata_with_all_fields(self):
        """Test creating ComponentMetadata with all fields."""
        metadata = ComponentMetadata(
            name="test",
            version="1.0.0",
            description="Test component",
            author="Test Author",
            dependencies=["dep1", "dep2"],
            tags={"test", "mock"},
            config_schema={"type": "object"},
        )
        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test component"
        assert metadata.author == "Test Author"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.tags == {"test", "mock"}
        assert metadata.config_schema == {"type": "object"}


class TestOptimizerComponent:
    """Test cases for OptimizerComponent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metadata = ComponentMetadata(name="test-component", version="1.0.0")
        self.component = MockOptimizerComponent(self.metadata)

    def test_component_initialization(self):
        """Test component initialization."""
        assert self.component.metadata.name == "test-component"
        assert self.component.metadata.version == "1.0.0"
        assert self.component.state == ComponentState.CREATED
        assert self.component.priority == ComponentPriority.NORMAL
        assert self.component.config == {}
        assert self.component.id is not None
        assert len(self.component.id) == 36  # UUID length
        assert self.component.execution_stats["execution_count"] == 0
        assert self.component.execution_stats["error_count"] == 0

    def test_component_properties(self):
        """Test component property access."""
        # Test priority property
        self.component.priority = ComponentPriority.HIGH
        assert self.component.priority == ComponentPriority.HIGH

        # Test execution stats
        stats = self.component.execution_stats
        assert "created_time" in stats
        assert "last_execution_time" in stats
        assert "execution_count" in stats
        assert "error_count" in stats
        assert "uptime" in stats
        assert stats["execution_count"] == 0
        assert stats["error_count"] == 0

    def test_set_config_valid(self):
        """Test setting valid configuration."""
        config = {"param1": "value1", "param2": 42}
        self.component.set_config(config)
        assert self.component.config == config

    def test_set_config_invalid_state(self):
        """Test setting configuration in invalid states."""
        # Test in RUNNING state
        self.component._state = ComponentState.RUNNING
        with pytest.raises(
            ValueError, match="Cannot set config in ComponentState.RUNNING state"
        ):
            self.component.set_config({"param": "value"})

        # Test in DESTROYED state
        self.component._state = ComponentState.DESTROYED
        with pytest.raises(
            ValueError, match="Cannot set config in ComponentState.DESTROYED state"
        ):
            self.component.set_config({"param": "value"})

    def test_initialize_success(self):
        """Test successful component initialization."""
        assert not self.component.initialized
        self.component.initialize()
        assert self.component.initialized
        assert self.component.state == ComponentState.INITIALIZED

    def test_initialize_already_initialized(self):
        """Test initializing already initialized component."""
        self.component.initialize()
        with pytest.raises(RuntimeError, match="Component already initialized"):
            self.component.initialize()

    def test_execute_success(self):
        """Test successful component execution."""
        self.component.initialize()
        context = OptimizationContext()

        result = self.component.execute(context)

        assert self.component.executed
        assert result == "Mock execution result for test-component"
        assert self.component.state == ComponentState.INITIALIZED
        assert self.component.execution_stats["execution_count"] == 1
        assert self.component.execution_stats["last_execution_time"] is not None

    def test_execute_not_initialized(self):
        """Test executing component without initialization."""
        context = OptimizationContext()
        with pytest.raises(RuntimeError, match="Component not ready for execution"):
            self.component.execute(context)

    def test_execute_with_exception(self):
        """Test component execution with exception."""

        class ErrorComponent(MockOptimizerComponent):
            def _execute_impl(self, context: OptimizationContext) -> str:
                raise ValueError("Test error")

        error_component = ErrorComponent(self.metadata)
        error_component.initialize()
        context = OptimizationContext()

        with pytest.raises(RuntimeError, match="Execution failed: Test error"):
            error_component.execute(context)

        assert error_component.state == ComponentState.ERROR
        assert error_component.execution_stats["error_count"] == 1

    def test_reset(self):
        """Test component reset functionality."""
        self.component.initialize()
        self.component.set_config({"test": "value"})
        # Set execution count via multiple successful executions
        for i in range(5):
            self.component._execution_count += 1
        # Simulate error count
        self.component._error_count = 2

        self.component.reset()

        assert self.component.state == ComponentState.CREATED
        assert self.component.config == {}
        assert self.component.execution_stats["execution_count"] == 0
        assert self.component.execution_stats["error_count"] == 0

    def test_reset_destroyed_component(self):
        """Test resetting destroyed component."""
        self.component._state = ComponentState.DESTROYED
        with pytest.raises(RuntimeError, match="Cannot reset destroyed component"):
            self.component.reset()

    def test_destroy(self):
        """Test component destroy functionality."""
        self.component.initialize()
        self.component._state = ComponentState.RUNNING

        self.component.destroy()
        assert self.component.state == ComponentState.DESTROYED

    def test_destroy_with_exception(self):
        """Test component destroy with exception."""

        class FailingDestroyComponent(MockOptimizerComponent):
            def destroy(self):
                raise ValueError("Destroy error")

        failing_component = FailingDestroyComponent(self.metadata)

        with pytest.raises(ValueError, match="Destroy error"):
            failing_component.destroy()
        # Note: The destroy method in the base class doesn't set state to ERROR on exception
        # because it re-raises the exception, so the state remains unchanged

    def test_string_representations(self):
        """Test string representations of component."""
        str_repr = str(self.component)
        assert "test-component" in str_repr
        assert "1.0.0" in str_repr
        assert "created" in str_repr

        repr_str = repr(self.component)
        assert "MockOptimizerComponent" in repr_str
        assert "test-component" in repr_str
        assert "1.0.0" in repr_str


class TestOptimizationContext:
    """Test cases for OptimizationContext."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = OptimizationContext({"global": "config"})

    def test_context_initialization(self):
        """Test context initialization."""
        assert self.context.config == {"global": "config"}
        assert self.context.data == {}
        assert self.context.get_stats()["registered_components"] == 0

    def test_config_management(self):
        """Test configuration management."""
        # Test get_config with default
        assert self.context.get_config("nonexistent", "default") == "default"
        assert self.context.get_config("global") == "config"

        # Test set_config
        self.context.set_config("new_key", "new_value")
        assert self.context.get_config("new_key") == "new_value"

    def test_data_management(self):
        """Test data management."""
        # Test get_data with default
        assert self.context.get_data("nonexistent", "default") == "default"

        # Test set_data and get_data
        self.context.set_data("key1", "value1")
        self.context.set_data("key2", 42)
        assert self.context.get_data("key1") == "value1"
        assert self.context.get_data("key2") == 42

        
    
    def test_component_registration(self):
        """Test component registration."""
        metadata = ComponentMetadata(name="test", version="1.0.0")
        component = MockOptimizerComponent(metadata)

        # Test register
        self.context.register_component(component)
        assert self.context.get_component(component.id) is component

    
    def test_stats(self):
        """Test context statistics."""
        stats = self.context.get_stats()
        assert "uptime" in stats
        assert "start_time" in stats
        assert "registered_components" in stats
        assert "data_keys" in stats
        assert "metadata_keys" in stats

        # Add some data and check stats
        self.context.set_data("key1", "value1")

        stats = self.context.get_stats()
        assert stats["data_keys"] == 1

    def test_clear(self):
        """Test context clearing."""
        self.context.set_data("key1", "value1")

        self.context.clear()

        assert len(self.context.data) == 0
        assert len(self.context._metadata) == 0
        assert len(self.context._components) == 0

    def test_copy(self):
        """Test context copying."""
        self.context.set_data("key1", "value1")

        copied = self.context.copy()

        assert copied.config == self.context.config
        assert copied.data == self.context.data
        assert copied._metadata == self.context._metadata

        # Verify it's a deep copy
        copied.set_data("key2", "value2")
        assert copied.get_data("key2") == "value2"
        assert self.context.get_data("key2") is None

    def test_context_manager(self):
        """Test context manager functionality."""
        with OptimizationContext() as ctx:
            ctx.set_data("test_key", "test_value")
            assert ctx.get_data("test_key") == "test_value"

    def test_string_representations(self):
        """Test string representations of context."""
        str_repr = str(self.context)
        assert "OptimizationContext" in str_repr
        assert "components=0" in str_repr
        assert "data=0" in str_repr

        repr_str = repr(self.context)
        assert "OptimizationContext" in repr_str
        assert "config_keys=1" in repr_str
        assert "data_keys=0" in repr_str


class TestComponentIntegration:
    """Integration tests for components and context."""

    def test_component_execution_with_context(self):
        """Test component execution within context."""
        metadata = ComponentMetadata(name="integration-test", version="1.0.0")
        component = MockOptimizerComponent(metadata)
        context = OptimizationContext()

        # Register component with context
        context.register_component(component)

        # Initialize and execute
        component.initialize()
        result = component.execute(context)

        assert result == "Mock execution result for integration-test"
        assert component.execution_stats["execution_count"] == 1
        assert context.get_component(component.id) is component

    def test_multiple_components_in_context(self):
        """Test multiple components operating in the same context."""
        metadata1 = ComponentMetadata(name="comp1", version="1.0.0")
        metadata2 = ComponentMetadata(name="comp2", version="1.0.0")

        component1 = MockOptimizerComponent(metadata1)
        component2 = MockOptimizerComponent(metadata2)

        context = OptimizationContext()
        context.register_component(component1)
        context.register_component(component2)

        # Initialize both components
        component1.initialize()
        component2.initialize()

        # Execute both components
        result1 = component1.execute(context)
        result2 = component2.execute(context)

        assert result1 == "Mock execution result for comp1"
        assert result2 == "Mock execution result for comp2"
        assert component1.execution_stats["execution_count"] == 1
        assert component2.execution_stats["execution_count"] == 1

        # Verify both components are registered
        assert context.get_component(component1.id) is component1
        assert context.get_component(component2.id) is component2
