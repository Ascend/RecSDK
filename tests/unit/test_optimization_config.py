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
优化配置管理模块的单元测试。

该模块测试 OptimizationConfigManager 类的功能，包括：
- 配置文件加载和创建
- Pass 和 Pattern 配置管理
- 配置同步和应用
- 全局配置管理器
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ngo.core.config.optimization_config import (
    OptimizationConfigManager,
    PassConfigEntry,
    PatternConfigEntry,
    get_global_optimization_config_manager,
    sync_config_with_registry
)
from ngo.core.config.manager import ConfigManager, ConfigError
from ngo.core.base import ComponentMetadata, ComponentPriority
from ngo.core.unified_registry import UnifiedRegistry, RegistrationInfo, RegistrationPhase, get_registry
from ngo.passes.base import BasePass
from ngo.patterns.base import BasePattern


class MockPass(BasePass):
    """用于测试的 Mock Pass。"""

    def __init__(self, name="MockPass"):
        from ngo.passes.base import PassType
        super().__init__(name, PassType.OPTIMIZATION)

    def analyze(self, context):
        from ngo.passes.base import AnalysisResult
        return AnalysisResult(should_proceed=True)

    def transform(self, context, analysis_result):
        from ngo.passes.base import TransformResult
        return TransformResult(success=True, modified_graph=False)

    def verify(self, context, transform_result):
        from ngo.passes.base import VerificationResult
        return VerificationResult(success=True)

    def execute(self, context):
        return {"result": "mock_pass_executed"}

    def initialize(self, context=None):
        return True

    def _execute_impl(self, context):
        return {"impl_result": "mock_pass_impl"}


class MockPattern(BasePattern):
    """用于测试的 Mock Pattern。"""

    def __init__(self, name="MockPattern"):
        metadata = ComponentMetadata(
            name=name,
            version="1.0.0",
            description="Mock Pattern for testing"
        )
        super().__init__(metadata)
        # 模拟真实的 pattern config 结构
        self._enabled = True

    @property
    def config(self):
        """获取组件配置。"""
        # 返回一个模拟对象，支持 enabled 属性
        class MockConfig:
            def __init__(self, pattern_ref):
                self.pattern_ref = pattern_ref

            @property
            def enabled(self):
                return self.pattern_ref._enabled

            @enabled.setter
            def enabled(self, value):
                self.pattern_ref._enabled = value

        return MockConfig(self)

    def match(self, graph_module, nodes=None):
        from ngo.patterns.base import PatternMatchResult
        return PatternMatchResult(matched=True)

    def execute(self, context):
        return {"result": "mock_pattern_executed"}

    def initialize(self, context=None):
        return True

    def _execute_impl(self, context):
        return {"impl_result": "mock_pattern_impl"}


class TestPassConfigEntry(unittest.TestCase):
    """测试 PassConfigEntry 类。"""

    def test_default_values(self):
        """测试默认值。"""
        entry = PassConfigEntry(name="test_pass")
        self.assertEqual(entry.name, "test_pass")
        self.assertTrue(entry.enabled)
        self.assertEqual(entry.priority, "NORMAL")
        self.assertEqual(entry.description, "")

    def test_custom_values(self):
        """测试自定义值。"""
        entry = PassConfigEntry(
            name="custom_pass",
            enabled=False,
            priority="HIGH",
            description="Custom description"
        )
        self.assertEqual(entry.name, "custom_pass")
        self.assertFalse(entry.enabled)
        self.assertEqual(entry.priority, "HIGH")
        self.assertEqual(entry.description, "Custom description")


class TestPatternConfigEntry(unittest.TestCase):
    """测试 PatternConfigEntry 类。"""

    def test_default_values(self):
        """测试默认值。"""
        entry = PatternConfigEntry(name="test_pattern")
        self.assertEqual(entry.name, "test_pattern")
        self.assertTrue(entry.enabled)
        self.assertEqual(entry.priority, "NORMAL")
        self.assertEqual(entry.description, "")

    def test_custom_values(self):
        """测试自定义值。"""
        entry = PatternConfigEntry(
            name="custom_pattern",
            enabled=False,
            priority="LOW",
            description="Custom description"
        )
        self.assertEqual(entry.name, "custom_pattern")
        self.assertFalse(entry.enabled)
        self.assertEqual(entry.priority, "LOW")
        self.assertEqual(entry.description, "Custom description")


class TestOptimizationConfigManager(unittest.TestCase):
    """测试 OptimizationConfigManager 类。"""

    def setUp(self):
        """设置测试夹具。"""
        # 创建临时目录用于配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = OptimizationConfigManager(
            config_dir=self.temp_dir,
            auto_sync=False  # 禁用自动同步以控制测试环境
        )

    def tearDown(self):
        """清理测试夹具。"""
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization_creates_config_dir(self):
        """测试初始化创建配置目录。"""
        self.assertTrue(self.config_manager.config_dir.exists())
        self.assertTrue(self.config_manager.config_dir.is_dir())

    def test_initialization_without_config_file(self):
        """测试没有配置文件时的初始化。"""
        # OptimizationConfigManager 会自动创建默认配置，所以配置文件应该存在
        self.assertTrue(self.config_manager.config_file.exists())

        # 应该创建默认配置
        self.assertIsNotNone(self.config_manager.config_manager.get("passes"))
        self.assertIsNotNone(self.config_manager.config_manager.get("patterns"))

    def test_initialization_with_existing_config_file(self):
        """测试使用现有配置文件初始化。"""
        # 创建配置文件
        config_content = """
[passes.test_pass]
enabled = true
priority = "HIGH"
description = "Test pass"

[patterns.test_pattern]
enabled = false
priority = "NORMAL"
description = "Test pattern"
"""
        with open(self.config_manager.config_file, 'w') as f:
            f.write(config_content)

        # 使用现有配置初始化
        config_manager = OptimizationConfigManager(
            config_dir=self.temp_dir,
            auto_sync=False
        )

        # 验证配置被加载
        pass_config = config_manager.get_pass_config("test_pass")
        self.assertIsNotNone(pass_config)
        self.assertTrue(pass_config.enabled)
        self.assertEqual(pass_config.priority, "HIGH")

        pattern_config = config_manager.get_pattern_config("test_pattern")
        self.assertIsNotNone(pattern_config)
        self.assertFalse(pattern_config.enabled)

    def test_initialization_with_corrupted_config_file(self):
        """测试配置文件损坏时的初始化。"""
        # 创建损坏的配置文件
        with open(self.config_manager.config_file, 'w') as f:
            f.write("invalid toml content [")

        # 应该创建默认配置而不是崩溃
        config_manager = OptimizationConfigManager(
            config_dir=self.temp_dir,
            auto_sync=False
        )

        # 验证默认配置被创建
        passes_config = config_manager.config_manager.get("passes", {})
        patterns_config = config_manager.config_manager.get("patterns", {})
        self.assertIsNotNone(passes_config)
        self.assertIsNotNone(patterns_config)

    def test_register_pass_config_new(self):
        """测试注册新的 Pass 配置。"""
        pass_component = MockPass()
        pass_component.config.enabled = True
        pass_component.config.priority = ComponentPriority.HIGH

        self.config_manager.register_pass_config(pass_component)

        # 验证配置被注册
        config = self.config_manager.get_pass_config("MockPass")
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "MockPass")
        self.assertTrue(config.enabled)
        self.assertEqual(config.priority, "HIGH")

    def test_register_pass_config_existing(self):
        """测试注册已存在的 Pass 配置。"""
        # 先设置现有配置
        self.config_manager.config_manager.set("passes.existing_pass", {
            "enabled": False,
            "priority": "LOW",
            "description": "Existing"
        })

        pass_component = MockPass()
        pass_component.metadata.name = "existing_pass"
        pass_component.config.enabled = True
        pass_component.config.priority = ComponentPriority.HIGH

        self.config_manager.register_pass_config(pass_component)

        # 应该保留现有的 enabled 设置
        config = self.config_manager.get_pass_config("existing_pass")
        self.assertIsNotNone(config)
        self.assertFalse(config.enabled)  # 保留现有设置
        self.assertEqual(config.priority, "HIGH")  # 更新优先级

    def test_register_pattern_config_new(self):
        """测试注册新的 Pattern 配置。"""
        pattern_component = MockPattern()

        self.config_manager.register_pattern_config(pattern_component)

        # 验证配置被注册
        config = self.config_manager.get_pattern_config("MockPattern")
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "MockPattern")
        self.assertTrue(config.enabled)
        self.assertEqual(config.priority, "NORMAL")

    def test_register_pattern_config_existing(self):
        """测试注册已存在的 Pattern 配置。"""
        # 先设置现有配置
        self.config_manager.config_manager.set("patterns.existing_pattern", {
            "enabled": False,
            "priority": "LOW",
            "description": "Existing"
        })

        pattern_component = MockPattern()
        pattern_component.metadata.name = "existing_pattern"

        self.config_manager.register_pattern_config(pattern_component)

        # 应该保留现有的 enabled 设置
        config = self.config_manager.get_pattern_config("existing_pattern")
        self.assertIsNotNone(config)
        self.assertFalse(config.enabled)  # 保留现有设置
        self.assertEqual(config.priority, "NORMAL")  # Pattern 使用默认优先级

    def test_get_pass_config_not_found(self):
        """测试获取不存在的 Pass 配置。"""
        config = self.config_manager.get_pass_config("nonexistent")
        self.assertIsNone(config)

    def test_get_pattern_config_not_found(self):
        """测试获取不存在的 Pattern 配置。"""
        config = self.config_manager.get_pattern_config("nonexistent")
        self.assertIsNone(config)

    def test_get_pass_config_missing_name(self):
        """测试获取缺少名称字段的 Pass 配置。"""
        # 设置没有 name 字段的配置
        self.config_manager.config_manager.set("passes.test_pass", {
            "enabled": True,
            "priority": "NORMAL"
        })

        config = self.config_manager.get_pass_config("test_pass")
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "test_pass")  # 应该自动设置名称

    def test_get_pattern_config_missing_name(self):
        """测试获取缺少名称字段的 Pattern 配置。"""
        # 设置没有 name 字段的配置
        self.config_manager.config_manager.set("patterns.test_pattern", {
            "enabled": True,
            "priority": "NORMAL"
        })

        config = self.config_manager.get_pattern_config("test_pattern")
        self.assertIsNotNone(config)
        self.assertEqual(config.name, "test_pattern")  # 应该自动设置名称

    def test_apply_pass_config_valid(self):
        """测试应用有效的 Pass 配置。"""
        # 设置配置
        self.config_manager.config_manager.set("passes.test_pass", {
            "name": "test_pass",
            "enabled": False,
            "priority": "LOW"
        })

        pass_component = MockPass()
        pass_component.metadata.name = "test_pass"

        self.config_manager.apply_pass_config(pass_component)

        # 验证配置被应用
        self.assertFalse(pass_component.config.enabled)
        self.assertEqual(pass_component.config.priority, ComponentPriority.LOW)

    def test_apply_pass_config_invalid_priority(self):
        """测试应用无效优先级的 Pass 配置。"""
        # 设置无效优先级
        self.config_manager.config_manager.set("passes.test_pass", {
            "name": "test_pass",
            "enabled": True,
            "priority": "INVALID_PRIORITY"
        })

        pass_component = MockPass()
        pass_component.metadata.name = "test_pass"

        self.config_manager.apply_pass_config(pass_component)

        # 应该使用默认优先级
        self.assertEqual(pass_component.config.priority, ComponentPriority.NORMAL)

    def test_apply_pass_config_no_existing_config(self):
        """测试应用没有现有配置的 Pass。"""
        pass_component = MockPass()

        self.config_manager.apply_pass_config(pass_component)

        # 应该创建默认配置
        config = self.config_manager.get_pass_config("MockPass")
        self.assertIsNotNone(config)
        self.assertTrue(config.enabled)

    def test_apply_pattern_config_valid(self):
        """测试应用有效的 Pattern 配置。"""
        # 设置配置
        self.config_manager.config_manager.set("patterns.test_pattern", {
            "name": "test_pattern",
            "enabled": False,
            "priority": "NORMAL"
        })

        pattern_component = MockPattern()
        pattern_component.metadata.name = "test_pattern"
        # 确保初始状态是启用的
        self.assertTrue(pattern_component.config.enabled)

        self.config_manager.apply_pattern_config(pattern_component)

        # 验证配置被应用
        self.assertFalse(pattern_component.config.enabled)

    def test_apply_pattern_config_without_config_enabled(self):
        """测试应用到没有 config.enabled 属性的 Pattern。"""
        # 设置配置
        self.config_manager.config_manager.set("patterns.test_pattern", {
            "name": "test_pattern",
            "enabled": False,
            "priority": "NORMAL"
        })

        pattern_component = MockPattern()
        pattern_component.metadata.name = "test_pattern"
        # 不添加 config.enabled 属性

        self.config_manager.apply_pattern_config(pattern_component)

        # 不应该抛出异常，只记录日志

    def test_apply_pattern_config_no_existing_config(self):
        """测试应用没有现有配置的 Pattern。"""
        pattern_component = MockPattern()

        self.config_manager.apply_pattern_config(pattern_component)

        # 应该创建默认配置
        config = self.config_manager.get_pattern_config("MockPattern")
        self.assertIsNotNone(config)
        self.assertTrue(config.enabled)

    def test_save_config(self):
        """测试保存配置。"""
        # 添加一些配置
        self.config_manager.register_pass_config(MockPass())
        self.config_manager.register_pattern_config(MockPattern())

        # 保存配置
        self.config_manager.save_config()

        # 验证配置文件被创建
        self.assertTrue(self.config_manager.config_file.exists())

        # 验证配置内容
        config_manager = OptimizationConfigManager(
            config_dir=self.temp_dir,
            auto_sync=False
        )

        pass_config = config_manager.get_pass_config("MockPass")
        pattern_config = config_manager.get_pattern_config("MockPattern")
        self.assertIsNotNone(pass_config)
        self.assertIsNotNone(pattern_config)

    def test_save_config_failure(self):
        """测试保存配置失败的情况。"""
        # 模拟配置管理器保存失败
        with patch.object(self.config_manager.config_manager, 'save_file') as mock_save:
            mock_save.side_effect = ConfigError("Save failed")

            # 不应该抛出异常，只记录错误
            self.config_manager.save_config()

    def test_generate_config_from_registry(self):
        """测试从注册表生成配置。"""
        # 清空全局注册器避免测试间干扰
        from ngo.core.unified_registry import clear_registry
        clear_registry()

        # 使用真实的注册表注册组件
        registry = get_registry()
        registry.register("test_pass")(MockPass)
        registry.register("test_pattern")(MockPattern)

        # 生成配置
        self.config_manager.generate_config_from_registry(registry)

        # 验证配置被生成
        pass_config = self.config_manager.get_pass_config("test_pass")
        pattern_config = self.config_manager.get_pattern_config("test_pattern")

        self.assertIsNotNone(pass_config)
        self.assertEqual(pass_config.name, "test_pass")
        self.assertEqual(pass_config.priority, "HIGHEST")  # 这是 MockPass 的实际优先级
        self.assertTrue(pass_config.enabled)

        self.assertIsNotNone(pattern_config)
        self.assertEqual(pattern_config.name, "test_pattern")
        self.assertEqual(pattern_config.priority, "NORMAL")  # Pattern 使用默认优先级
        self.assertTrue(pattern_config.enabled)

    def test_generate_config_from_registry_with_existing_config(self):
        """测试从注册表生成配置时保留现有配置。"""
        # 先设置现有配置
        self.config_manager.config_manager.set("passes.existing_pass", {
            "enabled": False,
            "priority": "LOW"
        })

        # 创建模拟注册表
        registry = Mock(spec=UnifiedRegistry)
        registration = RegistrationInfo(
            name="existing_pass",
            component_class=MockPass,
            component_type="pass",
            enabled=True,
            priority=3
        )

        registry._registrations = {"existing_pass": registration}

        # 生成配置
        self.config_manager.generate_config_from_registry(registry)

        # 应该保留现有的 enabled 设置
        config = self.config_manager.get_pass_config("existing_pass")
        self.assertIsNotNone(config)
        self.assertFalse(config.enabled)  # 保留现有设置

    def test_generate_config_from_registry_exception(self):
        """测试从注册表生成配置时的异常处理。"""
        # 创建模拟注册表，会抛出异常
        registry = Mock(spec=UnifiedRegistry)
        registry._registrations = Mock(side_effect=Exception("Registry error"))

        # 不应该抛出异常，只记录警告
        self.config_manager.generate_config_from_registry(registry)

    def test_get_enabled_passes(self):
        """测试获取启用的 Pass 列表。"""
        # 设置配置
        self.config_manager.config_manager.set("passes", {
            "enabled_pass": {"enabled": True},
            "disabled_pass": {"enabled": False},
            "default_pass": {}  # 默认启用
        })

        enabled_passes = self.config_manager.get_enabled_passes()
        self.assertEqual(set(enabled_passes), {"enabled_pass", "default_pass"})

    def test_get_enabled_patterns(self):
        """测试获取启用的 Pattern 列表。"""
        # 设置配置
        self.config_manager.config_manager.set("patterns", {
            "enabled_pattern": {"enabled": True},
            "disabled_pattern": {"enabled": False},
            "default_pattern": {}  # 默认启用
        })

        enabled_patterns = self.config_manager.get_enabled_patterns()
        self.assertEqual(set(enabled_patterns), {"enabled_pattern", "default_pattern"})

    def test_get_config_summary(self):
        """测试获取配置摘要。"""
        # 设置配置
        self.config_manager.config_manager.set("passes", {
            "pass1": {"enabled": True},
            "pass2": {"enabled": False},
            "pass3": {}
        })

        self.config_manager.config_manager.set("patterns", {
            "pattern1": {"enabled": True},
            "pattern2": {"enabled": False}
        })

        self.config_manager.config_manager.set("settings", {"key": "value"})

        summary = self.config_manager.get_config_summary()

        self.assertEqual(summary["total_passes"], 3)
        self.assertEqual(summary["enabled_passes"], 2)
        self.assertEqual(summary["total_patterns"], 2)
        self.assertEqual(summary["enabled_patterns"], 1)
        self.assertEqual(set(summary["enabled_pass_names"]), {"pass1", "pass3"})
        self.assertEqual(set(summary["enabled_pattern_names"]), {"pattern1"})
        self.assertEqual(summary["settings"]["key"], "value")

    def test_register_pass_config_invalid_type(self):
        """测试为无效类型注册 Pass 配置。"""
        with self.assertRaises(ValueError) as context:
            self.config_manager.register_pass_config("not_a_pass")
        self.assertIn("Expected BasePass", str(context.exception))

    def test_register_pattern_config_invalid_type(self):
        """测试为无效类型注册 Pattern 配置。"""
        with self.assertRaises(ValueError) as context:
            self.config_manager.register_pattern_config("not_a_pattern")
        self.assertIn("Expected BasePattern", str(context.exception))

    def test_apply_pass_config_invalid_type(self):
        """测试为无效类型应用 Pass 配置。"""
        with self.assertRaises(ValueError) as context:
            self.config_manager.apply_pass_config("not_a_pass")
        self.assertIn("Expected BasePass", str(context.exception))

    def test_apply_pattern_config_invalid_type(self):
        """测试为无效类型应用 Pattern 配置。"""
        with self.assertRaises(ValueError) as context:
            self.config_manager.apply_pattern_config("not_a_pattern")
        self.assertIn("Expected BasePattern", str(context.exception))

    def test_priority_to_string(self):
        """测试优先级数字转字符串。"""
        result = self.config_manager._priority_to_string(ComponentPriority.HIGH)
        self.assertEqual(result, "HIGH")

        result = self.config_manager._priority_to_string(ComponentPriority.NORMAL)
        self.assertEqual(result, "NORMAL")

        result = self.config_manager._priority_to_string(ComponentPriority.LOW)
        self.assertEqual(result, "LOW")


class TestGlobalFunctions(unittest.TestCase):
    """测试全局便捷函数。"""

    def setUp(self):
        """设置测试夹具。"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试夹具。"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_get_global_optimization_config_manager(self):
        """测试获取全局优化配置管理器。"""
        manager1 = get_global_optimization_config_manager(self.temp_dir)
        manager2 = get_global_optimization_config_manager(self.temp_dir)

        # 相同目录应该返回同一个实例
        self.assertIs(manager1, manager2)

    def test_get_global_optimization_config_manager_different_dirs(self):
        """测试不同目录返回不同的实例。"""
        temp_dir2 = tempfile.mkdtemp()

        manager1 = get_global_optimization_config_manager(self.temp_dir)
        manager2 = get_global_optimization_config_manager(temp_dir2)

        # 不同目录应该返回不同实例
        self.assertIsNot(manager1, manager2)

    def test_sync_config_with_registry(self):
        """测试同步配置与注册表。"""
        # 创建模拟注册表
        registry = Mock(spec=UnifiedRegistry)
        registry._registrations = {}

        with patch('ngo.core.config.optimization_config.get_global_optimization_config_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            sync_config_with_registry(registry, self.temp_dir)

            # 验证配置生成被调用
            mock_manager.generate_config_from_registry.assert_called_once_with(registry)

    def test_sync_config_with_registry_default(self):
        """测试使用默认注册表同步配置。"""
        with patch('ngo.core.config.optimization_config.get_global_optimization_config_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager

            # 验证同步被调用
            sync_config_with_registry(config_dir=self.temp_dir)

            # 验证全局配置管理器被调用
            mock_get_manager.assert_called_once_with(self.temp_dir)
            mock_manager.generate_config_from_registry.assert_called_once()


if __name__ == "__main__":
    unittest.main()