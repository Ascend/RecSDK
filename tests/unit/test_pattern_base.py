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
Pattern 和 Strategy 基类的单元测试。
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn
import unittest
import torch.fx
from torch.fx import GraphModule, symbolic_trace

# 将 src 目录添加到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ngo.core.base import ComponentMetadata, OptimizationContext
from ngo.patterns.base import (
    BasePattern, BaseStrategy, PatternMatchResult, StrategyExecutionResult,
    StrategyPriority, StrategyStatistics, clear_pattern_registry, register_strategy
)


class SimpleModel(nn.Module):
    """用于测试的简单模型。"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        """前向传播函数。"""
        x = self.linear(x)
        x = self.relu(x)
        return x


class TestPattern(BasePattern):
    """测试 Pattern 实现。"""

    def __init__(self):
        metadata = ComponentMetadata(
            name="TestPattern",
            version="1.0.0",
            description="用于单元测试的测试 Pattern",
        )
        super().__init__(metadata)

    def match(self, graph_module: GraphModule, nodes=None):
        """模式匹配方法 - 简单的模拟实现，总是匹配成功。"""
        # 简单的模拟实现 - 总是匹配
        return PatternMatchResult(matched=True, match_score=1.0)

    def initialize(self):
        """初始化方法。"""
        pass

    def _execute_impl(self, context: OptimizationContext):
        """内部执行实现。"""
        return {"pattern_executed": True}

    def execute(self, context: OptimizationContext):
        """执行方法。"""
        return self._execute_impl(context)


class TestStrategy(BaseStrategy):
    """测试 Strategy 实现。"""

    def __init__(self):
        metadata = ComponentMetadata(
            name="TestStrategy",
            version="1.0.0",
            description="用于单元测试的测试 Strategy",
        )
        super().__init__(metadata)

    def execute(self, graph_module: GraphModule, match_result: PatternMatchResult):
        """执行策略 - 简单的模拟实现，总是成功。"""
        # 简单的模拟实现 - 总是成功
        return StrategyExecutionResult(success=True)

    def initialize(self):
        """初始化方法。"""
        pass

    def _execute_impl(self, context: OptimizationContext):
        """内部执行实现。"""
        return {"strategy_executed": True}


class TestPatternMatchResult(unittest.TestCase):
    """测试 PatternMatchResult 类。"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化。"""
        result = PatternMatchResult(matched=True)
        self.assertTrue(result.matched)
        self.assertEqual(result.matched_nodes, [])
        self.assertEqual(result.match_score, 0.0)
        self.assertEqual(result.metadata, {})

    def test_init_with_values(self):
        """测试使用自定义值初始化。"""
        nodes = [Mock()]
        metadata = {"test": "value"}
        result = PatternMatchResult(
            matched=True, matched_nodes=nodes, match_score=0.8, metadata=metadata
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.matched_nodes, nodes)
        self.assertEqual(result.match_score, 0.8)
        self.assertEqual(result.metadata, metadata)

    def test_bool_conversion(self):
        """测试布尔值转换。"""
        self.assertTrue(PatternMatchResult(matched=True))
        self.assertFalse(PatternMatchResult(matched=False))


class TestStrategyExecutionResult(unittest.TestCase):
    """测试 StrategyExecutionResult 类。"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化。"""
        result = StrategyExecutionResult(success=True)
        self.assertTrue(result.success)
        self.assertEqual(result.transformed_nodes, [])
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.metadata, {})

    def test_init_with_values(self):
        """测试使用自定义值初始化。"""
        nodes = [Mock()]
        metadata = {"test": "value"}
        result = StrategyExecutionResult(
            success=True, transformed_nodes=nodes, execution_time=0.5, metadata=metadata
        )
        self.assertTrue(result.success)
        self.assertEqual(result.transformed_nodes, nodes)
        self.assertEqual(result.execution_time, 0.5)
        self.assertEqual(result.metadata, metadata)

    def test_bool_conversion(self):
        """测试布尔值转换。"""
        self.assertTrue(StrategyExecutionResult(success=True))
        self.assertFalse(StrategyExecutionResult(success=False))


class TestStrategyStatistics(unittest.TestCase):
    """测试 StrategyStatistics 类。"""

    def test_init_with_defaults(self):
        """测试使用默认值初始化。"""
        stats = StrategyStatistics()
        self.assertEqual(stats.execution_count, 0)
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.total_execution_time, 0.0)
        self.assertEqual(stats.average_execution_time, 0.0)
        self.assertIsNone(stats.last_execution_time)
        self.assertEqual(stats.success_rate, 0.0)

    def test_update_success(self):
        """测试使用成功执行更新统计信息。"""
        stats = StrategyStatistics()
        stats.update(True, 0.1)

        self.assertEqual(stats.execution_count, 1)
        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.failure_count, 0)
        self.assertEqual(stats.total_execution_time, 0.1)
        self.assertEqual(stats.average_execution_time, 0.1)
        self.assertEqual(stats.last_execution_time, 0.1)
        self.assertEqual(stats.success_rate, 1.0)

    def test_update_failure(self):
        """测试使用失败执行更新统计信息。"""
        stats = StrategyStatistics()
        stats.update(False, 0.2)

        self.assertEqual(stats.execution_count, 1)
        self.assertEqual(stats.success_count, 0)
        self.assertEqual(stats.failure_count, 1)
        self.assertEqual(stats.total_execution_time, 0.2)
        self.assertEqual(stats.average_execution_time, 0.2)
        self.assertEqual(stats.last_execution_time, 0.2)
        self.assertEqual(stats.success_rate, 0.0)

    def test_multiple_updates(self):
        """测试使用多次执行更新统计信息。"""
        stats = StrategyStatistics()
        stats.update(True, 0.1)
        stats.update(True, 0.2)
        stats.update(False, 0.3)

        self.assertEqual(stats.execution_count, 3)
        self.assertEqual(stats.success_count, 2)
        self.assertEqual(stats.failure_count, 1)
        self.assertAlmostEqual(stats.total_execution_time, 0.6, places=6)
        self.assertAlmostEqual(stats.average_execution_time, 0.2, places=6)
        self.assertEqual(stats.last_execution_time, 0.3)
        self.assertAlmostEqual(stats.success_rate, 2 / 3, places=6)


class TestBasePattern(unittest.TestCase):
    """测试 BasePattern 类。"""

    def setUp(self):
        """设置测试夹具。"""
        self.pattern = TestPattern()

    def test_init(self):
        """测试 Pattern 初始化。"""
        self.assertEqual(self.pattern.metadata.name, "TestPattern")
        self.assertEqual(self.pattern.metadata.version, "1.0.0")
        self.assertEqual(self.pattern.strategy_count, 0)
        self.assertEqual(self.pattern.strategy_names, [])

    def test_register_strategy(self):
        """测试注册 Strategy。"""
        strategy = TestStrategy()
        self.pattern.register_strategy(strategy, StrategyPriority.HIGH)

        self.assertEqual(self.pattern.strategy_count, 1)
        self.assertIn("TestStrategy", self.pattern.strategy_names)

    def test_register_multiple_strategies(self):
        """测试注册多个具有不同优先级的 Strategy。"""
        strategy1 = TestStrategy()
        strategy2 = TestStrategy()
        strategy1._metadata.name = "Strategy1"
        strategy2._metadata.name = "Strategy2"

        self.pattern.register_strategy(strategy1, StrategyPriority.NORMAL)
        self.pattern.register_strategy(strategy2, StrategyPriority.HIGH)

        self.assertEqual(self.pattern.strategy_count, 2)
        strategies = self.pattern.get_strategies()
        self.assertEqual(
            strategies[0].metadata.name, "Strategy2"
        )  # 高优先级在前
        self.assertEqual(strategies[1].metadata.name, "Strategy1")

    def test_unregister_strategy(self):
        """测试注销 Strategy。"""
        strategy = TestStrategy()
        self.pattern.register_strategy(strategy)

        self.assertTrue(self.pattern.unregister_strategy("TestStrategy"))
        self.assertEqual(self.pattern.strategy_count, 0)
        self.assertFalse(self.pattern.unregister_strategy("NonExistent"))

    def test_execute_strategies(self):
        """测试执行 Strategy。"""
        strategy = TestStrategy()
        self.pattern.register_strategy(strategy)

        # 创建一个简单的图模块
        model = SimpleModel()
        graph_module = symbolic_trace(model)
        match_result = PatternMatchResult(matched=True)

        results = self.pattern.execute_strategies(graph_module, match_result)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertGreaterEqual(results[0].execution_time, 0.0)

    def test_execute_strategies_with_failure(self):
        """测试当一个 Strategy 失败时执行 Strategy。"""

        class FailingStrategy(BaseStrategy):
            """失败的 Strategy 测试类。"""
            def __init__(self):
                metadata = ComponentMetadata(name="FailingStrategy", version="1.0.0")
                super().__init__(metadata)

            def execute(self, graph_module, match_result):
                """执行方法，总是抛出异常。"""
                raise Exception("Strategy failed")

            def initialize(self):
                """初始化方法。"""
                pass

            def _execute_impl(self, context):
                """内部执行实现。"""
                pass

        strategy = FailingStrategy()
        self.pattern.register_strategy(strategy)

        model = SimpleModel()
        graph_module = symbolic_trace(model)
        match_result = PatternMatchResult(matched=True)

        results = self.pattern.execute_strategies(graph_module, match_result)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertGreater(results[0].execution_time, 0.0)

    def test_match(self):
        """测试模式匹配。"""
        model = SimpleModel()
        graph_module = symbolic_trace(model)

        result = self.pattern.match(graph_module)

        self.assertTrue(result.matched)
        self.assertEqual(result.match_score, 1.0)


class TestBaseStrategy(unittest.TestCase):
    """测试 BaseStrategy 类。"""

    def setUp(self):
        """设置测试夹具。"""
        self.strategy = TestStrategy()

    def test_init(self):
        """测试 Strategy 初始化。"""
        self.assertEqual(self.strategy.metadata.name, "TestStrategy")
        self.assertEqual(self.strategy.metadata.version, "1.0.0")
        self.assertEqual(self.strategy.statistics.execution_count, 0)

    def test_execute(self):
        """测试 Strategy 执行。"""
        model = SimpleModel()
        graph_module = symbolic_trace(model)
        match_result = PatternMatchResult(matched=True)

        result = self.strategy.execute(graph_module, match_result)

        self.assertTrue(result.success)

    def test_statistics_reset(self):
        """测试统计信息重置。"""
        self.strategy._statistics.update(True, 0.1)
        self.assertEqual(self.strategy.statistics.execution_count, 1)

        self.strategy.reset_statistics()
        self.assertEqual(self.strategy.statistics.execution_count, 0)


class TestRegisterStrategyDecorator(unittest.TestCase):
    """测试 register_strategy 装饰器。"""

    def test_decorator_registration(self):
        """测试装饰器正确地将 Strategy 注册到 Pattern。"""

        class TestStrategy2(BaseStrategy):
            """测试 Strategy 2。"""
            def __init__(self):
                metadata = ComponentMetadata(name="TestStrategy2", version="1.0.0")
                super().__init__(metadata)

            def execute(self, graph_module, match_result):
                """执行方法。"""
                return StrategyExecutionResult(success=True)

            def initialize(self):
                """初始化方法。"""
                pass

            def _execute_impl(self, context):
                """内部执行实现。"""
                pass

        @register_strategy(TestStrategy2, StrategyPriority.HIGH)
        class DecoratedPattern(BasePattern):
            """使用装饰器的 Pattern。"""
            def __init__(self, metadata=None):
                if metadata is None:
                    metadata = ComponentMetadata(name="DecoratedPattern", version="1.0.0")
                super().__init__(metadata)

            def match(self, graph_module, nodes=None):
                """模式匹配方法。"""
                return PatternMatchResult(matched=True)

            def initialize(self):
                """初始化方法。"""
                pass

            def _execute_impl(self, context):
                """内部执行实现。"""
                pass

            def execute(self, context):
                """执行方法。"""
                return self._execute_impl(context)

        # 创建 Pattern 实例 - Strategy 应该自动注册
        pattern = DecoratedPattern()

        # 检查 Pattern 类是否有注册信息
        self.assertTrue(hasattr(DecoratedPattern, "_strategy_registrations"))
        self.assertEqual(len(DecoratedPattern._strategy_registrations), 1)
        self.assertEqual(
            DecoratedPattern._strategy_registrations[0]["strategy_class"], TestStrategy2
        )
        self.assertEqual(
            DecoratedPattern._strategy_registrations[0]["priority"], StrategyPriority.HIGH
        )

        # 验证 Strategy 自动注册到 Pattern
        self.assertEqual(pattern.strategy_count, 1)
        self.assertIn("TestStrategy2", pattern.strategy_names)

        # 验证 Strategy 正常工作
        strategies = pattern.get_strategies()
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].metadata.name, "TestStrategy2")

    def test_multiple_strategy_registration(self):
        """测试将多个 Strategy 注册到一个 Pattern。"""

        class TestStrategy3(BaseStrategy):
            """测试 Strategy 3。"""
            def __init__(self):
                metadata = ComponentMetadata(name="TestStrategy3", version="1.0.0")
                super().__init__(metadata)

            def execute(self, graph_module, match_result):
                """执行方法。"""
                return StrategyExecutionResult(success=True)

            def initialize(self):
                """初始化方法。"""
                pass

            def _execute_impl(self, context):
                """内部执行实现。"""
                pass

        class TestStrategy4(BaseStrategy):
            """测试 Strategy 4。"""
            def __init__(self):
                metadata = ComponentMetadata(name="TestStrategy4", version="1.0.0")
                super().__init__(metadata)

            def execute(self, graph_module, match_result):
                """执行方法。"""
                return StrategyExecutionResult(success=True)

            def initialize(self):
                """初始化方法。"""
                pass

            def _execute_impl(self, context):
                """内部执行实现。"""
                pass

        @register_strategy(TestStrategy3, StrategyPriority.NORMAL)
        @register_strategy(TestStrategy4, StrategyPriority.HIGH)
        class MultiStrategyPattern(BasePattern):
            """多 Strategy Pattern。"""
            def __init__(self, metadata=None):
                if metadata is None:
                    metadata = ComponentMetadata(name="MultiStrategyPattern", version="1.0.0")
                super().__init__(metadata)

            def match(self, graph_module, nodes=None):
                """模式匹配方法。"""
                return PatternMatchResult(matched=True)

            def initialize(self):
                """初始化方法。"""
                pass

            def _execute_impl(self, context):
                """内部执行实现。"""
                pass

            def execute(self, context):
                """执行方法。"""
                return self._execute_impl(context)

        # 创建 Pattern 实例 - Strategy 应该自动注册
        pattern = MultiStrategyPattern()

        # 检查 Pattern 类是否有两个 Strategy 的注册信息
        self.assertTrue(hasattr(MultiStrategyPattern, "_strategy_registrations"))
        self.assertEqual(len(MultiStrategyPattern._strategy_registrations), 2)

        # 检查两个 Strategy 都以正确的优先级注册
        strategy_classes = [
            reg["strategy_class"] for reg in MultiStrategyPattern._strategy_registrations
        ]
        priorities = [reg["priority"] for reg in MultiStrategyPattern._strategy_registrations]

        self.assertIn(TestStrategy3, strategy_classes)
        self.assertIn(TestStrategy4, strategy_classes)
        self.assertIn(StrategyPriority.NORMAL, priorities)
        self.assertIn(StrategyPriority.HIGH, priorities)

        # 验证 Strategy 自动注册到 Pattern
        self.assertEqual(pattern.strategy_count, 2)
        self.assertIn("TestStrategy3", pattern.strategy_names)
        self.assertIn("TestStrategy4", pattern.strategy_names)

        # 验证 Strategy 按优先级排序
        strategies = pattern.get_strategies()
        self.assertEqual(len(strategies), 2)
        self.assertEqual(strategies[0].metadata.name, "TestStrategy4")  # HIGH 优先级
        self.assertEqual(strategies[1].metadata.name, "TestStrategy3")  # NORMAL 优先级




class TestStrategyPriority(unittest.TestCase):
    """测试 StrategyPriority 枚举。"""

    def test_priority_values(self):
        """测试优先级枚举值。"""
        self.assertEqual(StrategyPriority.LOW.value, 1)
        self.assertEqual(StrategyPriority.NORMAL.value, 2)
        self.assertEqual(StrategyPriority.HIGH.value, 3)
        self.assertEqual(StrategyPriority.CRITICAL.value, 4)

    def test_priority_ordering(self):
        """测试优先级排序。"""
        priorities = [
            StrategyPriority.LOW,
            StrategyPriority.NORMAL,
            StrategyPriority.HIGH,
            StrategyPriority.CRITICAL,
        ]
        values = [p.value for p in priorities]
        self.assertEqual(values, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
