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
Pattern 和 Strategy 系统的基类。

该模块为 NGO 中的模式匹配和策略执行系统提供基础的抽象基类。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from torch.fx import GraphModule, Node

from ngo.utils.logger import get_logger
from ..core.base import ComponentMetadata, OptimizationContext, OptimizerComponent


class PatternMatchResult:
    """Result of pattern matching."""

    def __init__(
        self,
        matched: bool,
        matched_nodes: List[Node] = None,
        match_score: float = 0.0,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize pattern match result.

        Args:
            matched: Whether pattern was matched
            matched_nodes: List of matched nodes
            match_score: Confidence score of the match
            metadata: Additional match metadata
        """
        self.matched = matched
        self.matched_nodes = matched_nodes or []
        self.match_score = match_score
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        """Boolean conversion for convenience."""
        return self.matched


class StrategyExecutionResult:
    """Result of strategy execution."""

    def __init__(
        self,
        success: bool,
        transformed_nodes: List[Node] = None,
        execution_time: float = 0.0,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize strategy execution result.

        Args:
            success: Whether strategy execution was successful
            transformed_nodes: List of transformed nodes
            execution_time: Time taken for execution
            metadata: Additional execution metadata
        """
        self.success = success
        self.transformed_nodes = transformed_nodes or []
        self.execution_time = execution_time
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        """Boolean conversion for convenience."""
        return self.success


class StrategyPriority(Enum):
    """Strategy execution priorities."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class StrategyStatistics:
    """Statistics for strategy execution."""

    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_execution_time: Optional[float] = None

    def update(self, success: bool, execution_time: float) -> None:
        """Update statistics with execution result."""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.execution_count
        self.last_execution_time = execution_time

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


class BasePattern(OptimizerComponent):
    """
    Abstract base class for all optimization patterns.

    A pattern defines a specific graph structure that can be matched
    and optimized using one or more strategies.
    """

    def __init__(self, metadata: ComponentMetadata):
        """
        Initialize the pattern.

        Args:
            metadata: Pattern metadata
        """
        super().__init__(metadata)
        self._strategies: Dict[str, "BaseStrategy"] = {}
        self._strategy_priorities: Dict[str, StrategyPriority] = {}
        self._logger = get_logger(f"ngo.pattern.{self.__class__.__name__}")

    @abstractmethod
    def match(
        self, graph_module: GraphModule, nodes: List[Node] = None
    ) -> PatternMatchResult:
        """
        Match this pattern against a graph module.

        Args:
            graph_module: The graph module to match against
            nodes: Optional list of nodes to consider (if None, consider all nodes)

        Returns:
            PatternMatchResult indicating match success and details
        """
        pass

    def register_strategy(
        self,
        strategy: "BaseStrategy",
        priority: StrategyPriority = StrategyPriority.NORMAL,
    ) -> None:
        """
        Register a strategy with this pattern.

        Args:
            strategy: Strategy to register
            priority: Execution priority for this strategy
        """
        strategy_id = strategy.metadata.name
        self._strategies[strategy_id] = strategy
        self._strategy_priorities[strategy_id] = priority
        self._logger.debug(
            f"Registered strategy {strategy_id} with priority {priority.name}"
        )

    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a strategy from this pattern.

        Args:
            strategy_id: ID of strategy to unregister

        Returns:
            True if strategy was unregistered, False if not found
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            del self._strategy_priorities[strategy_id]
            self._logger.debug(f"Unregistered strategy {strategy_id}")
            return True
        return False

    def get_strategies(self) -> List["BaseStrategy"]:
        """
        Get all registered strategies, sorted by priority.

        Returns:
            List of strategies in priority order
        """
        strategies = list(self._strategies.values())
        strategies.sort(
            key=lambda s: self._strategy_priorities[s.metadata.name].value, reverse=True
        )
        return strategies

    def execute_strategies(
        self, graph_module: GraphModule, match_result: PatternMatchResult
    ) -> List[StrategyExecutionResult]:
        """
        Execute all registered strategies on a matched pattern.

        Args:
            graph_module: The graph module to transform
            match_result: Result of pattern matching

        Returns:
            List of strategy execution results
        """
        results = []
        for strategy in self.get_strategies():
            try:
                start_time = time.time()
                result = strategy.execute(graph_module, match_result)
                execution_time = time.time() - start_time
                result.execution_time = execution_time

                # Update strategy statistics
                strategy._statistics.update(result.success, execution_time)

                results.append(result)
                self._logger.debug(
                    f"Strategy {strategy.metadata.name} executed in {execution_time:.4f}s"
                )

            except Exception as e:
                self._logger.error(
                    f"Strategy {strategy.metadata.name} execution failed: {e}"
                )
                result = StrategyExecutionResult(
                    False, execution_time=time.time() - start_time
                )
                strategy._statistics.update(False, result.execution_time)
                results.append(result)

        return results

    @property
    def strategy_count(self) -> int:
        """Get number of registered strategies."""
        return len(self._strategies)

    @property
    def strategy_names(self) -> List[str]:
        """Get names of registered strategies."""
        return list(self._strategies.keys())


class BaseStrategy(OptimizerComponent):
    """
    Abstract base class for all optimization strategies.

    A strategy defines how to transform a matched pattern into
    an optimized version.
    """

    def __init__(self, metadata: ComponentMetadata):
        """
        Initialize the strategy.

        Args:
            metadata: Strategy metadata
        """
        super().__init__(metadata)
        self._statistics = StrategyStatistics()
        self._logger = get_logger(f"ngo.strategy.{self.__class__.__name__}")

    @abstractmethod
    def execute(
        self, graph_module: GraphModule, match_result: PatternMatchResult
    ) -> StrategyExecutionResult:
        """
        Execute this strategy on a matched pattern.

        Args:
            graph_module: The graph module to transform
            match_result: Result of pattern matching

        Returns:
            StrategyExecutionResult indicating execution success and details
        """
        pass

    @property
    def statistics(self) -> StrategyStatistics:
        """Get strategy execution statistics."""
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._statistics = StrategyStatistics()
        self._logger.debug("Statistics reset")


# Strategy registration decorator
def register_strategy(
    strategy_class: Type[BaseStrategy],
    priority: StrategyPriority = StrategyPriority.NORMAL,
):
    """
    Decorator to register a strategy with a pattern class.

    This decorator allows patterns to automatically register strategies when instantiated.

    Args:
        strategy_class: The strategy class to register with the pattern
        priority: Execution priority for this strategy

    Returns:
        Decorator function that can be applied to pattern classes
    """

    def decorator(pattern_class: Type[BasePattern]):
        # Store registration information on the pattern class
        if not hasattr(pattern_class, "_strategy_registrations"):
            pattern_class._strategy_registrations = []

        pattern_class._strategy_registrations.append(
            {"strategy_class": strategy_class, "priority": priority}
        )

        # Override pattern's __init__ to automatically register strategies
        original_init = pattern_class.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Automatically register strategies with this pattern instance
            if hasattr(pattern_class, "_strategy_registrations"):
                for registration in pattern_class._strategy_registrations:
                    strategy_instance = registration["strategy_class"]()
                    self.register_strategy(strategy_instance, registration["priority"])

        pattern_class.__init__ = new_init

        return pattern_class

    return decorator


# Global pattern manager instance
_global_pattern_manager = None


def get_global_pattern_manager() -> "PatternManager":
    """
    Get the global pattern manager instance.

    Returns:
        Global PatternManager instance
    """
    global _global_pattern_manager
    if _global_pattern_manager is None:
        from .manager import PatternManager

        _global_pattern_manager = PatternManager()
        _global_pattern_manager.initialize()
    return _global_pattern_manager


def register_pattern(pattern: BasePattern) -> bool:
    """
    Register a pattern globally.

    Args:
        pattern: Pattern to register

    Returns:
        True if registration was successful, False otherwise
    """
    manager = get_global_pattern_manager()
    return manager.register_pattern(pattern)


def get_pattern(pattern_id: str) -> Optional[BasePattern]:
    """
    Get a registered pattern by ID.

    Args:
        pattern_id: Pattern identifier

    Returns:
        Pattern instance or None if not found
    """
    manager = get_global_pattern_manager()
    return manager.get_pattern(pattern_id)


def list_patterns() -> List[str]:
    """
    List all registered pattern IDs.

    Returns:
        List of pattern identifiers
    """
    manager = get_global_pattern_manager()
    return manager.list_patterns()


def clear_pattern_registry() -> None:
    """Clear all registered patterns."""
    global _global_pattern_manager
    _global_pattern_manager = None


# Pattern registration decorator
def register_pattern_class(pattern_class: Type[BasePattern] = None):
    """
    Decorator to automatically register a pattern class with the global pattern manager.

    This decorator creates a pattern instance and registers it globally when the class is defined.

    Args:
        pattern_class: The pattern class to register (optional, for decorator usage)

    Returns:
        Decorator function or the original pattern class
    """

    def decorator(cls):
        # Create pattern instance and register it globally
        pattern_instance = cls()
        register_pattern(pattern_instance)
        return cls

    # Handle both @register_pattern_class and @register_pattern_class() usage
    if pattern_class is None:
        return decorator
    else:
        return decorator(pattern_class)
