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
NGO 优化系统的 Pass 基类。

该模块为实现可应用于计算图的优化 pass 提供基础类。
它定义了所有优化 pass 的生命周期、状态管理和接口契约。
"""

import datetime
import uuid
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from torch.fx import GraphModule

from ngo.utils.logger import get_logger
from ..core.base import (
    ComponentMetadata,
    ComponentPriority,
    OptimizationContext,
    OptimizerComponent,
)


class PassState(Enum):
    """Pass execution states."""
    CREATED = auto()
    INITIALIZED = auto()
    ANALYZING = auto()
    TRANSFORMING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    ROLLED_BACK = auto()


class PassType(Enum):
    """Types of optimization passes."""
    ANALYSIS = auto()           # Analysis-only passes (no transformation)
    TRANSFORMATION = auto()     # Transformation passes (modify graph)
    VERIFICATION = auto()       # Verification passes (check correctness)
    CLEANUP = auto()           # Cleanup passes (remove unused nodes)
    FUSION = auto()            # Fusion passes (combine operations)
    OPTIMIZATION = auto()      # General optimization passes


@dataclass
class PassMetrics:
    """Metrics collected during pass execution."""
    pass_id: str
    pass_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    nodes_analyzed: int = 0
    nodes_transformed: int = 0
    nodes_created: int = 0
    nodes_removed: int = 0
    edges_modified: int = 0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    skipped_reason: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark pass execution as finished."""
        self.end_time = datetime.now()
        self.execution_time_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message


@dataclass
class PassResult:
    """Result of pass execution."""
    pass_id: str
    pass_name: str
    success: bool
    last_stage: str
    modified_graph: bool
    metrics: PassMetrics
    applied_transformations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_follow_up_passes: List[str] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def suggest_follow_up(self, pass_name: str):
        """Suggest a follow-up pass."""
        self.suggested_follow_up_passes.append(pass_name)


@dataclass
class PassConfig:
    """Configuration for optimization passes."""
    enabled: bool = True
    priority: ComponentPriority = ComponentPriority.NORMAL
    max_iterations: int = 1
    timeout_seconds: float = 30.0
    skip_if_no_change: bool = True
    require_verification: bool = False
    verification_pass: Optional[str] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    mutually_exclusive: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'priority': self.priority.name,
            'max_iterations': self.max_iterations,
            'timeout_seconds': self.timeout_seconds,
            'skip_if_no_change': self.skip_if_no_change,
            'require_verification': self.require_verification,
            'verification_pass': self.verification_pass,
            'custom_options': self.custom_options,
            'dependencies': list(self.dependencies),
            'mutually_exclusive': list(self.mutually_exclusive)
        }


class BasePass(OptimizerComponent):
    """
    Base class for all optimization passes.

    This class provides the foundational interface and lifecycle management
    for optimization passes that operate on computational graphs. It inherits
    from OptimizerComponent to integrate with the NGO component system.
    """

    def __init__(self, name: str, pass_type: PassType, config: Optional[PassConfig] = None):
        """
        Initialize the pass.

        Args:
            name: Unique name for this pass
            pass_type: Type of optimization pass
            config: Configuration for this pass
        """
        # Create metadata for the OptimizerComponent
        metadata = ComponentMetadata(
            name=name,
            version="1.0.0",
            description=f"{pass_type.name} optimization pass"
        )

        super().__init__(metadata)
        self.pass_type = pass_type
        self._pass_config = config or PassConfig()
        self.pass_id = str(uuid.uuid4())
        self._pass_state = PassState.CREATED
        self.metrics = None
        self.logger = get_logger(f"ngo.pass.{name}")

        # Pass state tracking
        self._current_context = None
        self._original_graph_snapshot = None
        self._transformation_history = []
        self._analysis_results = {}

        # Performance tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._success_count = 0
        self._last_stage = None

        self.initialize()
        self.logger.info(f"Initialized {self.pass_type.name} pass: {self.metadata.name}")

    @property
    def config(self) -> PassConfig:
        """Get pass configuration."""
        return self._pass_config

    @property
    def state(self) -> PassState:
        """Get current pass state."""
        return self._pass_state

    @state.setter
    def state(self, value: PassState):
        """Set pass state."""
        self._pass_state = value

    @property
    def pass_info(self) -> Dict[str, Any]:
        """Get information about this pass."""
        return {
            'id': self.pass_id,
            'name': self.metadata.name,
            'type': self.pass_type.name,
            'state': self.state.name,
            'enabled': self.config.enabled,
            'priority': self.config.priority.name,
            'execution_count': self._execution_count,
            'success_rate': self._success_count / max(1, self._execution_count),
            'avg_execution_time': self._total_execution_time / max(1, self._execution_count)
        }

    def initialize(self) -> bool:
        """
        Initialize the pass.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.state = PassState.INITIALIZED
            self.logger.debug(f"Pass {self.metadata.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize pass {self.metadata.name}: {e}")
            self.state = PassState.FAILED
            return False

    def can_execute(self, context: OptimizationContext) -> Tuple[bool, Optional[str]]:
        """
        Check if this pass can be executed in the given context.

        Args:
            context: Optimization context to check

        Returns:
            Tuple of (can_execute, reason_if_not)
        """
        if not self.config.enabled:
            return False, "Pass is disabled"

        if self.state not in [PassState.INITIALIZED, PassState.COMPLETED]:
            return False, f"Pass not in ready state: {self.state.name}"

        # Check dependencies
        for dep_pass in self.config.dependencies:
            if not context.get_component_result(dep_pass):
                return False, f"Dependency {dep_pass} not completed"

        # Check mutually exclusive passes
        for exclusive_pass in self.config.mutually_exclusive:
            if context.get_component_result(exclusive_pass):
                return False, f"Mutually exclusive pass {exclusive_pass} already executed"

        return True, None

    def execute(self, context: OptimizationContext) -> PassResult:
        """
        Execute the optimization pass.

        This is the main entry point for pass execution. It orchestrates the
        complete lifecycle: analysis, transformation, and verification.

        Args:
            context: Optimization context containing the graph to optimize

        Returns:
            PassResult containing execution details and metrics
        """
        start_time = datetime.now()
        self.metrics = PassMetrics(
            pass_id=self.pass_id,
            pass_name=self.metadata.name,
            start_time=start_time
        )

        try:
            # Check if we can execute
            can_execute, reason = self.can_execute(context)
            if not can_execute:
                self.state = PassState.SKIPPED
                self.metrics.skipped_reason = reason
                self.metrics.finish(success=False)
                return PassResult(
                    pass_id=self.pass_id,
                    pass_name=self.metadata.name,
                    success=False,
                    modified_graph=False,
                    metrics=self.metrics,
                    last_stage="can_execute"
                )

            # Create snapshot for potential rollback
            self._create_graph_snapshot(context)

            # Use the standard three-phase execution from BasePass
            # Execute pass lifecycle: analysis, transform, verify
            self.state = PassState.ANALYZING
            analysis_result = self.analyze(context)
            self.logger.debug(f"Analysis result: {analysis_result}")

            if analysis_result.should_proceed:
                self.state = PassState.TRANSFORMING
                transform_result = self.transform(context, analysis_result)
                self.logger.debug(f"Transform result: {transform_result}")

                if transform_result.success:
                    self.state = PassState.VERIFYING
                    verification_result = self.verify(context, transform_result)
                    self.logger.debug(f"Verification result: {verification_result}")

                    if verification_result.success:
                        self.state = PassState.COMPLETED
                        self._execution_count += 1
                        self._success_count += 1
                        self._total_execution_time += (datetime.now() - start_time).total_seconds()

                        # Store result in context
                        context.set_component_result(self.metadata.name, transform_result)

                        # Complete metrics
                        self._complete_metrics(context, True)

                        result = PassResult(
                            pass_id=self.pass_id,
                            pass_name=self.metadata.name,
                            success=True,
                            modified_graph=transform_result.modified_graph,
                            metrics=self.metrics,
                            applied_transformations=transform_result.transformations_applied,
                            warnings=transform_result.warnings,
                            suggested_follow_up_passes=transform_result.suggested_follow_up_passes,
                            last_stage="verify"
                        )
                        self.logger.debug(f"Pass result: {result}")

                        # Call post_execute hook
                        self.post_execute(context, result)

                        return result
                    else:
                        # Verification failed, attempt rollback
                        self._rollback(context)
                        self.state = PassState.FAILED
                        self._complete_metrics(context, False, verification_result.error_message)
                        return PassResult(
                            pass_id=self.pass_id,
                            pass_name=self.metadata.name,
                            success=False,
                            modified_graph=False,
                            metrics=self.metrics,
                            last_stage="verify"
                        )
                else:
                    # Transformation failed
                    self._rollback(context)
                    self.state = PassState.FAILED
                    self._complete_metrics(context, False, transform_result.error_message)
                    return PassResult(
                        pass_id=self.pass_id,
                        pass_name=self.metadata.name,
                        success=False,
                        modified_graph=False,
                        metrics=self.metrics,
                        last_stage="transform"
                    )
            else:
                # Analysis determined we should skip
                self.state = PassState.SKIPPED
                self.metrics.skipped_reason = analysis_result.skip_reason
                self.metrics.finish(success=True)
                return PassResult(
                    pass_id=self.pass_id,
                    pass_name=self.metadata.name,
                    success=True,
                    modified_graph=False,
                    metrics=self.metrics,
                    last_stage="analyze"
                )

        except Exception as e:
            # Unexpected error
            self._rollback(context)
            self.state = PassState.FAILED
            error_msg = f"Unexpected error: {str(e)}"
            self._complete_metrics(context, False, error_msg)
            self.logger.exception(f"Pass {self.metadata.name} failed with unexpected error")
            return PassResult(
                pass_id=self.pass_id,
                pass_name=self.metadata.name,
                success=False,
                modified_graph=False,
                metrics=self.metrics,
                last_stage="exception"
            )

    def _create_graph_snapshot(self, context: OptimizationContext):
        """Create a snapshot of the current GraphModule for rollback."""
        try:
            graph_module = self._get_graph_from_context(context)
            if graph_module is not None:
                self._original_graph_snapshot = deepcopy(graph_module)
                # Create a deep copy of the graph module for rollback
                
            else:
                self._original_graph_snapshot = None
        except Exception as e:
            self.logger.warning(f"Failed to create graph snapshot: {e}")
            self._original_graph_snapshot = None

    def _rollback(self, context: OptimizationContext):
        """Roll back to the original GraphModule state."""
        try:
            if self._original_graph_snapshot:
                # For torch.fx.GraphModule, we would need to recreate the original graph
                # This is a simplified implementation - in practice, you'd need to
                # reconstruct the graph from the snapshot
                context.set_graph_module(self._original_graph_snapshot)
                self.logger.info(f"Attempting to rollback {self.metadata.name} - GraphModule rollback requires graph reconstruction")

                self.state = PassState.ROLLED_BACK
                self.logger.info(f"Pass {self.metadata.name} marked for rollback")
            else:
                self.logger.warning(f"No snapshot available for rollback of {self.metadata.name}")
        except Exception as e:
            self.logger.error(f"Failed to rollback pass {self.metadata.name}: {e}")

    def _complete_metrics(self, context: OptimizationContext, success: bool, error_message: Optional[str] = None):
        """Complete the metrics collection."""
        try:
            # Collect memory usage
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_after_mb = memory_info.rss / 1024 / 1024
        except ImportError:
            pass

        self.metrics.finish(success=success, error_message=error_message)

    def _get_graph_from_context(self, context: OptimizationContext) -> Optional[GraphModule]:
        """
        Extract torch.fx.GraphModule from optimization context.

        Args:
            context: Optimization context containing the graph

        Returns:
            GraphModule or None if not found
        """
        # Try to get graph module directly from context
        if hasattr(context, 'graph_module') and context.graph_module is not None:
            if isinstance(context.graph_module, GraphModule):
                return context.graph_module

        return None

    # Abstract methods that subclasses must implement

    @abstractmethod
    def analyze(self, context: OptimizationContext) -> 'AnalysisResult':
        """
        Analyze the graph to determine optimization opportunities.

        Args:
            context: Optimization context containing the graph

        Returns:
            AnalysisResult with optimization opportunities and recommendations
        """
        pass

    @abstractmethod
    def transform(self, context: OptimizationContext, analysis_result: 'AnalysisResult') -> 'TransformResult':
        """
        Apply transformations to the graph based on analysis.

        Args:
            context: Optimization context containing the graph
            analysis_result: Results from the analysis phase

        Returns:
            TransformResult with transformation details
        """
        pass

    def verify(self, context: OptimizationContext, transform_result: 'TransformResult') -> 'VerificationResult':
        """
        Verify that transformations are correct and preserve semantics.

        Args:
            context: Optimization context containing the transformed graph
            transform_result: Results from the transformation phase

        Returns:
            VerificationResult with correctness assessment
        """
        # Default implementation returns success
        return VerificationResult(success=True)

    # Optional hook methods that subclasses can override

    def pre_execute(self, context: OptimizationContext) -> bool:
        """
        Hook called before execution begins.

        Args:
            context: Optimization context

        Returns:
            True if execution should proceed, False to skip
        """
        return True

    def post_execute(self, context: OptimizationContext, result: PassResult):
        """
        Hook called after execution completes.

        Args:
            context: Optimization context
            result: Result of the execution
        """
        pass

    def cleanup(self, context: OptimizationContext):
        """Clean up resources after execution."""
        self._original_graph_snapshot = None
        self._transformation_history.clear()
        self._analysis_results.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for this pass."""
        return {
            'execution_count': self._execution_count,
            'success_count': self._success_count,
            'failure_count': self._execution_count - self._success_count,
            'success_rate': self._success_count / max(1, self._execution_count),
            'total_execution_time': self._total_execution_time,
            'average_execution_time': self._total_execution_time / max(1, self._execution_count),
            'current_state': self.state.name,
            'pass_type': self.pass_type.name
        }


# Result classes for pass phases

@dataclass
class AnalysisResult:
    """Result of the analysis phase."""
    should_proceed: bool = True
    skip_reason: Optional[str] = None
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    analysis_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    estimated_improvement: Optional[float] = None
    analysis_cache: Optional[Dict[str, Any]] = None

    def add_opportunity(self, opportunity_type: str, description: str,
                       location: Optional[str] = None, impact: Optional[float] = None):
        """Add an optimization opportunity."""
        opportunity = {
            'type': opportunity_type,
            'description': description,
            'location': location,
            'impact': impact,
            'timestamp': datetime.now()
        }
        self.optimization_opportunities.append(opportunity)

    def add_recommendation(self, recommendation: str):
        """Add a recommendation."""
        self.recommendations.append(recommendation)


@dataclass
class TransformResult:
    """Result of the transformation phase."""
    success: bool = True
    modified_graph: bool = False
    transformations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    transform_metrics: Dict[str, Any] = field(default_factory=dict)
    suggested_follow_up_passes: List[str] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None

    def add_transformation(self, transformation: str):
        """Add a transformation that was applied."""
        self.transformations_applied.append(transformation)

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class VerificationResult:
    """Result of the verification phase."""
    success: bool = True
    error_message: Optional[str] = None
    verification_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    performance_impact: Optional[Dict[str, float]] = None

    def add_warning(self, warning: str):
        """Add a verification warning."""
        self.warnings.append(warning)