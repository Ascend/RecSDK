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
公共子表达式消除Pass。

该Pass识别并消除计算图中相同表达式的冗余计算。它分析图以找到相同的子表达式，
只计算一次，并在多个使用中重用结果。

算法流程：
1. 遍历图以识别CSE的候选操作
2. 创建表达式键以识别相同的子表达式
3. 跟踪表达式出现并计算优化潜力
4. 用对计算结果的引用替换冗余计算
"""

from collections import deque
from typing import Any, Dict, List, Optional

from ngo.utils.logger import get_logger
from ..core.base import ComponentPriority, OptimizationContext
from ..core.unified_registry import RegistrationPhase, register_pass
from .base import (
    AnalysisResult,
    BasePass,
    PassConfig,
    PassMetrics,
    PassResult,
    PassType,
    TransformResult,
    VerificationResult,
)


@register_pass(
    name="common_subexpression_elimination",
    enabled=True,
    priority=3,
    phase=RegistrationPhase.BOTH
)
class CommonSubexpressionEliminationPass(BasePass):
    """
    公共子表达式消除Pass。

    该Pass识别并消除计算图中相同表达式的冗余计算。它分析图以找到相同的子表达式，
    只计算一次，并在多个使用中重用结果。

    算法流程：
    1. 遍历图以识别CSE的候选操作
    2. 创建表达式键以识别相同的子表达式
    3. 跟踪表达式出现并计算优化潜力
    4. 用对计算结果的引用替换冗余计算
    """

    def __init__(self, config: Optional[PassConfig] = None):
        """初始化公共子表达式消除Pass。"""
        super().__init__("common_subexpression_elimination", PassType.OPTIMIZATION, config or PassConfig())
        self.logger = get_logger(f"ngo.pass.{self.metadata.name}")

        # 配置选项
        self._max_complexity = self.config.custom_options.get('max_complexity', 50)
        self._min_occurrences = self.config.custom_options.get('min_occurrences', 2)
        self._enable_numeric_ops = self.config.custom_options.get('enable_numeric_ops', True)
        self._enable_boolean_ops = self.config.custom_options.get('enable_boolean_ops', True)
        self._enable_comparison_ops = self.config.custom_options.get('enable_comparison_ops', True)

    def _execute_impl(self, context: OptimizationContext) -> Any:
        pass

    def analyze(self, context: OptimizationContext) -> AnalysisResult:
        """
        Analyze the graph to identify common subexpression elimination opportunities.

        Args:
            context: Optimization context containing the graph

        Returns:
            AnalysisResult with identified CSE opportunities
        """
        graph_module = self._get_graph_from_context(context)
        if graph_module is None:
            return AnalysisResult(should_proceed=False, skip_reason="No GraphModule found in context")

        # Find all candidate expressions for CSE
        candidate_expressions = self._find_candidate_expressions(graph_module.graph)

        # Group identical expressions
        expression_groups = self._group_identical_expressions(candidate_expressions)

        # Filter expressions that meet optimization criteria
        optimizable_expressions = self._filter_optimizable_expressions(expression_groups)

        analysis_result = AnalysisResult(should_proceed=True)

        if len(optimizable_expressions) > 0:
            total_eliminations = sum(len(group['nodes']) - 1 for group in optimizable_expressions.values())
            analysis_result.add_opportunity(
                "common_subexpression_elimination",
                f"Found {len(optimizable_expressions)} expression groups to optimize, "
                f"eliminating {total_eliminations} redundant computations",
                impact=total_eliminations
            )
            analysis_result.estimated_improvement = total_eliminations / max(1, len(list(graph_module.graph.nodes)))

        # Store analysis results for transformation phase
        analysis_result.analysis_cache = {
            'optimizable_expressions': optimizable_expressions,
            'total_nodes': len(list(graph_module.graph.nodes)),
            'graph_module': graph_module
        }

        return analysis_result

    def transform(self, context: OptimizationContext, analysis_result: AnalysisResult) -> TransformResult:
        """
        Perform common subexpression elimination transformations.

        Args:
            context: Optimization context containing the graph
            analysis_result: Results from the analysis phase

        Returns:
            TransformResult with transformation details
        """
        if not analysis_result.analysis_cache:
            # Run analysis if not already done
            analysis_result = self.analyze(context)
            if not analysis_result.should_proceed:
                return TransformResult(success=True, modified_graph=False)

        graph_module = self._get_graph_from_context(context)
        if graph_module is None:
            return TransformResult(success=False, error_message="No GraphModule found in context")

        optimizable_expressions = analysis_result.analysis_cache['optimizable_expressions'] if 'optimizable_expressions' in analysis_result.analysis_cache else {}
        original_node_count = analysis_result.analysis_cache['total_nodes'] if 'total_nodes' in analysis_result.analysis_cache else 0

        if len(optimizable_expressions) == 0:
            return TransformResult(success=True, modified_graph=False)

        try:
            # Perform common subexpression elimination
            elimination_results = self._eliminate_common_subexpressions(graph_module, optimizable_expressions)

            # Verify graph integrity
            if not self._verify_graph_integrity(graph_module.graph):
                self.logger.warning("Graph integrity compromised after common subexpression elimination")
                return TransformResult(success=False, error_message="Graph integrity check failed")

            transform_result = TransformResult(
                success=True,
                modified_graph=len(elimination_results) > 0
            )

            if len(elimination_results) > 0:
                total_eliminations = sum(result['eliminated_count'] for result in elimination_results)
                transform_result.add_transformation(
                    f"Eliminated {total_eliminations} redundant computations across "
                    f"{len(elimination_results)} expression groups "
                    f"({total_eliminations/original_node_count*100:.1f}% of graph)"
                )

                self.logger.info(
                    f"Common subexpression elimination completed: eliminated {total_eliminations} "
                    f"redundant computations ({total_eliminations/original_node_count*100:.1f}%)"
                )

            return transform_result

        except Exception as e:
            error_msg = f"Error during common subexpression elimination: {str(e)}"
            self.logger.error(error_msg)
            return TransformResult(success=False, error_message=error_msg)

    def verify(self, context: OptimizationContext, transform_result: TransformResult) -> VerificationResult:
        """
        Verify that common subexpression elimination preserved graph semantics.

        Args:
            context: Optimization context containing the transformed graph
            transform_result: Results from the transformation phase

        Returns:
            VerificationResult with correctness assessment
        """
        if not transform_result.modified_graph:
            return VerificationResult(success=True)

        graph = self._get_graph_from_context(context)
        if graph is None:
            return VerificationResult(success=False, error_message="No graph found in context")

        # Basic integrity checks
        try:
            # Check that output nodes still exist
            output_nodes = self._find_output_nodes(graph)
            if not output_nodes:
                return VerificationResult(
                    success=False,
                    error_message="No output nodes found after common subexpression elimination"
                )

            # Check that the graph is still connected
            input_nodes = self._find_input_nodes(graph)
            for output_node in output_nodes:
                if not self._is_reachable_from_inputs(graph, output_node, input_nodes):
                    return VerificationResult(
                        success=False,
                        error_message=f"Output node not reachable from inputs after common subexpression elimination"
                    )

            self.logger.info("Common subexpression elimination verification passed")
            return VerificationResult(success=True)

        except Exception as e:
            error_msg = f"Verification failed: {str(e)}"
            self.logger.error(error_msg)
            return VerificationResult(success=False, error_message=error_msg)

    def _find_candidate_expressions(self, graph: Any) -> List[Dict[str, Any]]:
        """Find candidate expressions for common subexpression elimination."""
        candidates = []

        try:
            for node in graph.nodes:
                if self._is_optimizable_operation(node):
                    complexity = self._calculate_expression_complexity(node)
                    if complexity <= self._max_complexity:
                        candidates.append({
                            'node': node,
                            'complexity': complexity,
                            'expression_key': self._create_expression_key(node)
                        })
        except Exception as e:
            self.logger.warning(f"Error finding candidate expressions: {e}")

        return candidates

    def _is_optimizable_operation(self, node: Any) -> bool:
        """Check if a node represents an optimizable operation for CSE."""
        try:
            if not hasattr(node, 'op') or node.op != 'call_function':
                return False

            if not hasattr(node, 'target') or not node.target:
                return False

            # Skip nodes with side effects
            if self._has_side_effects(node):
                return False

            # Check operation type based on configuration
            target_str = str(node.target).lower()

            # Numeric operations
            if self._enable_numeric_ops:
                if any(op in target_str for op in ['add', 'sub', 'mul', 'div', 'pow']):
                    return True

            # Boolean operations
            if self._enable_boolean_ops:
                if any(op in target_str for op in ['and', 'or', 'not']):
                    return True

            # Comparison operations
            if self._enable_comparison_ops:
                if any(op in target_str for op in ['eq', 'ne', 'lt', 'le', 'gt', 'ge']):
                    return True

            return False

        except Exception:
            return False

    def _has_side_effects(self, node: Any) -> bool:
        """Check if a node has potential side effects."""
        try:
            if hasattr(node, 'op'):
                if node.op == 'store':
                    return True

                if hasattr(node, 'target') and node.target:
                    target_str = str(node.target).lower()
                    if any(op in target_str for op in ['store', 'print', 'open', 'write', 'read', 'close']):
                        return True

            return False

        except Exception:
            return True  # Conservative approach

    def _create_expression_key(self, node: Any) -> str:
        """Create a unique key for identifying identical expressions."""
        try:
            if not hasattr(node, 'target') or not node.target:
                return str(id(node))

            # Create key based on operation and argument structure
            key_parts = [str(node.target)]

            if hasattr(node, 'args'):
                for arg in node.args:
                    if isinstance(arg, (int, float, bool, str)):
                        key_parts.append(str(arg))
                    elif hasattr(arg, 'op'):
                        if arg.op == 'placeholder':
                            key_parts.append(f"placeholder_{getattr(arg, 'name', 'unknown')}")
                        elif arg.op == 'get_attr':
                            key_parts.append(f"get_attr_{getattr(arg, 'name', 'unknown')}")
                        else:
                            # For other operations, use a generic representation
                            key_parts.append(f"node_{getattr(arg, 'name', 'unknown')}")
                    else:
                        key_parts.append(str(type(arg).__name__))

            return "|".join(key_parts)

        except Exception:
            return str(id(node))

    def _calculate_expression_complexity(self, node: Any) -> int:
        """Calculate the complexity of an expression for optimization prioritization."""
        try:
            complexity = 1  # Base complexity for the operation

            if hasattr(node, 'args'):
                for arg in node.args:
                    if isinstance(arg, (tuple, list)):
                        complexity += len(arg)
                    else:
                        complexity += 1

            return complexity

        except Exception:
            return float('inf')

    def _group_identical_expressions(self, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group identical expressions together."""
        groups = {}

        for candidate in candidates:
            expr_key = candidate['expression_key']
            if expr_key not in groups:
                groups[expr_key] = {
                    'nodes': [],
                    'complexity': candidate['complexity'],
                    'expression_key': expr_key
                }
            groups[expr_key]['nodes'].append(candidate['node'])

        return groups

    def _filter_optimizable_expressions(self, expression_groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter expressions that meet optimization criteria."""
        optimizable = {}

        for expr_key, group in expression_groups.items():
            if len(group['nodes']) >= self._min_occurrences:
                # Additional checks can be added here
                optimizable[expr_key] = group

        return optimizable

    def _eliminate_common_subexpressions(self, graph_module: Any, optimizable_expressions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Eliminate common subexpressions in the torch.fx GraphModule."""
        results = []

        try:
            for expr_key, group in optimizable_expressions.items():
                nodes = group['nodes']
                if len(nodes) < self._min_occurrences:
                    continue

                # Keep the first occurrence and eliminate others
                target_node = nodes[0]
                nodes_to_eliminate = nodes[1:]

                eliminated_count = 0
                for node_to_eliminate in nodes_to_eliminate:
                    if self._replace_node_uses(graph_module, node_to_eliminate, target_node):
                        eliminated_count += 1

                if eliminated_count > 0:
                    results.append({
                        'expression_key': expr_key,
                        'total_occurrences': len(nodes),
                        'eliminated_count': eliminated_count,
                        'complexity': group['complexity']
                    })

                    self.logger.debug(f"Eliminated {eliminated_count} occurrences of expression {expr_key}")

        except Exception as e:
            self.logger.error(f"Error eliminating common subexpressions: {e}")

        return results

    def _replace_node_uses(self, graph_module: Any, old_node: Any, new_node: Any) -> bool:
        """Replace all uses of old_node with new_node in torch.fx GraphModule."""
        try:
            # Replace all uses of the old node with the new node
            old_node.replace_all_uses_with(new_node)

            # Remove the old node if it has no more users
            if len(old_node.users) == 0:
                graph_module.graph.erase_node(old_node)

            return True

        except Exception as e:
            self.logger.warning(f"Failed to replace node uses: {e}")
            return False

    def _remove_node_from_graph(self, graph_module: Any, node: Any) -> bool:
        """Remove a node from the torch.fx GraphModule."""
        try:
            if len(node.users) == 0:
                graph_module.graph.erase_node(node)
                return True
            return False

        except Exception as e:
            self.logger.warning(f"Failed to remove node from graph: {e}")
            return False

    def _find_output_nodes(self, graph: Any) -> List[Any]:
        """Find output nodes in the graph."""
        output_nodes = []

        try:
            for node in graph.nodes:
                if hasattr(node, 'op') and node.op == 'output':
                    output_nodes.append(node)
                elif hasattr(node, 'name') and 'output' in node.name.lower():
                    output_nodes.append(node)
                elif hasattr(node, 'users') and len(node.users) == 0:
                    if hasattr(node, 'op') and node.op not in ['placeholder', 'get_attr']:
                        if node.op == 'call_function' and hasattr(node, 'name'):
                            if any(keyword in node.name.lower() for keyword in ['output', 'result', 'final']):
                                output_nodes.append(node)
                        else:
                            output_nodes.append(node)
        except Exception:
            try:
                nodes = list(graph.nodes)
                if nodes:
                    output_nodes.append(nodes[-1])
            except Exception:
                pass

        return output_nodes

    def _find_input_nodes(self, graph: Any) -> List[Any]:
        """Find input nodes in the graph."""
        input_nodes = []

        try:
            for node in graph.nodes:
                if hasattr(node, 'op') and node.op == 'placeholder':
                    input_nodes.append(node)
                elif hasattr(node, 'name') and 'input' in node.name.lower():
                    input_nodes.append(node)
        except Exception:
            pass

        return input_nodes

    def _verify_graph_integrity(self, graph: Any) -> bool:
        """Verify that the graph maintains basic integrity."""
        try:
            for node in graph.nodes:
                if hasattr(node, 'args'):
                    for arg in node.args:
                        if hasattr(arg, 'name') and hasattr(arg, 'graph'):
                            if arg.graph != graph:
                                return False
            return True

        except Exception:
            return False

    def _is_reachable_from_inputs(self, graph: Any, node: Any, input_nodes: List[Any]) -> bool:
        """Check if a node is reachable from input nodes."""
        visited = set()
        queue = deque([node])

        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node in input_nodes:
                return True

            try:
                if hasattr(current_node, 'args'):
                    for arg in current_node.args:
                        if hasattr(arg, 'name') and arg not in visited:
                            queue.append(arg)
            except Exception:
                pass

        return False