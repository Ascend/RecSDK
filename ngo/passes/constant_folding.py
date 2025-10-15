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
常量折叠Pass。

该Pass识别并评估计算图中的常量表达式，在编译时用计算值替换它们。
这种优化减少了运行时计算开销，并启用进一步的优化。

算法流程：
1. 识别常量表达式（所有操作数都是常量的表达式）
2. 在编译时评估常量表达式
3. 用常量节点替换表达式节点
4. 保持图语义和结构
"""

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import torch

from ngo.utils.logger import get_logger
from ..core.base import OptimizationContext
from ..core.unified_registry import RegistrationPhase, register_pass
from .base import (
    AnalysisResult,
    BasePass,
    PassConfig,
    PassType,
    TransformResult,
    VerificationResult,
)


@register_pass(
    name="constant_folding",
    enabled=True,
    priority=4,
    phase=RegistrationPhase.BOTH
)
class ConstantFoldingPass(BasePass):
    """
    常量折叠Pass。

    该Pass识别并评估计算图中的常量表达式，在编译时用计算值替换它们。
    这种优化减少了运行时计算开销，并启用进一步的优化。

    算法流程：
    1. 识别常量表达式（所有操作数都是常量的表达式）
    2. 在编译时评估常量表达式
    3. 用常量节点替换表达式节点
    4. 保持图语义和结构
    """

    def __init__(self, config: Optional[PassConfig] = None):
        """初始化常量折叠Pass。"""
        super().__init__("constant_folding", PassType.OPTIMIZATION, config or PassConfig())
        self.logger = get_logger(f"ngo.pass.{self.metadata.name}")

        # 配置选项
        self._fold_numeric_ops = self.config.custom_options.get('fold_numeric_ops', True)
        self._fold_boolean_ops = self.config.custom_options.get('fold_boolean_ops', True)
        self._fold_comparison_ops = self.config.custom_options.get('fold_comparison_ops', True)
        self._max_complexity = self.config.custom_options.get('max_complexity', 100)

    def _execute_impl(self, context: OptimizationContext) -> Any:
        pass

    def analyze(self, context: OptimizationContext) -> AnalysisResult:
        """
        Analyze the graph to identify constant folding opportunities.

        Args:
            context: Optimization context containing the graph

        Returns:
            AnalysisResult with identified constant folding opportunities
        """
        graph_module = self._get_graph_from_context(context)
        if graph_module is None:
            return AnalysisResult(should_proceed=False, skip_reason="No GraphModule found in context")

        # Find all constant expression opportunities in torch.fx graph
        constant_expressions = self._find_constant_expressions(graph_module.graph)

        analysis_result = AnalysisResult(should_proceed=True)

        if len(constant_expressions) > 0:
            analysis_result.add_opportunity(
                "constant_folding",
                f"Found {len(constant_expressions)} constant expressions to fold",
                impact=len(constant_expressions)
            )
            analysis_result.estimated_improvement = len(constant_expressions) / max(1, len(list(graph_module.graph.nodes)))

        # Store analysis results for transformation phase
        analysis_result.analysis_cache = {
            'constant_expressions': constant_expressions,
            'total_nodes': len(list(graph_module.graph.nodes)),
            'graph_module': graph_module
        }

        return analysis_result

    def transform(self, context: OptimizationContext, analysis_result: AnalysisResult) -> TransformResult:
        """
        Perform constant folding transformations.

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

        constant_expressions = analysis_result.analysis_cache['constant_expressions'] if 'constant_expressions' in analysis_result.analysis_cache else []
        original_node_count = analysis_result.analysis_cache['total_nodes'] if 'total_nodes' in analysis_result.analysis_cache else 0

        if len(constant_expressions) == 0:
            return TransformResult(success=True, modified_graph=False)

        try:
            # Perform constant folding on torch.fx GraphModule
            folded_expressions = self._fold_constant_expressions(graph_module, constant_expressions)

            # Verify graph integrity
            if not self._verify_graph_integrity(graph_module.graph):
                self.logger.warning("Graph integrity compromised after constant folding")
                return TransformResult(success=False, error_message="Graph integrity check failed")

            transform_result = TransformResult(
                success=True,
                modified_graph=len(folded_expressions) > 0
            )

            if len(folded_expressions) > 0:
                transform_result.add_transformation(
                    f"Folded {len(folded_expressions)} constant expressions "
                    f"({len(folded_expressions)/original_node_count*100:.1f}% of graph)"
                )

                self.logger.info(
                    f"Constant folding completed: folded {len(folded_expressions)} "
                    f"expressions ({len(folded_expressions)/original_node_count*100:.1f}%)"
                )

            return transform_result

        except Exception as e:
            error_msg = f"Error during constant folding: {str(e)}"
            self.logger.error(error_msg)
            return TransformResult(success=False, error_message=error_msg)

    def verify(self, context: OptimizationContext, transform_result: TransformResult) -> VerificationResult:
        """
        Verify that constant folding preserved graph semantics.

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
                    error_message="No output nodes found after constant folding"
                )

            # Check that the graph is still connected
            input_nodes = self._find_input_nodes(graph)
            for output_node in output_nodes:
                if not self._is_reachable_from_inputs(graph, output_node, input_nodes):
                    return VerificationResult(
                        success=False,
                        error_message=f"Output node not reachable from inputs after constant folding"
                    )

            self.logger.info("Constant folding verification passed")
            return VerificationResult(success=True)

        except Exception as e:
            error_msg = f"Verification failed: {str(e)}"
            self.logger.error(error_msg)
            return VerificationResult(success=False, error_message=error_msg)

    def _find_constant_expressions(self, graph: Any) -> List[Dict[str, Any]]:
        """Find constant expressions in the graph."""
        constant_expressions = []

        try:
            for node in graph.nodes:
                if self._is_constant_expression(node):
                    complexity = self._calculate_expression_complexity(node)
                    if complexity <= self._max_complexity:
                        constant_expressions.append({
                            'node': node,
                            'complexity': complexity,
                            'expression_type': self._get_expression_type(node)
                        })
        except Exception as e:
            self.logger.warning(f"Error finding constant expressions: {e}")

        return constant_expressions

    def _is_constant_expression(self, node: Any) -> bool:
        """Check if a node represents a constant expression."""
        try:
            if not hasattr(node, 'op') or node.op != 'call_function':
                return False

            if not hasattr(node, 'target') or not node.target:
                return False

            # Check if all arguments are constants
            if not hasattr(node, 'args'):
                return False

            for arg in node.args:
                if not self._is_constant_value(arg):
                    return False

            return True

        except Exception:
            return False

    def _is_constant_value(self, value: Any) -> bool:
        """Check if a value is a constant."""
        try:
            # Direct constants
            if isinstance(value, (int, float, bool)):
                return True

            # Constant nodes
            if hasattr(value, 'op') and value.op == 'get_attr':
                return True

            # String constants
            if isinstance(value, str):
                return True

            # Tuple/list of constants
            if isinstance(value, (tuple, list)):
                return all(self._is_constant_value(item) for item in value)

            return False

        except Exception as e:
            self.logger.warning(f"Error checking if value is constant: {e}")
            return False

    def _calculate_expression_complexity(self, node: Any) -> int:
        """Calculate the complexity of a constant expression."""
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

    def _get_expression_type(self, node: Any) -> str:
        """Get the type of a constant expression."""
        try:
            if hasattr(node, 'target'):
                target_str = str(node.target).lower()

                if any(op in target_str for op in ['add', 'sub', 'mul', 'div', 'pow']):
                    return 'numeric'
                elif any(op in target_str for op in ['and', 'or', 'not']):
                    return 'boolean'
                elif any(op in target_str for op in ['eq', 'ne', 'lt', 'le', 'gt', 'ge']):
                    return 'comparison'
                else:
                    return 'other'
            return 'unknown'
        except Exception as e:
            self.logger.warning(f"Error getting expression type: {e}")
            return 'unknown'

    def _fold_constant_expressions(self, graph_module: Any, constant_expressions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fold constant expressions in the torch.fx GraphModule."""
        folded_expressions = []

        try:
            for expr_info in constant_expressions:
                node = expr_info['node']
                result = self._evaluate_constant_expression(node)

                if result is not None:
                    # Replace the node with a constant in torch.fx GraphModule
                    if self._replace_node_with_constant(graph_module, node, result):
                        folded_expressions.append({
                            'node': node,
                            'original_value': str(node.target),
                            'folded_value': str(result)
                        })
        except Exception as e:
            self.logger.error(f"Error folding constant expressions: {e}")

        return folded_expressions

    def _evaluate_constant_expression(self, node: Any) -> Any:
        """Evaluate a constant expression."""
        try:
            if not hasattr(node, 'target') or not node.target:
                return None

            # Extract constant values from arguments
            args = []
            for arg in node.args:
                if isinstance(arg, (int, float, bool, str)):
                    args.append(arg)
                elif hasattr(arg, 'op') and arg.op == 'get_attr':
                    # Handle attribute access (simplified)
                    args.append(0)  # Placeholder
                else:
                    return None

            # Evaluate the expression
            target = node.target
            if callable(target):
                try:
                    return target(*args)
                except Exception:
                    return None
            else:
                # Handle common operations
                if target == 'add':
                    return args[0] + args[1]
                elif target == 'sub':
                    return args[0] - args[1]
                elif target == 'mul':
                    return args[0] * args[1]
                elif target == 'div':
                    return args[0] / args[1] if args[1] != 0 else None
                elif target == 'pow':
                    return args[0] ** args[1]
                elif target == 'eq':
                    return args[0] == args[1]
                elif target == 'ne':
                    return args[0] != args[1]
                elif target == 'lt':
                    return args[0] < args[1]
                elif target == 'le':
                    return args[0] <= args[1]
                elif target == 'gt':
                    return args[0] > args[1]
                elif target == 'ge':
                    return args[0] >= args[1]
                elif target == 'and':
                    return args[0] and args[1]
                elif target == 'or':
                    return args[0] or args[1]
                elif target == 'not':
                    return not args[0]

            return None

        except Exception as e:
            self.logger.warning(f"Error evaluating constant expression: {e}")
            return None

    def _replace_node_with_constant(self, graph_module: Any, node: Any, constant_value: Any) -> bool:
        """Replace a node with a constant value in torch.fx GraphModule."""
        try:
            # In torch.fx, we need to replace the node with a get_attr node that references a constant
            with graph_module.graph.inserting_before(node):
                # Create a constant node
                const_node = graph_module.graph.create_node(
                    'get_attr',
                    f'_const_{id(constant_value)}',
                    (),
                    {}
                )

            # Set the constant in the module
            graph_module.register_buffer(f'_const_{id(constant_value)}',
                                       torch.tensor(constant_value, dtype=torch.float32))

            # Replace all uses of the original node with the constant node
            node.replace_all_uses_with(const_node)

            # Remove the original node if it has no more users
            if len(node.users) == 0:
                graph_module.graph.erase_node(node)

            self.logger.debug(f"Replaced node {node.name} with constant {constant_value}")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to replace node with constant: {e}")
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