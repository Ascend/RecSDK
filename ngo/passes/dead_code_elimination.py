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
死代码消除Pass。

该Pass识别并移除计算图中不可达的节点。它从输出节点开始执行可达性分析，
并消除任何输出都无法到达的节点。

算法流程：
1. 识别所有输出节点（向图输出提供数据的节点）
2. 从输出开始执行反向可达性分析
3. 标记所有可达的节点
4. 移除不可达的节点，同时保持图结构
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set

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
    name="dead_code_elimination",
    enabled=True,
    priority=5,
    phase=RegistrationPhase.BOTH
)
class DeadCodeEliminationPass(BasePass):
    """
    死代码消除Pass。

    该Pass识别并移除计算图中不可达的节点。它从输出节点开始执行可达性分析，
    并消除任何输出都无法到达的节点。

    算法流程：
    1. 识别所有输出节点（向图输出提供数据的节点）
    2. 从输出开始执行反向可达性分析
    3. 标记所有可达的节点
    4. 移除不可达的节点，同时保持图结构
    """

    def __init__(self, config: Optional[PassConfig] = None):
        """初始化死代码消除Pass。"""
        super().__init__("dead_code_elimination", PassType.CLEANUP, config or PassConfig())
        self.logger = get_logger(f"ngo.pass.{self.metadata.name}")

        # 配置选项
        self._preserve_placeholders = self.config.custom_options.get('preserve_placeholders', True)
        self._preserve_get_attr = self.config.custom_options.get('preserve_get_attr', True)
        self._aggressive_mode = self.config.custom_options.get('aggressive_mode', False)

    def _execute_impl(self, context: OptimizationContext) -> Any:
        pass

    def analyze(self, context: OptimizationContext) -> AnalysisResult:
        """
        Analyze the graph to identify dead code opportunities.

        Args:
            context: Optimization context containing the graph

        Returns:
            AnalysisResult with identified dead code opportunities
        """
        graph_module = self._get_graph_from_context(context)
        if graph_module is None:
            return AnalysisResult(should_proceed=False, skip_reason="No GraphModule found in context")

        # Find all nodes and output nodes in the torch.fx graph
        all_nodes = list(graph_module.graph.nodes)
        output_nodes = self._find_output_nodes(graph_module.graph)

        # Perform reachability analysis
        reachable_nodes = self._find_reachable_nodes(graph_module.graph, output_nodes)
        unreachable_nodes = set(all_nodes) - set(reachable_nodes)

        # Filter out nodes we want to preserve
        removable_nodes = self._filter_removable_nodes(unreachable_nodes)

        analysis_result = AnalysisResult(should_proceed=True)

        if len(removable_nodes) > 0:
            analysis_result.add_opportunity(
                "dead_code_elimination",
                f"Found {len(removable_nodes)} unreachable nodes to remove",
                impact=len(removable_nodes)
            )
            analysis_result.estimated_improvement = len(removable_nodes) / len(all_nodes) if all_nodes else 0.0

        # Store analysis results for transformation phase
        analysis_result.analysis_cache = {
            'removable_nodes': removable_nodes,
            'reachable_nodes': reachable_nodes,
            'total_nodes': len(all_nodes),
            'graph_module': graph_module
        }

        return analysis_result

    def transform(self, context: OptimizationContext, analysis_result: AnalysisResult) -> TransformResult:
        """
        Remove dead code from the graph.

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

        if not analysis_result.analysis_cache:
            return TransformResult(success=False, error_message="No analysis result found in cache")
        
        removable_nodes = analysis_result.analysis_cache['removable_nodes'] if 'removable_nodes' in analysis_result.analysis_cache else set()
        original_node_count = analysis_result.analysis_cache['total_nodes'] if 'total_nodes' in analysis_result.analysis_cache else 0

        if len(removable_nodes) == 0:
            return TransformResult(success=True, modified_graph=False)

        try:
            # Remove unreachable nodes from torch.fx graph
            removed_nodes = self._remove_nodes_from_graph(graph_module, removable_nodes)

            # Verify graph integrity
            if not self._verify_graph_integrity(graph_module.graph):
                # Rollback changes if graph integrity is compromised
                self.logger.warning("Graph integrity compromised after dead code elimination")
                return TransformResult(success=False, error_message="Graph integrity check failed")

            transform_result = TransformResult(
                success=True,
                modified_graph=len(removed_nodes) > 0
            )

            transform_result.add_transformation(
                f"Removed {len(removed_nodes)} unreachable nodes "
                f"({len(removed_nodes)/original_node_count*100:.1f}% of graph)"
            )

            if len(removed_nodes) > 0:
                transform_result.add_warning(
                    f"Dead code elimination removed {len(removed_nodes)} nodes"
                )

            self.logger.info(
                f"Dead code elimination completed: removed {len(removed_nodes)} "
                f"nodes ({len(removed_nodes)/original_node_count*100:.1f}%)"
            )

            return transform_result

        except Exception as e:
            error_msg = f"Error during dead code elimination: {str(e)}"
            self.logger.error(error_msg)
            return TransformResult(success=False, error_message=error_msg)

    def verify(self, context: OptimizationContext, transform_result: TransformResult) -> VerificationResult:
        """
        Verify that dead code elimination preserved graph semantics.

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
                    error_message="No output nodes found after dead code elimination"
                )

            # Check that the graph is still connected from inputs to outputs
            input_nodes = self._find_input_nodes(graph)
            for output_node in output_nodes:
                if not self._is_reachable_from_inputs(graph, output_node, input_nodes):
                    return VerificationResult(
                        success=False,
                        error_message=f"Output node {output_node} not reachable from inputs"
                    )

            self.logger.info("Dead code elimination verification passed")
            return VerificationResult(success=True)

        except Exception as e:
            error_msg = f"Verification failed: {str(e)}"
            self.logger.error(error_msg)
            return VerificationResult(success=False, error_message=error_msg)

    def _find_output_nodes(self, graph: Any) -> List[Any]:
        """Find output nodes in the graph."""
        output_nodes = []

        try:
            # In torch.fx, output nodes typically have users that are None
            # or are the final output node of the graph
            for node in graph.nodes:
                # Check if this node is explicitly marked as output
                if hasattr(node, 'op') and node.op == 'output':
                    output_nodes.append(node)
                # Check if node name contains 'output' (more specific than just empty users)
                elif hasattr(node, 'name') and 'output' in node.name.lower():
                    output_nodes.append(node)
                # Only consider nodes with empty users as output if they have specific characteristics
                elif hasattr(node, 'users') and len(node.users) == 0:
                    # Don't consider nodes with certain operations as output just because they have no users
                    if hasattr(node, 'op') and node.op not in ['placeholder', 'get_attr']:
                        # For call_function nodes, be more conservative
                        if node.op == 'call_function' and hasattr(node, 'name'):
                            # Only consider it an output if the name strongly suggests it
                            if any(keyword in node.name.lower() for keyword in ['output', 'result', 'final']):
                                output_nodes.append(node)
                        else:
                            output_nodes.append(node)
        except Exception:
            # Fallback: assume the last node is the output
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

    def _find_reachable_nodes(self, graph: Any, output_nodes: List[Any]) -> Set[Any]:
        """Perform backward reachability analysis from output nodes."""
        reachable = set()
        visited = set()
        queue = deque(output_nodes)

        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue

            visited.add(current_node)
            reachable.add(current_node)

            # Add all nodes that this node depends on
            try:
                if hasattr(current_node, 'args'):
                    for arg in current_node.args:
                        if hasattr(arg, 'nodes'):  # This is a subgraph
                            continue
                        elif hasattr(arg, 'name'):  # This is a node
                            if arg not in visited:
                                queue.append(arg)
                        elif hasattr(arg, '__iter__'):  # This might be a list of nodes
                            for item in arg:
                                if hasattr(item, 'name') and item not in visited:
                                    queue.append(item)
            except Exception:
                pass

        return reachable

    def _filter_removable_nodes(self, unreachable_nodes: Set[Any]) -> Set[Any]:
        """Filter nodes that can be safely removed."""
        removable = set()

        for node in unreachable_nodes:
            # Skip placeholder nodes if configured to preserve them
            if self._preserve_placeholders and hasattr(node, 'op') and node.op == 'placeholder':
                continue

            # Skip get_attr nodes if configured to preserve them
            if self._preserve_get_attr and hasattr(node, 'op') and node.op == 'get_attr':
                continue

            # Skip nodes with side effects
            if self._has_side_effects(node):
                continue

            removable.add(node)

        return removable

    def _has_side_effects(self, node: Any) -> bool:
        """Check if a node has potential side effects."""
        try:
            # Nodes with certain operations may have side effects
            if hasattr(node, 'op'):
                # Store operations might have side effects
                if node.op == 'store':
                    return True

                if hasattr(node, 'target') and node.target:
                    target_str = str(node.target).lower()
                    # Store operations might have side effects
                    if 'store' in target_str:
                        return True

                    # Print operations have side effects
                    if 'print' in target_str:
                        return True

                    # File operations have side effects
                    if any(op in target_str for op in ['open', 'write', 'read', 'close']):
                        return True

            return False

        except Exception:
            # Conservative approach: assume side effects if we can't determine
            return True

    def _remove_nodes_from_graph(self, graph_module: Any, nodes_to_remove: Set[Any]) -> List[Any]:
        """Remove nodes from the torch.fx GraphModule."""
        removed_nodes = []

        try:
            # For torch.fx GraphModule, we need to properly handle node removal
            for node in nodes_to_remove:
                try:
                    # Check if node exists and can be safely removed
                    if node in graph_module.graph.nodes:
                        # In torch.fx, we need to handle node users carefully
                        if len(node.users) == 0 or all(user.op == 'output' for user in node.users):
                            # Safe to remove - no dependent nodes or only output users
                            graph_module.graph.erase_node(node)
                            removed_nodes.append(node)
                        else:
                            # Node has users that depend on it - skip removal to maintain graph integrity
                            self.logger.debug(f"Skipping removal of node {node.name} - has dependent users")
                    else:
                        # Node already removed or not in graph
                        self.logger.debug(f"Node {node.name} not found in graph")
                except Exception as e:
                    # Continue with other nodes if one fails
                    self.logger.warning(f"Failed to remove node {getattr(node, 'name', 'unknown')}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error removing nodes from GraphModule: {e}")

        return removed_nodes

    def _verify_graph_integrity(self, graph: Any) -> bool:
        """Verify that the graph maintains basic integrity."""
        try:
            # Check that all nodes have valid references
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

            # Check if this is an input node
            if current_node in input_nodes:
                return True

            # Add predecessor nodes
            try:
                if hasattr(current_node, 'args'):
                    for arg in current_node.args:
                        if hasattr(arg, 'name') and arg not in visited:
                            queue.append(arg)
            except Exception:
                pass

        return False