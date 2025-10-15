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
AddLayerNorm优化Pattern。

该模块提供AddLayerNorm Pattern实现，用于将Add + LayerNorm操作融合
为npu_add_layer_norm操作，以实现昇腾NPU硬件优化。
"""

import operator
from typing import Any, Dict, List, Optional

import torch
from torch.fx import GraphModule, Node

from ngo.utils.logger import get_logger
from ..core.base import ComponentMetadata, OptimizationContext
from ..core.unified_registry import RegistrationPhase, register_pattern
from ..utils.optional_deps import has_torch_npu, torch_npu
from .base import (
    BasePattern,
    BaseStrategy,
    PatternMatchResult,
    StrategyExecutionResult,
    StrategyPriority,
    register_strategy,
)

if has_torch_npu():
    @torch.library.register_fake("npu::npu_add_layer_norm")
    def npu_add_layer_norm_fake(x1, x2, gamma, beta, epsilon, additional_output):
        if x1.dim() != x2.dim():
            raise ValueError("x1 and x2 must have the same number of dimensions")
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape")
        
        mean_output = torch.empty(x1.shape[:-1] + (1,), dtype=x1.dtype, device=x1.device)
        rstd_output = torch.empty(x1.shape[:-1] + (1,), dtype=x1.dtype, device=x1.device)
        
        return torch.empty_like(x1), mean_output, rstd_output, torch.empty_like(x1)


class AddLayerNormStrategy(BaseStrategy):
    """
    将Add + LayerNorm操作融合为npu_add_layer_norm的策略。

    该策略将匹配的Add + LayerNorm Pattern替换为优化的npu_add_layer_norm操作，
    用于昇腾NPU硬件。
    """

    def __init__(self):
        metadata = ComponentMetadata(
            name="AddLayerNormStrategy",
            version="1.0.0",
            description="Fuses Add + LayerNorm into npu_add_layer_norm operation",
        )
        super().__init__(metadata)
        self._logger = get_logger(f"ngo.strategy.{self.__class__.__name__}")

    def execute(self, graph_module: GraphModule, match_result: PatternMatchResult) -> StrategyExecutionResult:
        """
        执行Add + LayerNorm融合策略。

        Args:
            graph_module: 要转换的图模块
            match_result: Pattern匹配结果

        Returns:
            StrategyExecutionResult，指示成功和详细信息
        """
        if not match_result.matched:
            return StrategyExecutionResult(success=False)

        try:
            transformed_nodes = []

            for layernorm_node in match_result.matched_nodes:
                # Find the corresponding add operation
                add_node = self._find_add_node(layernorm_node, graph_module)
                if add_node is None:
                    self._logger.warning(f"Could not find add node for {layernorm_node}")
                    continue

                # Perform the fusion
                if self._fuse_add_layernorm(add_node, layernorm_node, graph_module):
                    transformed_nodes.extend([add_node, layernorm_node])

            success = len(transformed_nodes) > 0
            if success:
                self._logger.info(f"Successfully fused {len(transformed_nodes)//2} AddLayerNorm patterns")

            return StrategyExecutionResult(
                success=success,
                transformed_nodes=transformed_nodes,
                metadata={"fused_patterns": len(transformed_nodes) // 2},
            )

        except Exception as e:
            self._logger.error(f"AddLayerNorm fusion failed: {e}")
            return StrategyExecutionResult(success=False)

    def _is_add_operation(self, node: Node, graph_module: GraphModule) -> bool:
        """Check if node represents an add operation."""
        if node.op == "call_function" and node.target == torch.add:
            return True

        if node.op == "call_function" and node.target == torch.Tensor.__add__:
            return True

        # Check for built-in add function
        if node.op == "call_function" and str(node.target) == "<built-in function add>":
            return True

        # Check for add module (less common)
        if node.op == "call_module":
            if hasattr(graph_module, node.target):
                module = getattr(graph_module, node.target)
                return isinstance(module, (torch.nn.Identity,))  # Add modules are rare

        return False

    def _find_add_node(self, layernorm_node: Node, graph_module: GraphModule) -> Optional[Node]:
        """Find the add operation that feeds into the layer norm."""
        for arg in layernorm_node.args:
            if isinstance(arg, Node) and self._is_add_operation(arg, graph_module):
                return arg
        return None

    def _fuse_add_layernorm(self, add_node: Node, layernorm_node: Node, graph_module: GraphModule) -> bool:
        """
        Fuse Add + LayerNorm into npu_add_layer_norm.

        Args:
            add_node: The add operation node
            layernorm_node: The layer normalization node
            graph_module: The containing graph module

        Returns:
            True if fusion was successful
        """
        try:
            # Get the graph
            graph = graph_module.graph

            # Extract parameters for the fused operation
            # Add operation: add(input, other) or add(input, other, *, alpha=1)
            add_args = add_node.args
            input_tensor = add_args[0]
            other_tensor = add_args[1]

            # LayerNorm parameters: layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
            ln_args = layernorm_node.args
            weight = ln_args[2]
            bias = ln_args[3]
            eps = ln_args[4]

            # Check if torch_npu is available
            if not has_torch_npu() or not hasattr(torch_npu, "npu_add_layer_norm"):
                self._logger.warning("torch_npu.npu_add_layer_norm not available, skipping fusion")
                return False

            # Step 1: Create the fused operation after the add_node to maintain correct topology
            with graph.inserting_after(add_node):
                # Create the fused operation call
                # npu_add_layer_norm returns (output, mean, rstd, y)
                fused_node = graph.call_function(
                    torch_npu.npu_add_layer_norm,
                    args=(input_tensor, other_tensor, weight, bias, eps, True),
                    kwargs={}
                )

            # Step 2: Create getitem node after the fused node
            with graph.inserting_after(fused_node):
                # Extract only the first output (the normalized result)
                getitem_node = graph.call_function(
                    operator.getitem,
                    args=(fused_node, 0)
                )

            # Step 3: Replace all uses of the original layer norm with the getitem output
            # This maintains the correct data flow
            layernorm_node.replace_all_uses_with(getitem_node)

            # Step 4: Remove the original nodes (they're now unused)
            # Important: remove layernorm_node first since it depends on add_node
            graph.erase_node(layernorm_node)
            graph.erase_node(add_node)

            self._logger.debug(f"Successfully fused Add + LayerNorm into npu_add_layer_norm")
            return True

        except Exception as e:
            self._logger.error(f"Fusion failed for nodes {add_node} and {layernorm_node}: {e}")
            return False

    def initialize(self):
        """Initialize the strategy."""
        super().initialize()
        self._logger.info("AddLayerNormStrategy initialized")

    def _execute_impl(self, context: OptimizationContext) -> Dict[str, Any]:
        """Execute the strategy implementation."""
        return {"strategy_executed": True}
    
    
@register_strategy(AddLayerNormStrategy, StrategyPriority.NORMAL)
@register_pattern(
    name="add_layernorm",
    enabled=True,
    priority=5,
    phase=RegistrationPhase.BOTH
)
class AddLayerNormPattern(BasePattern):
    """
    Pattern that matches Add + LayerNorm subgraphs for fusion.

    This pattern identifies sequences where an add operation is followed by
    a layer normalization operation, which can be fused into a single
    npu_add_layer_norm operation on Ascend NPU hardware.
    """

    def __init__(self):
        metadata = ComponentMetadata(
            name="AddLayerNormPattern",
            version="1.0.0",
            description="Matches Add + LayerNorm subgraphs for NPU fusion",
        )
        super().__init__(metadata)
        self._logger = get_logger(f"ngo.pattern.{self.__class__.__name__}")

    def _is_available(self) -> bool:
        """Check if the pattern is available for execution."""
        return has_torch_npu() and hasattr(torch_npu, "npu_add_layer_norm")

    def match(self, graph_module: GraphModule, nodes: List[Node] = None) -> PatternMatchResult:
        """
        Match Add + LayerNorm patterns in the graph.

        Args:
            graph_module: The graph module to search for patterns
            nodes: Optional list of nodes to consider (if None, search all nodes)

        Returns:
            PatternMatchResult indicating match success and details
        """
        matched_nodes = []
        match_score = 0.0

        # Get all nodes to search
        if nodes is None:
            nodes = list(graph_module.graph.nodes)

        # Search for Add -> LayerNorm patterns
        for node in nodes:
            if self._is_add_layernorm_pattern(node, graph_module):
                matched_nodes.append(node)
                match_score += 1.0

        if matched_nodes:
            self._logger.info(f"Found {len(matched_nodes)} AddLayerNorm patterns")
            return PatternMatchResult(
                matched=True,
                matched_nodes=matched_nodes,
                match_score=match_score,
                metadata={"pattern_count": len(matched_nodes)}
            )

        return PatternMatchResult(matched=False)

    def _is_add_layernorm_pattern(self, node: Node, graph_module: GraphModule) -> bool:
        """
        Check if a node represents an Add + LayerNorm pattern.

        Args:
            node: The node to check
            graph_module: The containing graph module

        Returns:
            True if this node is part of an Add + LayerNorm pattern
        """
        # Check if node is a layer normalization operation
        is_layernorm = False
        if node.op == "call_function":
            is_layernorm = node.target == torch.nn.functional.layer_norm
        elif node.op == "call_module":
            is_layernorm = self._is_layernorm_module(node, graph_module)

        self._logger.debug(f"Node {node} is layernorm: {is_layernorm}")

        if not is_layernorm:
            return False

        # Check if the input to layer norm comes from an add operation
        add_node = None
        for arg in node.args:
            if isinstance(arg, Node) and self._is_add_operation(arg, graph_module):
                add_node = arg
                break

        self._logger.debug(f"Found add node: {add_node}")

        if add_node is None:
            return False

        # Additional validation: ensure the pattern is safe to fuse
        is_safe = self._is_safe_to_fuse(add_node, node, graph_module)
        self._logger.debug(f"Pattern is safe to fuse: {is_safe}")

        if not is_safe:
            return False

        return True

    def _is_layernorm_module(self, node: Node, graph_module: GraphModule) -> bool:
        """Check if node calls a LayerNorm module."""
        if node.op != "call_module":
            return False

        if not hasattr(graph_module, node.target):
            return False

        module = getattr(graph_module, node.target)
        return isinstance(module, torch.nn.LayerNorm)

    def _is_add_operation(self, node: Node, graph_module: GraphModule) -> bool:
        """Check if node represents an add operation."""
        if node.op == "call_function" and node.target == torch.add:
            return True

        if node.op == "call_function" and node.target == torch.Tensor.__add__:
            return True

        # Check for built-in add function
        if node.op == "call_function" and str(node.target) == "<built-in function add>":
            return True

        # Check for add module (less common)
        if node.op == "call_module":
            if hasattr(graph_module, node.target):
                module = getattr(graph_module, node.target)
                return isinstance(module, (torch.nn.Identity,))  # Add modules are rare

        return False

    def _is_safe_to_fuse(self, add_node: Node, layernorm_node: Node, graph_module: GraphModule) -> bool:
        """
        Check if it's safe to fuse the Add + LayerNorm pattern.

        Args:
            add_node: The add operation node
            layernorm_node: The layer normalization node
            graph_module: The containing graph module

        Returns:
            True if safe to fuse
        """
        # Check if there are other users of the add operation
        # If add has multiple users, fusing might affect other operations
        add_users = list(add_node.users)
        if len(add_users) > 1:
            # Check if all users are layer norm operations that can be fused
            for user in add_users:
                if not self._is_layernorm_operation(user, graph_module):
                    return False

        return True

    def _is_layernorm_operation(self, node: Node, graph_module: GraphModule) -> bool:
        """Check if node is a layer normalization operation."""
        return (node.op == "call_function" and
                (node.target == torch.nn.functional.layer_norm or
                 self._is_layernorm_module(node, graph_module)))

    def initialize(self):
        """Initialize the pattern."""
        super().initialize()
        self._logger.info("AddLayerNormPattern initialized")

    def _execute_impl(self, context: OptimizationContext) -> Dict[str, Any]:
        """Execute the pattern optimization."""
        self._logger.info("AddLayerNormPattern executing")
        self._logger.info(f"strategy count: {self.strategy_count}")

        match_result = self.match(context.graph_module)
        result = self.execute_strategies(context.graph_module, match_result)
        self._logger.info("AddLayerNormPattern executed")
        self._logger.info(f"AddLayerNormPattern executed result: {result}")
        return {"pattern_executed": True}

    def execute(self, context: OptimizationContext) -> Dict[str, Any]:
        """Execute the pattern optimization."""
        return self._execute_impl(context)
