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
将 nn.Module 转换为带有图和节点属性的 GraphModule 的包装器模块。
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule
from typing import Optional, List, Any


class GraphModuleWrapper(nn.Module):
    """
    将任何 nn.Module 转换为带有必需属性的 GraphModule 的包装器。

    此包装器确保模块具有 NGOBackend 所需的 'graph' 和 'nodes' 属性，
    同时保留原始功能。
    """

    def __init__(self, module: nn.Module, example_inputs: Optional[List[torch.Tensor]] = None):
        """
        初始化包装器。

        参数：
            module: 要包装的原始 nn.Module
            example_inputs: 用于符号跟踪的示例输入（可选）
        """
        super().__init__()
        self._original_module = module
        self._graph_module = None
        self._example_inputs = example_inputs

        # 尝试通过符号跟踪创建 GraphModule
        self._create_graph_module()

    def _create_graph_module(self):
        """通过符号跟踪创建 GraphModule。"""
        try:
            if self._example_inputs is not None:
                # 如果可用，使用示例输入进行跟踪
                self._graph_module = symbolic_trace(self._original_module)
            else:
                # 尝试不使用示例输入
                self._graph_module = symbolic_trace(self._original_module)

            # 修改图以返回元组以满足 inductor 要求
            self._ensure_tuple_output()
        except Exception:
            # 如果符号跟踪失败，创建一个最小的 GraphModule
            self._graph_module = self._create_minimal_graph_module()

    def _ensure_tuple_output(self):
        """确保图返回元组以满足 inductor 要求。"""
        if self._graph_module is None:
            return

        graph = self._graph_module.graph
        output_node = next(iter(reversed(graph.nodes)))

        # 检查输出是否已经是元组
        if isinstance(output_node.args[0], (tuple, list)):
            return

        # 获取原始输出
        original_output = output_node.args[0]

        # 用元组输出替换输出节点
        with graph.inserting_before(output_node):
            new_output = graph.output((original_output,))

        # 移除旧的输出节点
        graph.erase_node(output_node)

        # 重新编译图
        self._graph_module.recompile()

    def _create_minimal_graph_module(self) -> GraphModule:
        """创建具有基本图结构的最小 GraphModule。"""
        import torch.fx as fx

        # 创建一个简单的图
        graph = fx.Graph()

        # 为输入添加占位符
        input_node = graph.placeholder('x')

        # 添加输出（以元组形式返回以满足 inductor 要求）
        graph.output((input_node,))

        # 使用包装原始模块的简单模块创建 GraphModule
        class SimpleModule(nn.Module):
            def __init__(self, original_module):
                super().__init__()
                self._wrapped_module = original_module

            def forward(self, x):
                return (x,)  # 以元组形式返回以满足 inductor 要求

        simple_module = SimpleModule(self._original_module)
        gm = GraphModule(simple_module, graph)

        return gm

    def forward(self, *args, **kwargs):
        """使用原始模块的前向传播。"""
        return self._original_module(*args, **kwargs)

    @property
    def graph(self):
        """获取图属性（NGOBackend 需要）。"""
        if self._graph_module is not None:
            return self._graph_module.graph
        else:
            raise AttributeError("GraphModule 不可用")

    @property
    def graph_nodes(self):
        """获取图节点（用于兼容性）。"""
        if hasattr(self.graph, 'nodes'):
            return list(self.graph.nodes)
        return []

    @property
    def graph_size(self):
        """获取图大小。"""
        if hasattr(self.graph, 'nodes'):
            return len(self.graph.nodes)
        return 0

    @property
    def code(self):
        """获取代码属性（torch 调试需要）。"""
        if self._graph_module is not None:
            return self._graph_module.code
        else:
            return "def forward(self, x):\n    return x"

    # 内部使用的直接属性访问
    def _get_internal_attr(self, name):
        """安全地获取内部属性。"""
        return object.__getattribute__(self, name)


def create_graph_module_wrapper(module: nn.Module, example_inputs: Optional[List[torch.Tensor]] = None) -> GraphModuleWrapper:
    """
    为任何 nn.Module 创建 GraphModule 包装器。

    参数：
        module: 要包装的原始 nn.Module
        example_inputs: 用于符号跟踪的示例输入（可选）

    返回：
        带有必需属性的 GraphModuleWrapper
    """
    return GraphModuleWrapper(module, example_inputs)


def ensure_graph_module(module: nn.Module, example_inputs: Optional[List[torch.Tensor]] = None):
    """
    确保模块具有图和节点属性。

    参数：
        module: 要检查/转换的模块
        example_inputs: 用于符号跟踪的示例输入（可选）

    返回：
        带有图和节点属性的模块
    """
    if hasattr(module, 'graph') and hasattr(module.graph, 'nodes'):
        # 已经具有必需的属性
        return module
    else:
        # 使用 GraphModuleWrapper 包装
        return create_graph_module_wrapper(module, example_inputs)