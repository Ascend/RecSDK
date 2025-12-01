# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import collections
from typing import Any

import torch
from torch.fx import Interpreter


def to_clean_cpu_tensor(data: Any) -> Any:
    """
    递归地将数据结构中的所有PyTorch张量移动到CPU并与计算图分离。
    支持处理张量、列表、元组和字典。
    """
    if isinstance(data, torch.Tensor):
        # 1. 确保张量在CPU上并与计算图分离
        return data.detach().cpu()
    elif isinstance(data, (list, tuple)):
        # 2. 如果是列表或元组，递归处理其每个元素
        return type(data)(to_clean_cpu_tensor(item) for item in data)
    elif isinstance(data, dict):
        # 3. 如果是字典，递归处理其每个值
        return {key: to_clean_cpu_tensor(value) for key, value in data.items()}
    else:
        # 4. 如果是其他类型，原样返回
        return data


class CapturingInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.captured_data = collections.OrderedDict()

    def run_node(self, n: torch.fx.Node) -> Any:
        node_name = n.name
        self.captured_data[node_name] = {"type": n.op}

        args, kwargs = self.fetch_args_kwargs_from_env(n)
        self.captured_data[node_name]["input"] = to_clean_cpu_tensor(args)

        # 执行原始的节点计算
        output = super().run_node(n)
        self.captured_data[node_name]["output"] = to_clean_cpu_tensor(output)

        if n.op == "call_module":
            submodule = self.fetch_attr(n.target)
            # --- 使用【新的】辅助函数统一处理参数 ---
            self.captured_data[node_name]["parameter"] = to_clean_cpu_tensor(
                dict(submodule.state_dict())
            )

        return output