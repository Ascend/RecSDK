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

import importlib


def _try_import(module_name):
    try:
        status = importlib.import_module(module_name)
        if status and module_name == "torch_npu":
            try:
                import torch_npu._inductor
            except ImportError:
                raise ImportError("torch_npu._inductor not found")
        return status
    except ImportError:
        return None


# 在这里集中管理所有可能的依赖
torch_npu = _try_import("torch_npu")


# 提供全局判断函数
def has_torch_npu():
    return torch_npu is not None
