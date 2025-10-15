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
图优化的Pass系统。
"""

from .base import BasePass
from .common_subexpression_elimination import CommonSubexpressionEliminationPass
from .constant_folding import ConstantFoldingPass
from .dead_code_elimination import DeadCodeEliminationPass
from .manager import PassManager

__all__ = [
    "BasePass",
    "PassManager",
    "DeadCodeEliminationPass",
    "ConstantFoldingPass",
    "CommonSubexpressionEliminationPass",
]