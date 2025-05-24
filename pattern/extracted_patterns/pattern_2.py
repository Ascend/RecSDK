#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import torch

from pattern.util import perform_test


class PatternModel(torch.nn.Module):
    def forward(self, input_tensor):
        # Step 1: 使用 reduce_sum 对第二个维度进行求和，结果形状为 (128, 1, 16)
        reduced_tensor = torch.sum(input_tensor, dim=1, keepdim=True)  # shape: (128, 1, 16)

        # Step 2: 使用 repeat 扩展到 (128, 120, 16)
        expanded_tensor = reduced_tensor.repeat(1, 120, 1)  # shape: (128, 120, 16)

        return expanded_tensor
    

def main():
    # 创建一个形状为 (128, 5, 16) 的输入张量
    input_tensor = torch.randn(128, 5, 16)

    input_list = [input_tensor]
    
    perform_test(PatternModel(), input_list)
    
if __name__ == "__main__":
    main()