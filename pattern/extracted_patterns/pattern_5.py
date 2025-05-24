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
    def forward(self, input0, input1):
        # Step 1: 对 input0 进行 transpose 和 reshape
        input0_trans = input0.permute(1, 0, 2)  # shape: (1523, 128, 1)
        input0_reshape = input0_trans.reshape(1523, 128)  # shape: (1523, 128)

        # Step 2: 矩阵乘法
        matmul_output = torch.matmul(input1, input0_reshape)  # shape: (32, 128)

        # Step 3: 添加维度并激活
        matmul_output = matmul_output.unsqueeze(-1)  # shape: (32, 128, 1)
        activated_output = torch.relu(matmul_output)  # 默认使用 ReLU

        # Step 4: 输出的形状是 (128, 32, 1)
        activated_output = activated_output.permute(1, 0, 2)  # shape: (128, 32, 1)

        return activated_output


def main():
    # 创建输入张量
    input0 = torch.randn(128, 1523, 1)
    input1 = torch.randn(32, 1523)
    
    input_list = [input0, input1]
    perform_test(PatternModel(), input_list)

if __name__ == "__main__":
    main()