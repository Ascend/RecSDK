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

from utils.logger import default_logger


class PatternModel(torch.nn.Module):
    def forward(self, input0, input1):
        # Step 1: 执行逐元素减法（sub）
        sub_output = input0 - input1  # shape: (128, 128, 144)

        # Step 2: 执行逐元素乘法（mul）
        mul_output = input0 * input1  # shape: (128, 128, 144)

        # Step 3: 在 dim=2 上进行 concat
        output = torch.cat([input0, input1, sub_output, mul_output], dim=2)

        return output


def main():
    # 创建两个形状为 (128, 128, 144) 的输入张量
    input0 = torch.randn(128, 128, 144)
    input1 = torch.randn(128, 128, 144)

    # 调用处理函数
    model = PatternModel()

    output_tensor = model(input0, input1)

    # 打印输出张量的形状，确保它是 (128, 128, 576)
    default_logger.info("Output shape: %s", output_tensor.shape)

if __name__ == "__main__":
    main()