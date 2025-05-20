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

import numpy as np
import torch
import torch_npu

from utils.logger import default_logger

device = torch.device("npu")


class PatternModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices):
        """
        x: 输入张量，形状为 [batch_size, row, col]
        输出形状为 [batch_size, output_num, col]
        """
        outputs = []
        # 遍历每个切片
        for i in range(len(indices)):
            start = indices[i, 0]
            end = indices[i, 1]
            # 提取切片并求和
            sliced = x[:, start:end, :]
            summed = sliced.sum(dim=1)
            outputs.append(summed)
        # 沿新维度拼接结果
        return torch.stack(outputs, dim=1)


def main():
    batch_size = 256
    row = 10
    col = 5
    output_num = 3

    indices = torch.tensor([[1, 3], [4, 7], [8, 9]])
    torch.manual_seed(2025)
    input_tensor_cpu = torch.randint(0, 10, (batch_size, row, col)).float()
    input_tensor_npu = input_tensor_cpu.to(device)

    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    with torch.no_grad():
        output_cpu = model_cpu(input_tensor_cpu, indices)
        output_npu = model_npu(input_tensor_npu, indices)

    default_logger.info("Input shape: %s", input_tensor_npu.shape)
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
