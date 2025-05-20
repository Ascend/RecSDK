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
import torch.nn as nn
import torch_npu

from utils.logger import default_logger

device = torch.device("npu")


class PatternModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, start_indices, slice_len):
        """
        x: 输入张量形状 [batch, row, col]
        输出形状 [batch, num_slices, row, slice_len]
        """
        batch_size = x.size(0)
        outputs = []

        for idx in start_indices:
            start_col = idx.item()
            end_col = start_col + slice_len

            sliced = x[..., start_col:end_col]

            outputs.append(sliced.unsqueeze(1))

        # 拼接所有切片
        return torch.cat(outputs, dim=1)


def main():
    batch_size = 256
    input_row = 1000
    input_col = 2048
    slice_num = 4
    slice_len = 128

    torch.manual_seed(2025)
    start_indices = torch.tensor([256, 512, 768, 1024])

    input_tensor_cpu = torch.randn(batch_size, input_row, input_col)
    input_tensor_npu = input_tensor_cpu.to(device)

    model = PatternModel()

    # 执行推理
    with torch.no_grad():
        output_cpu = model(input_tensor_cpu, start_indices, slice_len)
        output_npu = model(input_tensor_npu, start_indices.to(device), slice_len)

    # 结果验证
    default_logger.info("Input shape: %s", input_tensor_npu.shape)
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
