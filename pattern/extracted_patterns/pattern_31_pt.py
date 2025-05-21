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

    def forward(self, in0, in1):
        # Step 1: Slice (1, 64, 34) -> (1, 64, 33)
        sliced = in0[:, :, :33]

        # Step 2: Repeat (1, 64, 33) -> (192, 64, 33)
        repeated = sliced.expand(192, sliced.size(1), sliced.size(2))

        # Step 3: Strided Slice (192, 64, 33) -> (192, 64, 1)
        strided = repeated[:, :, :1]

        # Step 4: Mul (192, 64, 1) * (1, 64, 1) -> (192, 64, 1)
        mul_out = strided * in1

        # Step 5: Sum (192, 64, 1) -> (192, 1)
        sum_out = torch.sum(mul_out, dim=1)

        return sum_out


def main():
    in0 = torch.randn(1, 64, 34)
    in1 = torch.randn(1, 64, 1)

    # 初始化模型
    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    # 推理
    with torch.no_grad():
        output_cpu = model_cpu(in0, in1)
        output_npu = model_npu(in0.to(device), in1.to(device))

    # 结果验证
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
