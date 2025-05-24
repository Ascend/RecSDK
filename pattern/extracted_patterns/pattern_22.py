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

    def forward(self, input0, input1, in0_start, in1_start):
        tmp0 = input0[:, :, in0_start:]
        tmp1 = input1[:, :, in1_start:]
        return torch.sum(tmp0 * tmp1, dim=1)


def main():
    # 设置参数
    y0_numel = 256
    r0_numel = 64
    x0_numel = 130
    y1_numel = 1
    r1_numel = 64
    x1_numel = 33
    in0_start = 129
    in1_start = 32

    input0_cpu = torch.randn(y0_numel, r0_numel, x0_numel)
    input1_cpu = torch.randn(y1_numel, r1_numel, x1_numel)
    input0_npu = input0_cpu.to(device)
    input1_npu = input1_cpu.to(device)

    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    with torch.no_grad():
        output_cpu = model_cpu(input0_cpu, input1_cpu, in0_start, in1_start)
        output_npu = model_npu(
            input0_npu.to(device), input1_npu.to(device), in0_start, in1_start
        )

    default_logger.info("Output shape: %s", output_npu.shape)
    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
