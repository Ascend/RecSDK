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

    def forward(self, input0, input1):
        tmp0 = input0[:, :, 129]
        return torch.sum(tmp0 * input1, dim=1).unsqueeze(1)


def main():
    # 设置参数
    y0_numel = 256
    x0_numel = 130
    r1_numel = 64

    input0_cpu = torch.randn(y0_numel, r1_numel, x0_numel)
    input1_cpu = torch.randn(r1_numel)
    input0_npu = input0_cpu.to(device)
    input1_npu = input1_cpu.to(device)

    # 初始化模型
    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    # 推理
    with torch.no_grad():
        output_cpu = model_cpu(input0_cpu, input1_cpu)
        output_npu = model_npu(input0_npu, input1_npu)

    # 结果验证
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
