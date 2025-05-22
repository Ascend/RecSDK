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

    def forward(self, input0, x1_numel):
        sliced = input0[:, :, 1:1 + x1_numel]
        return sliced.sum(dim=1)


def main():
    y_dim = 256
    r_dim = 32
    x_dim = 130
    output_dim = 128

    input0_cpu = torch.randn(y_dim, r_dim, x_dim)
    input0_npu = input0_cpu.to(device)

    model = PatternModel()
    with torch.no_grad():
        output0_cpu = model(input0_cpu, output_dim)
        output0_npu = model(input0_npu.to(device), output_dim.to(device))

    default_logger.info("Output0 shape: %s", output0_npu.shape)

    judgement_0 = torch.allclose(
        output0_cpu, output0_npu.to("cpu"), rtol=1e-3, atol=1e-3
    )
    if not judgement_0:
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
