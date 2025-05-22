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

    def forward(self, in0, in1, in2, in3, in4):
        # Step 1: ReduceMean (128, 1024) -> (128, 1)
        mean = in0.mean(dim=1, keepdim=True)

        # Step 2: SquaredDifference (128, 1024) 与 (128, 1) -> (128, 1024)
        squared_diff = (in0 - mean) ** 2

        # Step 3: ReduceMeanD (128, 1024) -> (128, 1)
        mean01 = in0.mean(dim=1, keepdim=True)

        # Step 4: Add01 (128, 1)
        add01 = mean01 + in1

        # Step 5: Rsqrt (128, 1) -> (128, 1)
        rsqrt = torch.rsqrt(add01)

        # Step 6: mul01 (128, 1) * (1024,) -> (128, 1024)
        mul01 = rsqrt * in2.unsqueeze(0)

        # Step 7: mul02 (128, 1) * (128, 1024) -> (128, 1024)
        mul02 = mul01 * mean01

        # Step 8: sub (1024,) - (128, 1024) -> (128, 1024)
        sub = in3.unsqueeze(0) - mul02

        # Step 9: mul03 (128, 1024) * (128, 1024) -> (128, 1024)
        mul03 = mul01 * in4

        # Step 10: add02 (128, 1024) + (128, 1024) -> (128, 1024)
        add02 = sub + mul03

        out1, out2 = torch.split(add02, 512, dim=1)

        return out1, out2


def main():
    torch.manual_seed(2025)
    in0 = torch.abs(torch.randn(128, 1024, dtype=torch.float16))
    in1 = torch.abs(torch.randn(128, 1, dtype=torch.float16))
    in2 = torch.abs(torch.randn(1024, dtype=torch.float16))
    in3 = torch.abs(torch.randn(1024, dtype=torch.float16))
    in4 = torch.abs(torch.randn(128, 1024, dtype=torch.float16))
    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    with torch.no_grad():
        output0_cpu, output1_cpu = model_cpu(in0, in1, in2, in3, in4)
        output0_npu, output1_npu = model_cpu(
            in0.to(device),
            in1.to(device),
            in2.to(device),
            in3.to(device),
            in4.to(device),
        )

    default_logger.info("Output0 shape: %s", output0_npu.shape)
    default_logger.info("Output1 shape: %s", output1_npu.shape)

    judgement_0 = torch.allclose(
        output0_cpu, output0_npu.to("cpu"), rtol=1e-3, atol=1e-3
    )
    judgement_1 = torch.allclose(
        output1_cpu, output1_npu.to("cpu"), rtol=1e-3, atol=1e-3
    )
    if not judgement_0 or not judgement_1:
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
