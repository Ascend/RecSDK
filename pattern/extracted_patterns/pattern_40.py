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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_npu
from utils.logger import default_logger

device = torch.device("npu")


@dataclass
class ModelInputs:
    in0: torch.tensor
    in1: torch.tensor
    in2: torch.tensor
    in3: torch.tensor
    in4: torch.tensor
    in5: torch.tensor

    def to(self, device_type):
        return ModelInputs(
            in0=self.in0.to(device),
            in1=self.in1.to(device),
            in2=self.in2.to(device),
            in3=self.in3.to(device),
            in4=self.in4.to(device),
            in5=self.in5.to(device),
        )


class PatternModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm([640])

    def forward(self, inputs: ModelInputs):
        concat01 = torch.cat(
            [inputs.in0, inputs.in1, inputs.in2, inputs.in3, inputs.in4], dim=1
        )
        mul0 = concat01 * inputs.in5
        ln_output = self.layer_norm(mul0)
        return ln_output


def main():
    torch.manual_seed(2025)
    in0 = torch.randn(192, 128, dtype=torch.float16)
    in1 = torch.randn(192, 128, dtype=torch.float16)
    in2 = torch.randn(192, 128, dtype=torch.float16)
    in3 = torch.randn(192, 128, dtype=torch.float16)
    in4 = torch.randn(192, 128, dtype=torch.float16)
    in5 = torch.randn(640, dtype=torch.float16)

    inputs = ModelInputs(in0, in1, in2, in3, in4, in5)

    model_cpu = PatternModel()

    with torch.no_grad():
        output_cpu = model_cpu(inputs)
        output_npu = model_npu(inputs.to(device))

    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
