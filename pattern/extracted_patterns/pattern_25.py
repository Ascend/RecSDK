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
    mask: torch.Tensor
    in1: torch.Tensor
    in2: torch.Tensor
    in3: torch.Tensor
    in4: torch.Tensor
    in5: torch.Tensor

    def to(self, device_type):
        return ModelInputs(
            mask=self.mask.to(device_type),
            in1=self.in1.to(device_type),
            in2=self.in2.to(device_type),
            in3=self.in3.to(device_type),
            in4=self.in4.to(device_type),
            in5=self.in5.to(device_type),
        )


class PatternModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_input: ModelInputs):
        mask_expanded = model_input.mask.unsqueeze(1).expand_as(model_input.in1)
        out0 = torch.where(mask_expanded, model_input.in1, model_input.in2)
        out1 = torch.where(mask_expanded, model_input.in1, model_input.in3)
        out2 = torch.where(mask_expanded, model_input.in1, model_input.in4)
        out3 = torch.where(mask_expanded, model_input.in1, model_input.in5)

        return out0, out1, out2, out3


def main():
    mask = torch.randint(0, 2, (256,)).bool()
    in1 = torch.randn(256, 4)
    in2 = torch.randn(256, 4)
    in3 = torch.randn(256, 4)
    in4 = torch.randn(256, 4)
    in5 = torch.randn(256, 4)

    inputs = ModelInputs(mask, in1, in2, in3, in4, in5)

    model = PatternModel()
    with torch.no_grad():
        out0_cpu, out1_cpu, out2_cpu, out3_cpu = model(inputs)
        out0_npu, out1_npu, out2_npu, out3_npu = model(inputs.to(device))

    default_logger.info("Output0 shape: %s", out0_npu.shape)
    default_logger.info("Output1 shape: %s", out1_npu.shape)
    default_logger.info("Output2 shape: %s", out2_npu.shape)
    default_logger.info("Output3 shape: %s", out3_npu.shape)

    judgement = (
        torch.allclose(out0_cpu, out0_npu.to("cpu"), rtol=1e-3, atol=1e-3)
        and torch.allclose(out1_cpu, out1_npu.to("cpu"), rtol=1e-3, atol=1e-3)
        and torch.allclose(out2_cpu, out2_npu.to("cpu"), rtol=1e-3, atol=1e-3)
        and torch.allclose(out3_cpu, out3_npu.to("cpu"), rtol=1e-3, atol=1e-3)
    )
    if not judgement:
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
