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

    def forward(self, in0, in1, bias1, in2, bias2):
        tmp4 = in1[:, 0] + bias1[0]
        tmp5 = torch.sigmoid(tmp4)
        tmp7 = tmp5 * 2.0
        tmp8 = in0[:, 0] * tmp7

        tmp13 = in1[:, 1] + bias1[1]
        tmp14 = torch.sigmoid(tmp13)
        tmp15 = tmp14 * 2.0
        tmp16 = in0[:, 1] * tmp15

        tmp22 = in1[:, 2] + bias1[2]
        tmp23 = torch.sigmoid(tmp22)
        tmp24 = tmp23 * 2.0
        tmp25 = in0[:, 2] * tmp24

        out0 = tmp8 + tmp16 + tmp25

        tmp30 = in2[:, 0] + bias2[0]
        tmp31 = torch.sigmoid(tmp30)
        tmp32 = tmp31 * 2.0
        tmp33 = in0[:, 0] * tmp32

        tmp37 = in2[:, 1] + bias2[1]
        tmp38 = torch.sigmoid(tmp37)
        tmp39 = tmp38 * 2.0
        tmp40 = in0[:, 1] * tmp39

        tmp45 = in2[:, 2] + bias2[2]
        tmp46 = torch.sigmoid(tmp45)
        tmp47 = tmp46 * 2.0
        tmp48 = in0[:, 2] * tmp47

        out1 = tmp33 + tmp40 + tmp48

        return out0, out1


def main():
    in0 = torch.randn(256, 3)
    in1 = torch.randn(256, 3)
    in2 = torch.randn(256, 3)

    bias1 = torch.randn(3)
    bias2 = torch.randn(3)

    model = PatternModel()
    with torch.no_grad():
        output0_cpu, output1_cpu = model(in0, in1, bias1, in2, bias2)
        output0_npu, output1_npu = model(
            in0.to(device),
            in1.to(device),
            bias1.to(device),
            in2.to(device),
            bias2.to(device),
        )

    default_logger.info("Output0 shape: %s", output0_npu.shape)
    default_logger.info("Output1 shape: %s", output1_npu.shape)

    judgement_0 = torch.allclose(output0_cpu, output0_npu.to("cpu"), rtol=1e-3, atol=1e-3)
    judgement_1 = torch.allclose(output1_cpu, output1_npu.to("cpu"), rtol=1e-3, atol=1e-3)
    if not judgement_0 or not judgement_1:
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
