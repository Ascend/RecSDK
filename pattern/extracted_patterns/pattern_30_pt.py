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
        self.layer_norm = nn.LayerNorm([320])

    def forward(
        self,
        in0,
        concat_in0,
        concat_in1,
        concat_in2,
        concat_in3,
        concat_in4,
        concat_in5,
        concat_in6,
        concat_in7,
        mul_in0,
    ):
        # ReduceSumD 节点
        rs_output = torch.sum(in0, dim=1)  # (192, 32)

        # Concat1 节点
        concat1_output = torch.cat([rs_output, concat_in0], dim=1)  # (192, 64)

        # Concat2 节点
        concat2_inputs = [
            concat1_output,
            concat_in1,
            concat_in2,
            concat_in3,
            concat_in4,
            concat_in5,
            concat_in6,
            concat_in7,
        ]
        concat2_output = torch.cat(concat2_inputs, dim=1)  # (192, 320)

        # Mul 节点
        mul_output = concat2_output * mul_in0

        # LayerNorm 节点
        ln_output = self.layer_norm(mul_output)

        return ln_output


def main():
    in0 = torch.randn(192, 64, 32)
    concat_in0 = torch.randn(192, 64)
    concat_in1 = torch.randn(192, 32)
    concat_in2 = torch.randn(192, 32)
    concat_in3 = torch.randn(192, 32)
    concat_in4 = torch.randn(192, 32)
    concat_in5 = torch.randn(192, 32)
    concat_in6 = torch.randn(192, 32)
    concat_in7 = torch.randn(192, 32)

    mul_in0 = torch.randn(1)

    # 初始化模型
    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    # 推理
    with torch.no_grad():
        output_cpu = model_cpu(
            in0,
            concat_in0,
            concat_in1,
            concat_in2,
            concat_in3,
            concat_in4,
            concat_in5,
            concat_in6,
            concat_in7,
            mul_in0,
        )
        output_npu = model_npu(
            in0.to(device),
            concat_in0.to(device),
            concat_in1.to(device),
            concat_in2.to(device),
            concat_in3.to(device),
            concat_in4.to(device),
            concat_in5.to(device),
            concat_in6.to(device),
            concat_in7.to(device),
            mul_in0.to(device),
        )

    # 结果验证
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
