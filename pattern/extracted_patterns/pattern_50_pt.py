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

    def forward(
        self,
        in0,
        cat_in0,
        cat_in1,
        cat_in2,
        cat_in3,
        cat_in4,
        cat_in5,
        cat_in6,
        cat_in7,
    ):
        # Step 1: (128, 50, 33) -> (128, 50, 32)
        slice_output = in0[:, :, :32]

        concat_inputs = [
            slice_output,
            cat_in0,
            cat_in1,
            cat_in2,
            cat_in3,
            cat_in4,
            cat_in5,
            cat_in6,
            cat_in7,
        ]

        # 沿第3维拼接
        concat_output = torch.cat(concat_inputs, dim=2)  # (128, 50, 432)

        return concat_output


def main():
    in0 = torch.randn(128, 50, 33)
    cat_in0 = torch.randn(128, 50, 128)
    cat_in1 = torch.randn(128, 50, 32)
    cat_in2 = torch.randn(128, 50, 48)
    cat_in3 = torch.randn(128, 50, 48)
    cat_in4 = torch.randn(128, 50, 48)
    cat_in5 = torch.randn(128, 50, 48)
    cat_in6 = torch.randn(128, 50, 48)
    cat_in7 = torch.randn(128, 50, 48)

    # 初始化模型
    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    # 推理
    with torch.no_grad():
        output_cpu = model_cpu(
            in0, cat_in0, cat_in1, cat_in2, cat_in3, cat_in4, cat_in5, cat_in6, cat_in7
        )
        output_npu = model_npu(
            in0.to(device),
            cat_in0.to(device),
            cat_in1.to(device),
            cat_in2.to(device),
            cat_in3.to(device),
            cat_in4.to(device),
            cat_in5.to(device),
            cat_in6.to(device),
            cat_in7.to(device),
        )

    # 结果验证
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
