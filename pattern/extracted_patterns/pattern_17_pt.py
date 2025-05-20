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

    def forward(self, input0, input1, input2, input3):
        """
        PyTorch 实现的三输出计算逻辑：
        1. 输入张量维度扩展
        2. 执行 add/mul/sub 三组并行计算
        3. 返回三个结果张量
        """
        tmp1 = input1.unsqueeze(1)
        tmp3 = input2.unsqueeze(0).unsqueeze(0)
        tmp7 = input3.unsqueeze(1)

        tmp5 = tmp1 + tmp3
        tmp6 = input0 + tmp5
        tmp9 = tmp7 - tmp6
        tmp10 = tmp7 * tmp6

        return tmp6, tmp9, tmp10


def main():
    # 参数设置
    p0_numel = 256
    y1_numel = 64
    x2_numel = 32

    # 生成可解释的测试数据
    input0 = torch.arange(p0_numel * y1_numel * x2_numel, dtype=torch.float32)
    input0_cpu = input0.reshape(p0_numel, y1_numel, x2_numel)
    input1_cpu = torch.randn(p0_numel, x2_numel).float()
    input2_cpu = torch.randn(x2_numel).float()
    input3_cpu = torch.randn(p0_numel, x2_numel).float()

    input0_npu = input0_cpu
    input1_npu = input1_cpu
    input2_npu = input2_cpu
    input3_npu = input3_cpu

    model = PatternModel()
    # 推理
    with torch.no_grad():
        output0_cpu, output1_cpu, output2_cpu = model(
            input0_cpu, input1_cpu, input2_cpu, input3_cpu
        )
        output0_npu, output1_npu, output2_npu = model(
            input0_npu, input1_npu, input2_npu, input3_npu
        )

    # 结果验证
    default_logger.info("Output0 shape: %s", output0_npu.shape)
    default_logger.info("Output1 shape: %s", output1_npu.shape)
    default_logger.info("Output2 shape: %s", output2_npu.shape)

    judgement_0 = torch.allclose(
        output0_cpu, output0_npu.to("cpu"), rtol=1e-3, atol=1e-3
    )
    judgement_1 = torch.allclose(
        output1_cpu, output1_npu.to("cpu"), rtol=1e-3, atol=1e-3
    )
    judgement_2 = torch.allclose(
        output2_cpu, output2_npu.to("cpu"), rtol=1e-3, atol=1e-3
    )
    if not judgement_0 or not judgement_1 or not judgement_2:
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
