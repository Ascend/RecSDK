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

    def forward(self, input0, input1, input2, input3, input4, eps):
        # 加法与乘法
        combined = (input0 + input1 + input2) * 0.1
        # 转换为FP32计算
        combined_fp32 = combined.float()
        # 计算均值和方差
        mean = combined_fp32.mean(dim=-1, keepdim=True)
        var = combined_fp32.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        normalized = (combined_fp32 - mean) * torch.rsqrt(var + eps)

        gamma = input3
        beta = input4
        output_fp32 = normalized * gamma + beta
        return output_fp32


def main():
    # 参数配置
    batch_size = 256
    feature_dim = 48
    eps = 1e-6
    torch.manual_seed(2025)

    input0_cpu = torch.randn(batch_size, feature_dim, dtype=torch.float16)
    input1_cpu = torch.randn(batch_size, feature_dim, dtype=torch.float16)
    input2_cpu = torch.randn(batch_size, feature_dim, dtype=torch.float16)
    input3_cpu = torch.randn(feature_dim, dtype=torch.float32)
    input4_cpu = torch.randn(feature_dim, dtype=torch.float32)
    input0_npu = input0_cpu.to(device)
    input1_npu = input1_cpu.to(device)
    input2_npu = input2_cpu.to(device)
    input3_npu = input3_cpu.to(device)
    input4_npu = input4_cpu.to(device)

    # 初始化模型
    model_cpu = PatternModel()
    model_npu = PatternModel().to(device)

    # 推理
    with torch.no_grad():
        output_cpu = model_cpu(
            input0_cpu, input1_cpu, input2_cpu, input3_cpu, input4_cpu, eps
        )
        output_npu = model_npu(
            input0_npu, input1_npu, input2_npu, input3_npu, input4_npu, eps
        )

    # 结果验证
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
