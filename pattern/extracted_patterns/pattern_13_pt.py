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
        in0: torch.Tensor,
        in1: torch.Tensor,
        in2: torch.Tensor,
        in3: torch.Tensor,
        in4: torch.Tensor,
        in5: torch.Tensor,
    ):

        tmp0 = in0.to(torch.float32)
        tmp1 = in1.expand_as(tmp0[..., 0]).unsqueeze(1).expand_as(tmp0)
        tmp2 = tmp0 + tmp1

        tmp3 = in2.to(torch.float32)
        tmp5 = tmp2 / tmp3
        tmp6 = torch.sigmoid(tmp5)

        tmp7 = in3.unsqueeze(2).expand(-1, -1, tmp0.shape[2]).to(torch.float32)
        sign = torch.sign(tmp7)

        tmp15 = tmp6 * sign
        tmp16 = in4.to(torch.float32)
        tmp17 = tmp15 * tmp16
        output0 = tmp17.sum(dim=1)

        tmp22 = in5.expand_as(tmp16).to(torch.float32)
        tmp23 = tmp22 * tmp16
        output1 = tmp23.sum(dim=1)

        return output0, output1


def main():
    # 配置参数
    y_dim = 256
    x_dim = 32
    r_dim = 32

    # 生成测试数据
    torch.manual_seed(2025)
    in0 = torch.randn(y_dim, r_dim, x_dim) * 2
    in1 = torch.randn(x_dim)
    in2 = torch.tensor([0.5])
    in3 = torch.randn(y_dim, r_dim)
    in4 = torch.randn(y_dim, r_dim, x_dim)
    in5 = torch.randn(r_dim, x_dim)

    in0_npu = in0.to(device)
    in1_npu = in1.to(device)
    in2_npu = in2.to(device)
    in3_npu = in3.to(device)
    in4_npu = in4.to(device)
    in5_npu = in5.to(device)

    model_cpu = PatternModel().to("cpu")
    model_npu = PatternModel().to(device)

    with torch.no_grad():
        out0_cpu, out1_cpu = model_cpu(in0, in1, in2, in3, in4, in5)
        out0_npu, out1_npu = model_npu(
            in0_npu, in1_npu, in2_npu, in3_npu, in4_npu, in5_npu
        )

    # 结果验证
    default_logger.info("Output0 shape: %s", out0_npu.shape)
    default_logger.info("Output1 shape: %s", out1_npu.shape)

    if not torch.allclose(
        out0_cpu, out0_npu.to("cpu"), rtol=1e-3, atol=1e-3
    ) or not torch.allclose(out1_cpu, out1_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
