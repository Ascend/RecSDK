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

from pattern.util import perform_test


class PatternModel(torch.nn.Module):
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
    p0_numel = 256
    y1_numel = 64
    x2_numel = 32

    input0 = torch.arange(p0_numel * y1_numel * x2_numel, dtype=torch.float32)
    in0 = input0.reshape(p0_numel, y1_numel, x2_numel)
    in1 = torch.randn(p0_numel, x2_numel).float()
    in2 = torch.randn(x2_numel).float()
    in3 = torch.randn(p0_numel, x2_numel).float()

    input_list = [in0, in1, in2, in3]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
