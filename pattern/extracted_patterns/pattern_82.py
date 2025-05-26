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

    def forward(self, inputs):
        tmp0 = inputs[0]
        square = torch.square(tmp0)
        out_0 = square.sum(dim=0)
        mul1 = inputs[1] * tmp0
        out_1 = mul1.sum(dim=0)
        mul2 = inputs[2] * tmp0
        mul3 = mul2 + inputs[4]
        out_2 = mul3.sum(dim=0)
        mul4 = inputs[3] * tmp0
        mul5 = mul4 * inputs[5]
        out_3 = mul5 + inputs[6]

        return out_0, out_1, out_2, out_3


def main():
    torch.manual_seed(2025)
    in0 = torch.randn(16, 1, dtype=torch.float)
    in1 = torch.randn(1, dtype=torch.float)
    in2 = torch.randn(1, dtype=torch.float)
    in3 = torch.randn(1, dtype=torch.float)
    in4 = torch.randn(16, 1, dtype=torch.float)
    in5 = torch.randn(16, 1, dtype=torch.float)
    in6 = torch.randn(16, 1, dtype=torch.float)

    input_list = [[in0, in1, in2, in3, in4, in5, in6]]

    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
