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
        mul0 = inputs[0] * inputs[1]
        mul1 = mul0 * inputs[2]
        sum0 = mul1.sum(dim=1)
        sqrt = torch.sqrt(sum0).unsqueeze(0)

        select = torch.where(inputs[4], inputs[5], sqrt)
        tile = select.tile((48, 1))

        mul2 = inputs[2] * mul1
        mul3 = tile * mul2
        mul4 = mul0 * inputs[3]
        out = mul3 + mul4
        return out


def main():
    torch.manual_seed(2025)
    in0 = torch.abs(torch.randn(48, 48, dtype=torch.float))
    in1 = torch.abs(torch.randn(48, dtype=torch.float))
    in2 = torch.abs(torch.randn(48, 48, dtype=torch.float))
    in3 = torch.randn(48, 48, dtype=torch.float)
    selcet_in = torch.randint(0, 2, (1, 48), dtype=torch.bool)
    in4 = torch.randn(1, 48, dtype=torch.float)

    input_list = [[in0, in1, in2, in3, selcet_in, in4]]

    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
