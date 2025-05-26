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
        self.layer_norm = torch.nn.LayerNorm([320])

    def forward(self, inputs):
        rs_output = torch.sum(inputs[0], dim=1)

        concat1_output = torch.cat([rs_output, inputs[1]], dim=1)

        concat2_param = [
            concat1_output,
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
        ]
        concat2_output = torch.cat(concat2_param, dim=1)

        mul_output = concat2_output * inputs[9]

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

    input_list = [
        [
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
        ]
    ]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
