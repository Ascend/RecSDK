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
        # Step 1: (128, 50, 33) -> (128, 50, 32)
        slice_output = inputs[0][:, :, :32]

        concat_inputs = [
            slice_output,
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
        ]

        # 沿第3维拼接
        concat_output = torch.cat(concat_inputs, dim=2)

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

    input_list = [
        [in0, cat_in0, cat_in1, cat_in2, cat_in3, cat_in4, cat_in5, cat_in6, cat_in7]
    ]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
