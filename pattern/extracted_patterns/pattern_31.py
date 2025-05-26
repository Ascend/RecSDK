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

    def forward(self, in0, in1):
        # Step 1: Slice (1, 64, 34) -> (1, 64, 33)
        sliced = in0[:, :, :33]

        # Step 2: Repeat (1, 64, 33) -> (192, 64, 33)
        repeated = sliced.expand(192, sliced.size(1), sliced.size(2))

        # Step 3: Strided Slice (192, 64, 33) -> (192, 64, 1)
        strided = repeated[:, :, :1]

        # Step 4: Mul (192, 64, 1) * (1, 64, 1) -> (192, 64, 1)
        mul_out = strided * in1

        # Step 5: Sum (192, 64, 1) -> (192, 1)
        sum_out = torch.sum(mul_out, dim=1)

        return sum_out


def main():
    in0 = torch.randn(1, 64, 34)
    in1 = torch.randn(1, 64, 1)

    input_list = [in0, in1]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
