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

    def forward(self, in0, in1, in2, in3, in4):
        # Step 1: ReduceMeanD (128, 512) -> (128, 1)
        mean = in0.mean(dim=1, keepdim=True)

        # Step 2: Add01 (128, 1)
        add01 = mean + in1

        # Step 3: Rsqrt (128, 1) -> (128, 1)
        rsqrt = torch.rsqrt(add01)

        # Step 4: mul01 (128, 1) * (512,) -> (128, 512)
        mul01 = rsqrt * in2.unsqueeze(0)

        # Step 5: mul02 (128, 1) * (128, 512) -> (128, 512)
        mul02 = mul01 * mean

        # Step 6: sub (512,) - (128, 512) -> (128, 512)
        sub = in3.unsqueeze(0) - mul02

        # Step 7: mul03 (128, 512) * (128, 512) -> (128, 512)
        mul03 = mul01 * in4

        # Step 8: add02 (128, 512) + (128, 512) -> (128, 512)
        add02 = sub + mul03

        return add02


def main():
    torch.manual_seed(2025)
    in0 = torch.abs(torch.randn(128, 512, dtype=torch.float16))
    in1 = torch.abs(torch.randn(128, 1, dtype=torch.float16))
    in2 = torch.abs(torch.randn(512, dtype=torch.float16))
    in3 = torch.abs(torch.randn(512, dtype=torch.float16))
    in4 = torch.abs(torch.randn(128, 512, dtype=torch.float16))

    input_list = [in0, in1, in2, in3, in4]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
