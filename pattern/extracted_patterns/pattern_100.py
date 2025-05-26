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
        tile0 = inputs[0].tile((256, 1))
        mul0 = tile0 * inputs[1]
        mul1 = mul0 * tile0
        addn_output = mul1 + inputs[2]
        return addn_output


def main():
    torch.manual_seed(2025)
    in0 = torch.randn(1, 64, dtype=torch.float)
    in1 = torch.randn(256, 64, dtype=torch.float)
    in2 = torch.randn(256, 64, dtype=torch.float)

    input_list = [[in0, in1, in2]]

    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
