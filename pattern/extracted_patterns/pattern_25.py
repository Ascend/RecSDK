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
        mask_expanded = inputs[0].unsqueeze(1).expand_as(inputs[1])
        out0 = torch.where(mask_expanded, inputs[1], inputs[2])
        out1 = torch.where(mask_expanded, inputs[1], inputs[3])
        out2 = torch.where(mask_expanded, inputs[1], inputs[4])
        out3 = torch.where(mask_expanded, inputs[1], inputs[5])

        return out0, out1, out2, out3


def main():
    mask = torch.randint(0, 2, (256,)).bool()
    in1 = torch.randn(256, 4)
    in2 = torch.randn(256, 4)
    in3 = torch.randn(256, 4)
    in4 = torch.randn(256, 4)
    in5 = torch.randn(256, 4)

    input_list = [[mask, in1, in2, in3, in4, in5]]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
