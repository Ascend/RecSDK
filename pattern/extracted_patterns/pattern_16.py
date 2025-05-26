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

    def forward(self, input0, input1):
        tmp0 = input0[:, :, 129]
        return torch.sum(tmp0 * input1, dim=1).unsqueeze(1)


def main():
    y0_numel = 256
    x0_numel = 130
    r1_numel = 64

    in0 = torch.randn(y0_numel, r1_numel, x0_numel)
    in1 = torch.randn(r1_numel)

    input_list = [in0, in1]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
