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

    def forward(self, in0, in1, in2, in3):
        repeat1 = in0.expand(192, -1, -1)
        slice1 = repeat1[:, :, 1:]
        slice2 = repeat1[:, :, -1:]
        out_mul = slice2 * in1

        sum0 = slice1[:, :32, :].sum(dim=1)
        sum1 = slice1[:, 32:48, :].sum(dim=1)
        sum2 = slice1[:, 48:56, :].sum(dim=1)
        sum3 = slice1[:, 56:60, :].sum(dim=1)
        sum4 = slice1[:, 60:62, :].sum(dim=1)

        sum5 = slice1.sum(dim=1)
        out_cat = torch.cat([sum0, sum1, in2, in3, sum2, sum3, sum4, sum5], dim=1)

        return out_mul, out_cat


def main():
    torch.manual_seed(2025)
    in0 = torch.randn(1, 64, 33, dtype=torch.float16)
    in1 = torch.randn(1, 64, 1, dtype=torch.float16)
    in2 = torch.randn(192, 64, dtype=torch.float16)
    in3 = torch.randn(192, 64, dtype=torch.float16)

    input_list = [in0, in1, in2, in3]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
