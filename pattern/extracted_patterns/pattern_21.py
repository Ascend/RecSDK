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

    def forward(self, in0, scale, bias):
        tmp0 = in0 * 0.1
        # LayerNorm
        mean = tmp0.mean(dim=1, keepdim=True)
        var = tmp0.var(dim=1, keepdim=True, unbiased=False)
        inv_std = 1.0 / torch.sqrt(var + 1e-6)
        x_normalized = (tmp0 - mean) * inv_std
        out = x_normalized * scale + bias

        return out


def main():
    in0 = torch.randn(256, 896)
    scale = torch.randn(896)
    bias = torch.randn(896)

    input_list = [in0, scale, bias]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
