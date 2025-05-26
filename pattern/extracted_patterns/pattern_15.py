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

    def forward(self, input0, input1, input2, input3, input4):
        combined = (input0 + input1 + input2) * 0.1
        combined_fp32 = combined.float()
        mean = combined_fp32.mean(dim=-1, keepdim=True)
        var = combined_fp32.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (combined_fp32 - mean) * torch.rsqrt(var + 1e-6)

        gamma = input3
        beta = input4
        output_fp32 = normalized * gamma + beta
        return output_fp32


def main():
    batch_size = 256
    feature_dim = 48
    torch.manual_seed(2025)

    in0 = torch.randn(batch_size, feature_dim, dtype=torch.float16)
    in1 = torch.randn(batch_size, feature_dim, dtype=torch.float16)
    in2 = torch.randn(batch_size, feature_dim, dtype=torch.float16)
    in3 = torch.randn(feature_dim, dtype=torch.float32)
    in4 = torch.randn(feature_dim, dtype=torch.float32)

    input_list = [in0, in1, in2, in3, in4]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
