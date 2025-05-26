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
        tmp0 = inputs[0].to(torch.float32)
        tmp1 = inputs[1].expand_as(tmp0[..., 0]).unsqueeze(1).expand_as(tmp0)
        tmp2 = tmp0 + tmp1

        tmp3 = inputs[2].to(torch.float32)
        tmp5 = tmp2 / tmp3
        tmp6 = torch.sigmoid(tmp5)

        tmp7 = inputs[3].unsqueeze(2).expand(-1, -1, tmp0.shape[2]).to(torch.float32)
        sign = torch.sign(tmp7)

        tmp15 = tmp6 * sign
        tmp16 = inputs[4].to(torch.float32)
        tmp17 = tmp15 * tmp16
        output0 = tmp17.sum(dim=1)

        tmp22 = inputs[5].expand_as(tmp16).to(torch.float32)
        tmp23 = tmp22 * tmp16
        output1 = tmp23.sum(dim=1)

        return output0, output1


def main():
    y_dim = 256
    x_dim = 32
    r_dim = 32

    torch.manual_seed(2025)
    in0 = torch.randn(y_dim, r_dim, x_dim) * 2
    in1 = torch.randn(x_dim)
    in2 = torch.tensor([0.5])
    in3 = torch.randn(y_dim, r_dim)
    in4 = torch.randn(y_dim, r_dim, x_dim)
    in5 = torch.randn(r_dim, x_dim)

    input_list = [[in0, in1, in2, in3, in4, in5]]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
