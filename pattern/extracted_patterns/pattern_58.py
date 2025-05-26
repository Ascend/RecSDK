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
        super(PatternModel, self).__init__()

    def forward(self, input_tensor_mul, input_tensor_select):
        # Step 1: 输出shape 为 (64)
        mul_result = torch.mul(input_tensor_mul[0], input_tensor_mul[1]).sum(-1)
        # Step 2: 输出shape 为 (64)
        inf_r = torch.isinf(mul_result)
        nan_r = torch.isnan(mul_result)
        # Step 3: 输出shape 为 (64)
        select_r = torch.where(
            torch.logical_or(inf_r, nan_r), mul_result, input_tensor_select
        )
        # Step 4: 输出shape 为 (1)
        square_sum = torch.sum(torch.square(select_r))
        # Step 5: 输出shape 为 (1)
        final_result = torch.sqrt(square_sum)
        return final_result


def main():
    # 示例输入
    input_tensor_mul = [torch.randn(64, 64) for _ in range(2)]

    input_tensor_select = torch.randn(64)

    input_list = [input_tensor_mul, input_tensor_select]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
