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

    def forward(
        self, input_tensor_or_1, input_tensor_or_2, select_tensor_1, select_tensor_2
    ):
        # Step 1: 输出shape 为 (8192)
        or_result = torch.logical_or(input_tensor_or_1, input_tensor_or_2)
        select_result = torch.where(or_result, select_tensor_1, select_tensor_2)
        # Step 2: 输出shape 为 (1)
        square_sum = torch.sum(torch.square(select_result))
        # Step 3: 输出shape 为 (1)
        final_result = torch.sqrt(square_sum)
        return final_result


def main():
    # 示例输入
    input_tensor_or_1 = torch.randn([8192]) > 0.5
    input_tensor_or_2 = torch.randn([8192]) > 0.5
    select_tensor_1 = torch.randn([8192])
    select_tensor_2 = torch.randn([8192])

    input_list = [
        input_tensor_or_1,
        input_tensor_or_2,
        select_tensor_1,
        select_tensor_2,
    ]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
