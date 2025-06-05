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

    def forward(self, input_tensor, sub_tensor, mul_tensor):
        # Step 1: 输出shape 为 (30720, 64)
        cast_result = input_tensor.type(torch.int32)
        # Step 2: 输出shape 为 (30720)
        sum_result = torch.sum(cast_result, dim=1)
        sum_result = sum_result.type(torch.int64)

        sub_tensor = sub_tensor.type(torch.int64)
        sub_result = torch.sub(sum_result, sub_tensor)
        sub_result = sub_result.type(torch.float32)
        # Step 3: 输出shape 为 (30720)
        final_result = torch.mul(sub_result, mul_tensor)
        return final_result


def main():
    # 示例输入
    input_tensor = torch.randn([30720, 64]) > 0.5
    sub_tensor = torch.randn([30720]) > 0.5
    mul_tensor = torch.randn([30720])

    input_list = [input_tensor, sub_tensor, mul_tensor]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
