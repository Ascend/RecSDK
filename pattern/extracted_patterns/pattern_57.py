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

    def forward(self, input_tensor_list, value):
        # Step 1: 输出shape 为 (3072, 10, 16)
        result = torch.stack(input_tensor_list).sum(dim=0)
        # Step 2: 输出shape 为 (3072, 10, 64)
        final_result = torch.cat([result, value], dim=-1)
        return final_result


def main():
    # 示例输入
    input_tensor_list = [torch.randn(3072, 10, 16) for _ in range(6)]

    value = torch.randn(3072, 10, 48)

    input_list = [input_tensor_list, value]
    perform_test(PatternModel(), input_list)


if __name__ == "__main__":
    main()
