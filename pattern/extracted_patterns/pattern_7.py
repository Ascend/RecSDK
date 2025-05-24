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
        self.batch_norm = torch.nn.BatchNorm2d(256)

    def forward(self, input_tensor):
        # Step 1: reshape 输入为 (128, 192, 1, 256)
        reshaped_tensor = input_tensor.view(128, 192, 1, 256)

        # Step 2: BatchNorm
        bn_input = reshaped_tensor.permute(0, 3, 1, 2)  # -> (128, 256, 192, 1)
        batch_norm_tensor = self.batch_norm(bn_input)

        # Step 3: reshape 回 (128, 192, 256)
        reshaped_bn_tensor = batch_norm_tensor.permute(0, 2, 3, 1).view(128, 192, 256)

        # Step 4: 创建常量张量 (128, 192, 256)
        constant_tensor = torch.ones_like(reshaped_bn_tensor)

        # Step 5: 逐元素相减
        sub_output = reshaped_bn_tensor - constant_tensor

        # Step 6: 乘法操作
        mul_output = reshaped_bn_tensor * input_tensor

        # Step 7: 常量张量
        const2_tensor = torch.ones_like(input_tensor)

        # Step 8: 输入与常量相乘
        mul2_output = input_tensor * const2_tensor

        # Step 9: sub_output 与 mul2_output 相乘
        mul3_output = sub_output * mul2_output

        # Step 10: 加法操作
        add_output = mul_output + mul3_output

        return add_output


def main():
    # 创建输入张量
    input_tensor = torch.randn(128, 192, 256)

    input_list = [input_tensor]
    perform_test(PatternModel(), input_list)
    
if __name__ == "__main__":
    main()