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

from utils.logger import default_logger


class PatternModel(torch.nn.Module):
    def forward(self, input1, input2, input3):
        # Step 1: Add two tensors
        add_output = input1 + input2  # shape: (128, 150, 148)

        # Step 2: Reduce mean on last dim
        reduce_output = add_output.mean(dim=-1, keepdim=True)  # shape: (128, 150, 1)

        # Step 3: Subtract mean
        sub_output = add_output - reduce_output  # shape: (128, 150, 148)

        # Step 4: Square the difference
        square_output = (sub_output) ** 2  # shape: (128, 150, 148)

        # Step 5: Reduce mean again
        reduce2_output = square_output.mean(dim=-1, keepdim=True)  # shape: (128, 150, 1)

        # Step 6: Add to input3
        add2_output = reduce2_output + input3  # shape: (128, 150, 1)

        # Step 7: Reciprocal square root
        rsqrt_output = torch.rsqrt(add2_output)  # shape: (128, 150, 1)

        # Step 8: Multiply with sub_output
        mul_output = sub_output * rsqrt_output  # shape: (128, 150, 148)

        # Step 9: Broadcast multiply and add constant
        const_tensor1 = torch.tensor(1.0, device=mul_output.device, dtype=mul_output.dtype)
        const_tensor2 = torch.tensor(2.0, device=mul_output.device, dtype=mul_output.dtype)

        mul2_output = mul_output * const_tensor1  # shape: (128, 150, 148)
        add3_output = mul2_output + const_tensor2  # shape: (128, 150, 148)

        return add3_output


def main():
    # 创建输入张量
    input1 = torch.randn(128, 150, 148)
    input2 = torch.randn(128, 150, 148)
    input3 = torch.randn(128, 150, 1)

    # 调用函数
    model = PatternModel()
    output_tensor = model(input1, input2, input3)

    # 打印输出形状 应该输出: torch.Size([128, 150, 148])
    default_logger.info("Output shape: %s", output_tensor.shape)

if __name__ == "__main__":
    main()