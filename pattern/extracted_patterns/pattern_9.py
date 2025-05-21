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
    def forward(self, input1, input2, input3, input4):
        # Step 1: matmul [128, 128] x [128, 2560] = [128, 2560]
        matmul1 = torch.matmul(input1, input2)

        # Step 2: matmul [128, 1861] x [1861, 2560] = [128, 2560]
        matmul2 = torch.matmul(input3, input4)

        # Step 3: bias_add with broadcasted constants
        constant1 = torch.full((2560,), 3.0, device=matmul1.device, dtype=matmul1.dtype)
        bias_add1 = matmul1 + constant1

        constant2 = torch.full((2560,), 10.0, device=matmul2.device, dtype=matmul2.dtype)
        bias_add2 = matmul2 + constant2

        # Step 4: sigmoid
        sigmoid_output = torch.sigmoid(bias_add1)

        # Step 5: multiply (mul1)
        mul1 = sigmoid_output * bias_add2

        # Step 6: multiply (mul2) by scalar
        mul2 = mul1 * 5.0

        # Step 7: reduce_mean (dim=1, keepdim=True)
        reduce_mean1 = torch.mean(mul2, dim=1, keepdim=True)

        # Step 8: stop_gradient (detach from computation graph)
        stop_grad_output = reduce_mean1.detach()

        # Step 9: squared difference between mul2 and stop_grad_output
        squared_diff = (mul2 - stop_grad_output) ** 2

        # Step 10: reduce_mean again
        reduce_mean2 = torch.mean(squared_diff, dim=1)

        # Step 11: add scalar
        add_result = reduce_mean2 + 5.0

        # Step 12: rsqrt
        rsqrt_output = torch.rsqrt(add_result + 1e-8)  # Add small epsilon for numerical stability

        # Step 13: multiply by scalar
        mul3 = rsqrt_output * 11.11

        # Step 14: final output is reduce_mean1 * mul3
        final_output = reduce_mean1.squeeze() * mul3  # shape: [128]

        return final_output
    
    
def main():
    # 创建输入张量
    input1 = torch.randn(128, 128, requires_grad=False)
    input2 = torch.randn(128, 2560, requires_grad=False)
    input3 = torch.randn(128, 1861, requires_grad=False)
    input4 = torch.randn(1861, 2560, requires_grad=False)
    
    model = PatternModel()

    # 运行模型
    output_tensor = model(input1, input2, input3, input4)

    # 打印输出张量的形状 应该输出: torch.Size([128])
    default_logger.info("Output shape: %s", output_tensor.shape)
    
if __name__ == "__main__":
    main()

