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
        # Step 1: matmul [128, 512] x [512, 8192] = [128, 8192]
        matmul_output = torch.matmul(input1, input2)

        # Step 2: bias_add [128, 8192] + [8192] = [128, 8192]
        bias_add_output = matmul_output + input3

        # Step 3: multiply [128, 8192] x [128, 8192] = [128, 8192]
        mul_output = bias_add_output * input4

        # Step 4: sigmoid
        sigmoid_output = torch.sigmoid(mul_output)

        # Step 5: less_equal (create a condition mask)
        constant1 = torch.full((128, 8192), 0.75, device=sigmoid_output.device, dtype=sigmoid_output.dtype)
        less_equal_output = sigmoid_output <= constant1

        # Step 6: zeros_like
        zeros_like_output = torch.zeros_like(sigmoid_output)

        # Step 7: select using torch.where
        select_output = torch.where(less_equal_output, zeros_like_output, sigmoid_output)

        # Step 8: sub
        sub_output = select_output - sigmoid_output

        # Step 9: stop_gradient (detach from computation graph)
        stop_grad_output = sub_output.detach()

        # Step 10: add
        add_output = sigmoid_output + stop_grad_output

        # Step 11: reshape to [128, 64, 128]
        reshape_output = add_output.view(128, 64, 128)

        return reshape_output
    

def main():
    # 创建输入张量
    input1 = torch.randn(128, 512, requires_grad=False)
    input2 = torch.randn(512, 8192, requires_grad=False)
    input3 = torch.randn(8192, requires_grad=False)
    input4 = torch.randn(128, 8192, requires_grad=False)
    
    model = PatternModel()

    # 运行模型
    output_tensor = model(input1, input2, input3, input4)

    # 打印输出张量的形状 应该输出: torch.Size([128, 64, 128])
    default_logger.info("Output shape: %s", output_tensor.shape)
    
if __name__ == "__main__":
    main()